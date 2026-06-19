"""5분 간격 SL 폴러.

Testnet이 STOP_MARKET을 막아놓은 경우, 1h 봉마다만 SL을 체크하면
최대 1시간 지연. 이를 5분으로 단축하기 위해 별도 스레드에서
거래소의 5m 봉 high/low를 폴링하여 SL 터치 시 즉시 market_close.

스레드 안전성: `lock`을 engine._process_bar 실행 시에도 잡아 배타적 접근.
"""
from __future__ import annotations

import logging
import threading
import time

import pandas as pd

from execution.models import OrderType
from regime.models import MarketRegime

logger = logging.getLogger(__name__)


class SLPoller:
    def __init__(
        self,
        engine,
        broker,
        exchange,
        interval_sec: int = 300,
        tf: str = "5m",
        lock: threading.Lock | None = None,
    ) -> None:
        self.engine = engine
        self.broker = broker
        self.exchange = exchange
        self.interval_sec = interval_sec
        self.tf = tf
        self.lock = lock or threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # 심볼별 마지막으로 검사한 완성봉 open_time — 폴 간격 드리프트로 봉을 건너뛰지 않도록.
        self._last_seen: dict[str, pd.Timestamp] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="sl-poller")
        self._thread.start()
        logger.info("SL poller 시작 (interval=%ds, tf=%s)", self.interval_sec, self.tf)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            # 열린 포지션이 있을 때만 폴링 (자원 절약)
            has_positions = False
            with self.lock:
                has_positions = bool(self.engine.tracker.snapshot().positions)
            if has_positions:
                try:
                    self.check_once()
                except Exception as e:
                    logger.exception("SL poller 예외: %s", e)
                self._stop.wait(self.interval_sec)
            else:
                # 포지션 없으면 30초 간격으로 가볍게 재확인만
                self._stop.wait(30)

    def check_once(self) -> None:
        """모든 열린 포지션에 대해 마지막 폴 이후 마감된 5m 봉들의 high/low로 SL 터치 체크.

        폴 간격(300s)이 check_once 소요시간만큼 5m 그리드 대비 드리프트하므로, 한 폴에
        2개 이상 봉이 마감될 수 있다. 직전 1봉만 보면 그 사이 봉의 SL 터치를 놓친다 →
        last_seen 이후의 모든 완성봉을 검사(extreme low/high 집계)해 누락을 방지한다.
        """
        with self.lock:
            state = self.engine.tracker.snapshot()
            syms = list(state.positions.keys())
        if not syms:
            return

        now = pd.Timestamp.now(tz="UTC")
        tf_delta = pd.Timedelta(self.tf)
        for sym in syms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(sym, self.tf, limit=5)
            except Exception as e:
                logger.warning("SL poller OHLCV 실패 %s: %s", sym, e)
                continue
            if not ohlcv:
                continue
            # last_seen 이후의 완성봉만 수집 (forming 봉 = close_time이 미래 → 제외)
            last_seen = self._last_seen.get(sym)
            new_bars = []  # (bar_ts, high, low)
            for ts_ms, o, h, l, c, v in ohlcv:
                bar_ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
                if bar_ts + tf_delta > now:
                    continue  # 미마감(forming) 봉
                if last_seen is not None and bar_ts <= last_seen:
                    continue  # 이미 검사한 봉
                new_bars.append((bar_ts, float(h), float(l)))
            if not new_bars:
                continue
            self._last_seen[sym] = new_bars[-1][0]

            # lock 내에서 SL 히트 여부만 판단, 네트워크 호출은 밖에서
            close_info = None
            with self.lock:
                st = self.engine.tracker.snapshot()
                pos = st.positions.get(sym)
                if pos is None:
                    continue
                # 진입 봉 이후의 완성봉만 (같은 봉 noise 제외)
                relevant = [(h, l) for (bts, h, l) in new_bars if bts > pos.opened_at]
                if not relevant:
                    continue
                bar_h = max(h for h, l in relevant)
                bar_l = min(l for h, l in relevant)

                sl = float(pos.sl_price)
                hit_sl = (pos.direction == "long" and bar_l <= sl) or \
                         (pos.direction == "short" and bar_h >= sl)
                if not hit_sl:
                    continue

                # SL 히트 확인 — 청산 정보 저장
                qty = pos.size_usd / pos.entry_price
                close_info = {
                    "sym": sym, "direction": pos.direction, "qty": qty,
                    "sl": sl, "size_usd": pos.size_usd,
                    "entry_price": pos.entry_price,
                    "strategy": pos.strategy,
                    "confluence_score": pos.confluence_score,
                    "bar_l": bar_l, "bar_h": bar_h,
                }

            if close_info is None:
                continue

            # 네트워크 호출은 lock 밖에서 수행 (메인 스레드 블로킹 방지)
            logger.warning(
                "[SL-POLL] %s %s SL 히트 (sl=%.4f bar_l=%.4f bar_h=%.4f) → market_close",
                close_info["sym"], close_info["direction"],
                close_info["sl"], close_info["bar_l"], close_info["bar_h"],
            )
            try:
                self.broker.market_close(
                    close_info["sym"], close_info["direction"], close_info["qty"]
                )
            except Exception as e:
                logger.error("SL market_close 실패 %s: %s", close_info["sym"], e)
                continue

            # 실제 체결가 조회 (없으면 SL 가격 폴백)
            actual_fill = None
            if hasattr(self.broker, "fetch_recent_fill_price"):
                actual_fill = self.broker.fetch_recent_fill_price(close_info["sym"])
            exit_price = actual_fill if actual_fill is not None else close_info["sl"]

            # lock 재획득 후 tracker 업데이트
            with self.lock:
                # 이미 다른 경로로 청산되었을 수 있음 — 방어
                st = self.engine.tracker.snapshot()
                cur_pos = st.positions.get(close_info["sym"])
                if cur_pos is None:
                    continue
                # 경합 방어: 포지션이 교체되었으면 (방향/진입가 다름) 건너뜀
                if (cur_pos.direction != close_info["direction"]
                        or cur_pos.entry_price != close_info.get("entry_price", cur_pos.entry_price)):
                    logger.warning("[SL-POLL] %s 포지션 교체 감지 — tracker 업데이트 건너뜀", close_info["sym"])
                    continue

                exit_notional = close_info["size_usd"] / close_info.get("entry_price", exit_price) * exit_price
                commission = self.broker.commission.calculate(
                    exit_notional, OrderType.MARKET
                )
                # actual_fill(실제 체결가)엔 슬리피지가 이미 반영됨 → 이중부과 방지.
                # SL 가격 폴백일 때만 이론 슬리피지 부과 (live_broker.submit과 동일 관례).
                slip = (
                    0.0 if actual_fill is not None
                    else self.broker.slippage.cost(exit_notional, OrderType.MARKET)
                )
                exit_regime = MarketRegime.RANGING

                self.engine.tracker.close_position(
                    symbol=close_info["sym"],
                    exit_price=exit_price,
                    exit_time=pd.Timestamp.now(tz="UTC"),
                    exit_reason="sl_polled",
                    regime=exit_regime,
                    confluence_score=close_info["confluence_score"],
                    commission=commission,
                    slippage_cost=slip,
                )
                # circuit breaker 기록 — 격리 북(macross_d 등)은 제외(엔진 본경로와 동일
                # 규칙: 격리 북 손익이 글로벌 CB 카운터를 오염시켜 추세/슬리브 게이팅을
                # 잘못 트립/리셋하는 것 방지).
                last_trade = self.engine.ledger._records[-1] if self.engine.ledger._records else None
                if (last_trade is not None
                        and close_info["strategy"] not in self.engine._strategy_guard_isolated):
                    self.engine.circuit_breaker.record_result(
                        close_info["strategy"], last_trade.pnl > 0
                    )
