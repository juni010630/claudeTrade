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
        """모든 열린 포지션에 대해 직전 5m 봉 high/low로 SL 터치 체크."""
        with self.lock:
            state = self.engine.tracker.snapshot()
            syms = list(state.positions.keys())
        if not syms:
            return

        for sym in syms:
            try:
                ohlcv = self.exchange.fetch_ohlcv(sym, self.tf, limit=3)
            except Exception as e:
                logger.warning("SL poller OHLCV 실패 %s: %s", sym, e)
                continue
            if not ohlcv:
                continue
            # 마지막 완성봉: 마지막이 미완성이면 두 번째부터
            # 보수적으로 뒤에서 두 번째 사용 (완전히 마감된 봉)
            last = ohlcv[-2] if len(ohlcv) >= 2 else ohlcv[-1]
            ts_ms, o, h, l, c, v = last
            bar_ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")

            # lock 내에서 SL 히트 여부만 판단, 네트워크 호출은 밖에서
            close_info = None
            with self.lock:
                st = self.engine.tracker.snapshot()
                pos = st.positions.get(sym)
                if pos is None:
                    continue
                # 진입 직후 같은 봉이면 SL 체크 건너뜀 (같은 봉 noise)
                if bar_ts <= pos.opened_at:
                    continue

                sl = float(pos.sl_price)
                hit_sl = (pos.direction == "long" and float(l) <= sl) or \
                         (pos.direction == "short" and float(h) >= sl)
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
                    "bar_l": float(l), "bar_h": float(h),
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
                slip = self.broker.slippage.cost(
                    exit_notional, OrderType.MARKET
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
                # circuit breaker 기록
                last_trade = self.engine.ledger._records[-1] if self.engine.ledger._records else None
                if last_trade is not None:
                    self.engine.circuit_breaker.record_result(
                        close_info["strategy"], last_trade.pnl > 0
                    )
