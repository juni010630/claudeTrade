"""실시간 MarketSnapshot 피드 — 1h 봉 마감 시 스냅샷 생성."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from data.schemas import MarketSnapshot

logger = logging.getLogger(__name__)


class LiveFeed:
    """
    ccxt 기반 실시간 OHLCV 피드.

    - 1h 봉이 마감될 때마다 MarketSnapshot 을 생성한다.
    - 각 MarketSnapshot 은 lookback 개 봉을 포함한다.
    - exchange 는 인증 불필요 (퍼블릭 엔드포인트만 사용).
    """

    TF_MINS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        primary_tf: str = "1h",
        lookback: int = 300,
        exchange_id: str = "binanceusdm",
        demo: bool = True,
        notifier=None,
        funding_history_symbols: list[str] | None = None,
        funding_history_lookback: int = 180,
    ) -> None:
        self.symbols    = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.lookback   = lookback
        self.notifier   = notifier
        self.funding_history_symbols = set(funding_history_symbols or ())
        self.funding_history_lookback = int(funding_history_lookback)
        self._funding_history: dict[str, pd.DataFrame] = {}
        self._funding_history_seeded: set[str] = set()

        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(
            {"enableRateLimit": True}
        )
        if demo:
            self.exchange.set_sandbox_mode(True)

    # ── 단일 심볼/TF OHLCV 조회 ──────────────────────────────────────
    def _fetch_bars(self, symbol: str, tf: str) -> pd.DataFrame:
        """최근 lookback+1 개 봉 조회."""
        try:
            raw = self.exchange.fetch_ohlcv(symbol, tf, limit=self.lookback + 1)
        except Exception as e:
            logger.warning("OHLCV 조회 실패 %s %s: %s", symbol, tf, e)
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")

    # ── 기대 봉 freshness 검증 ───────────────────────────────────────
    def _expected_last_closed_bar(self, tf: str) -> pd.Timestamp:
        """tf 기준 '지금 기준 가장 최근에 마감된 봉의 open_time'."""
        tf_mins = self.TF_MINS.get(tf, 60)
        now = pd.Timestamp.now(tz="UTC")
        current_open = now.floor(f"{tf_mins}min")
        return current_open - pd.Timedelta(minutes=tf_mins)

    def _fetch_bars_with_freshness(
        self, symbol: str, tf: str,
        max_retries: int = 15, retry_wait: float = 2.0,
    ) -> pd.DataFrame:
        """fetch_bars + 기대 마지막봉 timestamp 검증 + 재시도.
        거래소 지연으로 n번째 봉이 아직 안 올라온 경우 retry_wait초 대기 후 재시도.
        모두 실패 시 마지막 결과 반환(오래된 데이터라도 다운스트림에서 처리)."""
        expected_last = self._expected_last_closed_bar(tf)
        last_df = None
        for attempt in range(max_retries):
            df = self._fetch_bars(symbol, tf)
            last_df = df
            if df.empty:
                time.sleep(retry_wait)
                continue
            # 미완성 봉 포함되어 있으면 마지막은 현재 open, 그 앞이 기대 마감봉
            last_ts = df.index[-1]
            # 마지막 봉이 완성봉인지(현재 open 이전) 아니면 미완성인지에 따라 expected 위치 다름
            candidate = last_ts if last_ts <= expected_last else (
                df.index[-2] if len(df) >= 2 else None
            )
            if candidate is not None and candidate >= expected_last:
                return df
            logger.warning(
                "[%s %s] 기대 마감봉=%s, 실제 최근=%s → %.0f초 대기 재시도 (%d/%d)",
                symbol, tf, expected_last, last_ts, retry_wait, attempt + 1, max_retries,
            )
            time.sleep(retry_wait)
        msg = f"[{symbol} {tf}] {max_retries}회 시도 후에도 최신 봉 미수신 (기대: {expected_last})"
        logger.error(msg + " → 오래된 데이터로 진행")
        if self.notifier is not None and getattr(self.notifier, "enabled", False):
            try:
                self.notifier.notify_info(f"⚠️ 봉 수신 실패\n{msg}")
            except Exception:
                pass
        return last_df if last_df is not None else pd.DataFrame(
            columns=["timestamp","open","high","low","close","volume"]
        )

    # ── 펀딩비 조회 ───────────────────────────────────────────────────
    def _fetch_funding(self, symbol: str) -> tuple[float, pd.Timestamp | None]:
        # 백테 패리티: 백테 캐시는 '직전 정산 확정 rate'의 as-of 룩업 — 라이브도 동일
        # 의미로 정렬. fetch_funding_rate는 차기 정산 예측치라 scorer 펀딩 점수(±0.0003
        # 임계)가 백테와 갈라진다. 실패 시 예측치 → 0.0 순 폴백.
        # 반환 = (rate, 정산 시각) — 정산 시각은 FundingRateSimulator의 정산 발생
        # 감지용(4h/8h 심볼별 주기 자동). 폴백 경로는 ts 없음 → 시뮬레이터가 8h 버킷.
        try:
            hist = self.exchange.fetch_funding_rate_history(symbol, limit=1)
            if hist:
                ms = hist[-1].get("timestamp")
                fts = pd.Timestamp(ms, unit="ms", tz="UTC") if ms else None
                return float(hist[-1].get("fundingRate") or 0.0), fts
        except Exception:
            pass
        try:
            info = self.exchange.fetch_funding_rate(symbol)
            return float(info.get("fundingRate") or 0.0), None
        except Exception:
            return 0.0, None

    def _seed_funding_history(self, symbol: str) -> pd.DataFrame:
        """확정 정산 이력만 1회 로드. 실패 시 빈 이력을 반환해 D를 no-op 처리한다."""
        if symbol in self._funding_history_seeded:
            return self._funding_history.get(
                symbol, pd.DataFrame(columns=["timestamp", "rate"])
            )
        rows = []
        fetched = False
        try:
            raw = self.exchange.fetch_funding_rate_history(
                symbol, limit=max(self.funding_history_lookback, 90)
            )
            fetched = True
            for item in raw or ():
                ms = item.get("timestamp")
                if ms is None:
                    continue
                rows.append({
                    "timestamp": pd.Timestamp(ms, unit="ms", tz="UTC"),
                    "rate": float(item.get("fundingRate") or 0.0),
                })
        except Exception as error:
            logger.warning("Candidate D 펀딩 이력 로드 실패 %s: %s", symbol, error)
        history = pd.DataFrame(rows, columns=["timestamp", "rate"])
        existing = self._funding_history.get(symbol)
        if existing is not None and not existing.empty:
            history = pd.concat([existing, history], ignore_index=True)
        if not history.empty:
            history = (
                history.drop_duplicates("timestamp", keep="last")
                .sort_values("timestamp")
                .tail(self.funding_history_lookback)
                .reset_index(drop=True)
            )
        self._funding_history[symbol] = history
        if fetched:
            self._funding_history_seeded.add(symbol)
        return history

    def _update_funding_history(
        self, symbol: str, rate: float, timestamp: pd.Timestamp | None,
    ) -> None:
        history = self._seed_funding_history(symbol)
        if timestamp is None:
            return  # 차기 정산 예측치는 확정 이력에 섞지 않는다.
        timestamp = pd.Timestamp(timestamp)
        timestamp = (
            timestamp.tz_localize("UTC") if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        addition = pd.DataFrame([{"timestamp": timestamp, "rate": float(rate)}])
        self._funding_history[symbol] = (
            pd.concat([history, addition], ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .tail(self.funding_history_lookback)
            .reset_index(drop=True)
        )

    # ── 현재 스냅샷 즉시 생성 ─────────────────────────────────────────
    def snapshot_now(self) -> MarketSnapshot:
        """지금 당장 MarketSnapshot 을 만들어 반환."""
        bars: dict[str, dict[str, pd.DataFrame]] = {}
        funding: dict[str, float] = {}
        funding_ts: dict[str, pd.Timestamp] = {}
        funding_history: dict[str, pd.DataFrame] = {}
        pf_closes: dict[str, pd.Timestamp] = {}  # 심볼별 primary 최신 완성봉 close

        for sym in self.symbols:
            bars[sym] = {}
            for tf in self.timeframes:
                df = self._fetch_bars_with_freshness(sym, tf)
                # 마지막 봉이 아직 미완성(close time이 미래)일 때만 제외.
                # 경계 직후(HH:00:0X) fetch 시 거래소가 forming 봉을 아직 안 주면 df의
                # 마지막은 '직전 완성봉'이므로 무조건 떨구면 snapshot이 1h stale이 됨
                # (봉 #N 타임스탬프가 1시간 밀려 슬리브 hour==0 게이트가 안 열림). 완성봉이면 유지.
                # (빈 검사보다 먼저 — 1행 프레임의 forming 봉이 완성봉처럼 통과하지 않도록)
                tf_mins = self.TF_MINS.get(tf, 60)
                if not df.empty:
                    last_close = df.index[-1] + pd.Timedelta(minutes=tf_mins)
                    if last_close > pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=30):
                        df = df.iloc[:-1]  # forming 봉만 제외 (30s = 시계오차 여유)
                if df.empty:
                    logger.warning("빈 데이터: %s %s — skip", sym, tf)
                    bars[sym][tf] = df.reset_index() if df.index.name == "timestamp" else df
                    continue
                closed = df.iloc[-self.lookback:]
                bars[sym][tf] = closed.reset_index()
                if tf == self.primary_tf:
                    pf_closes[sym] = closed.index[-1] + pd.Timedelta(minutes=tf_mins)
            funding[sym], _fts = self._fetch_funding(sym)
            if _fts is not None:
                funding_ts[sym] = _fts
            if sym in self.funding_history_symbols:
                self._update_funding_history(sym, funding[sym], _fts)
            time.sleep(self.exchange.rateLimit / 1000 * 0.5)

        # 타임스탬프 앵커 = 전 심볼 primary 완성봉 close의 최대값.
        # 첫 심볼(ETH) 단독 앵커는 그 심볼이 1봉 stale이면 전 심볼 hour 게이트/펀딩
        # 버킷이 1시간 밀리고 stale 제외 검사도 무력화된다 — max면 양쪽 다 견고.
        snap_ts = max(pf_closes.values()) if pf_closes else None

        # 심볼별 staleness 방어: 한 심볼의 primary_tf 최신 완성봉이 snap_ts보다 1봉
        # 이상 뒤처지면(15회 재시도 실패로 stale fetch) 그 심볼을 snapshot에서 제외 —
        # stale 가격에 신호/MTM이 작동하지 않게 한다. 전역 snapshot staleness는
        # live_trade가 별도(7200s)로 가드한다.
        if snap_ts is not None:
            pf = self.primary_tf
            pf_mins = self.TF_MINS.get(pf, 60)
            tol = pd.Timedelta(minutes=pf_mins)
            for sym in list(bars.keys()):
                pdf = bars[sym].get(pf)
                if pdf is None or len(pdf) == 0:
                    # primary 프레임이 비면(30초 연속 fetch 실패) 심볼 제외 — 남기면
                    # 엔진 _get_bars가 4h/1d 봉으로 폴백해 진입 전 가격대의 high/low로
                    # TP/SL을 오판정, 실포지션을 오청산한다.
                    logger.warning("심볼 제외: %s primary(%s) 데이터 없음", sym, pf)
                    bars.pop(sym, None)
                    funding.pop(sym, None)
                    funding_ts.pop(sym, None)
                    continue
                pf_close = pdf["timestamp"].iloc[-1] + pd.Timedelta(minutes=pf_mins)
                if snap_ts - pf_close >= tol:  # 1봉 이상 뒤처지면 제외 (== 1봉도 stale)
                    logger.warning(
                        "심볼 stale 제외: %s primary 최신봉 close=%s vs snap=%s",
                        sym, pf_close, snap_ts,
                    )
                    bars.pop(sym, None)
                    funding.pop(sym, None)
                    funding_ts.pop(sym, None)
                    continue
                # 비-primary TF(4h/1d) stale 검사: 기대 마지막 close보다 뒤처지면 그 TF
                # 프레임만 비운다 — "신선한 1h 트리거 + 낡은 상위TF 필터"라는 백테에 없는
                # 신호 조합 차단. 1h 기반 TP/SL·MTM은 유지(전략 가드가 빈 프레임=무신호).
                for tf in self.timeframes:
                    if tf == pf:
                        continue
                    tdf = bars[sym].get(tf)
                    if tdf is None or len(tdf) == 0:
                        continue
                    tf_mins = self.TF_MINS.get(tf, 60)
                    expected_close = snap_ts.floor(f"{tf_mins}min")
                    tf_close = tdf["timestamp"].iloc[-1] + pd.Timedelta(minutes=tf_mins)
                    if tf_close < expected_close:
                        logger.warning("stale %s 프레임 제외: %s close=%s < 기대 %s",
                                       tf, sym, tf_close, expected_close)
                        bars[sym][tf] = tdf.iloc[0:0]

        # 폴백: 타임스탬프를 결정 못한 경우 현재 시각 floor
        if snap_ts is None:
            snap_ts = pd.Timestamp.now(tz="UTC").floor(
                f"{self.TF_MINS.get(self.primary_tf, 60)}min"
            )

        # evaluator에 스냅샷 시각 이후의 정산 행은 전달하지 않는다.
        for sym in self.funding_history_symbols:
            history = self._funding_history.get(sym)
            if history is not None and not history.empty:
                funding_history[sym] = history.loc[
                    pd.to_datetime(history["timestamp"], utc=True) <= snap_ts
                ].copy()

        return MarketSnapshot(
            timestamp=snap_ts,
            bars=bars,
            funding_rates=funding,
            funding_ts=funding_ts,
            funding_history=funding_history,
            open_interest={},
            btc_dominance=0.0,
        )

    # ── 다음 1h 봉 마감까지 대기 ─────────────────────────────────────
    @staticmethod
    def _seconds_to_next_close(tf_minutes: int = 60, margin_sec: int = 10) -> float:
        """다음 봉 마감 후 margin_sec 초 시점까지 남은 시간(초)."""
        now_ts = time.time()
        tf_sec = tf_minutes * 60
        elapsed = now_ts % tf_sec
        return tf_sec - elapsed + margin_sec

    def wait_and_snap(self) -> MarketSnapshot:
        """다음 1h 봉 마감을 기다렸다가 스냅샷 반환."""
        wait = self._seconds_to_next_close(
            self.TF_MINS.get(self.primary_tf, 60),
            margin_sec=1
        )
        logger.info("다음 봉 마감까지 %.0f초 대기...", wait)
        time.sleep(wait)
        return self.snapshot_now()

    # ── 무한 이터레이터 ──────────────────────────────────────────────
    def stream(self):
        """1h 봉마다 MarketSnapshot 을 yield 하는 무한 제너레이터."""
        logger.info("LiveFeed 시작: %s | TF=%s | lookback=%d",
                    self.symbols, self.timeframes, self.lookback)
        while True:
            yield self.wait_and_snap()
