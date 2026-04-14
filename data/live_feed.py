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

    TF_MINS = {"1h": 60, "4h": 240, "1d": 1440}

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        primary_tf: str = "1h",
        lookback: int = 300,
        exchange_id: str = "binanceusdm",
        demo: bool = True,
    ) -> None:
        self.symbols    = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.lookback   = lookback

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

    # ── 펀딩비 조회 ───────────────────────────────────────────────────
    def _fetch_funding(self, symbol: str) -> float:
        try:
            info = self.exchange.fetch_funding_rate(symbol)
            return float(info.get("fundingRate") or 0.0)
        except Exception:
            return 0.0

    # ── 현재 스냅샷 즉시 생성 ─────────────────────────────────────────
    def snapshot_now(self) -> MarketSnapshot:
        """지금 당장 MarketSnapshot 을 만들어 반환."""
        ts = pd.Timestamp.now(tz="UTC").floor("1h")
        bars: dict[str, dict[str, pd.DataFrame]] = {}
        funding: dict[str, float] = {}

        for sym in self.symbols:
            bars[sym] = {}
            for tf in self.timeframes:
                df = self._fetch_bars(sym, tf)
                # 현재 미완성 봉(마지막) 제외하고 lookback 개만 사용
                bars[sym][tf] = df.iloc[:-1].iloc[-self.lookback:].reset_index()
            funding[sym] = self._fetch_funding(sym)
            time.sleep(self.exchange.rateLimit / 1000 * 0.5)

        return MarketSnapshot(
            timestamp=ts,
            bars=bars,
            funding_rates=funding,
            open_interest={},
            btc_dominance=0.0,
        )

    # ── 다음 1h 봉 마감까지 대기 ─────────────────────────────────────
    @staticmethod
    def _seconds_to_next_close(tf_minutes: int = 60, margin_sec: int = 5) -> float:
        """다음 봉 마감 후 margin_sec 초 시점까지 남은 시간(초)."""
        now_ts = time.time()
        tf_sec = tf_minutes * 60
        elapsed = now_ts % tf_sec
        return tf_sec - elapsed + margin_sec

    def wait_and_snap(self) -> MarketSnapshot:
        """다음 1h 봉 마감을 기다렸다가 스냅샷 반환."""
        wait = self._seconds_to_next_close(
            self.TF_MINS.get(self.primary_tf, 60)
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
