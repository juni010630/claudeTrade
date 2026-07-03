"""ccxt 기반 Binance OHLCV + 펀딩비 다운로더."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVFetcher:
    """ccxt Binance 선물에서 OHLCV 데이터를 페이지 단위로 가져온다."""

    COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, exchange_id: str = "binanceusdm") -> None:
        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(
            {"enableRateLimit": True}
        )

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since: datetime,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """since ~ until 범위의 OHLCV를 DataFrame으로 반환."""
        since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
        until_ms = (
            int(until.replace(tzinfo=timezone.utc).timestamp() * 1000)
            if until
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        all_rows: list[list] = []
        cursor = since_ms

        while cursor < until_ms:
            rows = self.exchange.fetch_ohlcv(
                symbol, timeframe, since=cursor, limit=limit
            )
            if not rows:
                break
            all_rows.extend(rows)
            cursor = rows[-1][0] + 1
            if len(rows) < limit:
                break
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_rows:
            return pd.DataFrame(columns=self.COLUMNS)

        df = pd.DataFrame(all_rows, columns=self.COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[df["timestamp"] < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
        # 마지막 봉이 미완성일 수 있으므로 제거 (현재 시각 기준)
        now = pd.Timestamp.now(tz="UTC")
        tf_mins = {"1m":1,"5m":5,"15m":15,"1h":60,"4h":240,"1d":1440}.get(timeframe, 60)
        current_open = now.floor(f"{tf_mins}min")
        df = df[df["timestamp"] < current_open]
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        # short-page 조기종료 검증: 요청 범위 대비 마지막 봉이 1봉 이상 못 미치면 경고 —
        # 거래소가 일시적으로 limit 미만 페이지를 주면 잔여 범위가 무음 스킵돼 캐시 갭이 됨
        if not df.empty:
            tf_ms = tf_mins * 60_000
            expected_last = (min(until_ms, int(current_open.value // 1_000_000)) - tf_ms)
            actual_last = int(df["timestamp"].iloc[-1].value // 1_000_000)
            if expected_last - actual_last >= tf_ms:
                logger.warning(
                    "%s %s 다운로드 조기종료 의심: 마지막 봉 %s < 기대 %s — 재실행 권장",
                    symbol, timeframe, df["timestamp"].iloc[-1],
                    pd.Timestamp(expected_last, unit="ms", tz="UTC"),
                )
        return df


class FundingRateFetcher:
    """Binance 선물 펀딩비 이력 다운로더."""

    def __init__(self, exchange_id: str = "binanceusdm") -> None:
        self.exchange: ccxt.Exchange = getattr(ccxt, exchange_id)(
            {"enableRateLimit": True}
        )

    def fetch(
        self,
        symbol: str,
        since: datetime,
        until: datetime | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
        until_ms = (
            int(until.replace(tzinfo=timezone.utc).timestamp() * 1000)
            if until
            else int(datetime.now(timezone.utc).timestamp() * 1000)
        )

        all_rows: list[dict] = []
        cursor = since_ms

        while cursor < until_ms:
            rows = self.exchange.fetch_funding_rate_history(
                symbol, since=cursor, limit=limit
            )
            if not rows:
                break
            all_rows.extend(rows)
            cursor = rows[-1]["timestamp"] + 1
            if len(rows) < limit:
                break
            time.sleep(self.exchange.rateLimit / 1000)

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "fundingRate"])

        df = pd.DataFrame(all_rows)[["timestamp", "symbol", "fundingRate"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[df["timestamp"] < pd.Timestamp(until_ms, unit="ms", tz="UTC")]
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df = df.rename(columns={"fundingRate": "rate"})
        return df
