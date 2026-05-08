"""ccxt 기반 Binance OHLCV + 펀딩비 다운로더."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import ccxt
import pandas as pd


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
