"""멀티 심볼/타임프레임 DataLoader — MarketSnapshot 이터레이터."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from data.cache import ParquetCache
from data.schemas import MarketSnapshot


class DataLoader:
    """
    캐시에서 데이터를 읽어 타임스탬프별로 MarketSnapshot을 생성한다.
    기준 타임프레임(primary_tf)의 봉 마감 시각을 이터레이션 기준으로 삼는다.
    """

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        primary_tf: str = "1h",
        cache_dir: str | Path = "data/cache",
        lookback: int = 300,  # 각 타임스탬프에서 제공할 과거 봉 수
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes
        self.primary_tf = primary_tf
        self.cache = ParquetCache(cache_dir)
        self.lookback = lookback

        # 전체 데이터 로드
        self._ohlcv: dict[str, dict[str, pd.DataFrame]] = {}
        self._funding: dict[str, pd.DataFrame] = {}

        for sym in symbols:
            self._ohlcv[sym] = {}
            for tf in timeframes:
                df = self.cache.load(sym, tf)
                if df is None:
                    raise FileNotFoundError(
                        f"캐시 없음: {sym} {tf}. scripts/fetch_data.py 먼저 실행하세요."
                    )
                self._ohlcv[sym][tf] = df.set_index("timestamp")

            funding_df = self.cache.load(sym, "8h", data_type="funding")
            self._funding[sym] = (
                funding_df.set_index("timestamp") if funding_df is not None else pd.DataFrame()
            )

    def iterate(
        self,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
    ) -> Iterator[MarketSnapshot]:
        """기준 타임프레임의 각 봉에 대해 MarketSnapshot을 yield."""
        primary_sym = self.symbols[0]
        primary_df = self._ohlcv[primary_sym][self.primary_tf]
        all_timestamps = primary_df.index

        # lookback 기준은 전체 데이터에서의 위치 (since/until 필터 전)
        min_idx = self.lookback
        if since is not None:
            # since 이전에 lookback만큼 데이터가 있는지 확인
            first_valid = all_timestamps.searchsorted(since)
            min_idx = max(min_idx, first_valid)

        timestamps = all_timestamps[min_idx:]
        if since is not None:
            timestamps = timestamps[timestamps >= since]
        if until is not None:
            timestamps = timestamps[timestamps < until]

        for ts in timestamps:

            bars: dict[str, dict[str, pd.DataFrame]] = {}
            # ts = primary_tf 봉의 open time. effective time = ts + primary_delta (봉 close 시점).
            # 각 TF의 봉은 close_time(= open + tf_delta) ≤ effective_time 인 것만 포함.
            # 이래야 미마감 4h/1d 봉의 미래 데이터를 사용하지 않음.
            primary_delta = pd.Timedelta(self.primary_tf)
            effective_time = ts + primary_delta
            for sym in self.symbols:
                bars[sym] = {}
                for tf in self.timeframes:
                    df = self._ohlcv[sym][tf]
                    tf_delta = pd.Timedelta(tf)
                    # 봉의 close time = open time + tf_delta
                    mask = (df.index + tf_delta) <= effective_time
                    sub = df[mask].iloc[-self.lookback :]
                    bars[sym][tf] = sub.reset_index()

            funding_rates: dict[str, float] = {}
            for sym in self.symbols:
                fd = self._funding[sym]
                if not fd.empty:
                    mask = fd.index <= effective_time
                    recent = fd[mask]
                    funding_rates[sym] = float(recent["rate"].iloc[-1]) if not recent.empty else 0.0
                else:
                    funding_rates[sym] = 0.0

            yield MarketSnapshot(
                timestamp=effective_time,  # close time 기준 (LiveFeed와 일치)
                bars=bars,
                funding_rates=funding_rates,
                open_interest={},   # 필요 시 별도 캐시에서 로드
                btc_dominance=0.0,
            )
