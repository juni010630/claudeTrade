"""Parquet 기반 로컬 캐시."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


class ParquetCache:
    """심볼/타임프레임별 Parquet 파일 캐시."""

    def __init__(self, base_dir: str | Path = "data/cache") -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: str, data_type: str = "ohlcv") -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.base / f"{data_type}_{safe_symbol}_{timeframe}.parquet"

    def save(
        self, df: pd.DataFrame, symbol: str, timeframe: str, data_type: str = "ohlcv"
    ) -> None:
        path = self._path(symbol, timeframe, data_type)
        if path.exists():
            existing = pd.read_parquet(path)
            df = (
                pd.concat([existing, df])
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        df.to_parquet(path, index=False)

    def load(
        self,
        symbol: str,
        timeframe: str,
        since: pd.Timestamp | None = None,
        until: pd.Timestamp | None = None,
        data_type: str = "ohlcv",
    ) -> pd.DataFrame | None:
        path = self._path(symbol, timeframe, data_type)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if since is not None:
            df = df[df["timestamp"] >= since]
        if until is not None:
            df = df[df["timestamp"] < until]
        return df.reset_index(drop=True) if not df.empty else None

    def exists(self, symbol: str, timeframe: str, data_type: str = "ohlcv") -> bool:
        return self._path(symbol, timeframe, data_type).exists()
