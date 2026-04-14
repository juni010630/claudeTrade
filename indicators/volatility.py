"""변동성 지표 — 순수 함수."""
from __future__ import annotations

import pandas as pd


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(upper, mid, lower) 반환."""
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std(ddof=0)
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def bb_width(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.Series:
    """밴드 폭 = (upper - lower) / mid. 작을수록 수렴 구간."""
    upper, mid, lower = bollinger_bands(df, period, std_dev, col)
    return (upper - lower) / mid.replace(0, float("nan"))
