"""Donchian Channel 지표."""
from __future__ import annotations

import pandas as pd


def donchian_channels(
    df: pd.DataFrame, period: int = 20
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (upper, mid, lower).
      upper = period 최고가
      lower = period 최저가
      mid   = (upper + lower) / 2
    """
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    mid   = (upper + lower) / 2.0
    return upper, mid, lower
