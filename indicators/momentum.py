"""모멘텀 지표 — 순수 함수."""
from __future__ import annotations

import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """현재 거래량 / 최근 N봉 평균 거래량."""
    avg = df["volume"].rolling(period).mean()
    return df["volume"] / avg.replace(0, float("nan"))
