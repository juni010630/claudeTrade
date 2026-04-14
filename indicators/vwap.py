"""VWAP (Volume Weighted Average Price) — UTC 자정 기준 세션 초기화."""
from __future__ import annotations

import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    세션 VWAP 계산. UTC 자정마다 리셋.

    Parameters
    ----------
    df : timestamp, high, low, close, volume 컬럼 포함 DataFrame

    Returns
    -------
    pd.Series — df와 동일 인덱스
    """
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    else:
        ts = pd.to_datetime(df.index)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]

    # 날짜별 누적합 → VWAP = Σ(TP×V) / Σ(V)
    date_key = ts.dt.date.values
    cum_tpvol = tp_vol.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()

    result = cum_tpvol / cum_vol.replace(0.0, float("nan"))
    result.index = df.index
    return result
