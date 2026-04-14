"""Supertrend 지표 — ATR 기반 동적 지지/저항선 (numpy 구현)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.trend import atr as calc_atr


def supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """
    Supertrend 계산 (numpy 기반, pandas iloc 루프 버그 없음).

    Returns
    -------
    st_line  : pd.Series  — Supertrend 선 값
    direction: pd.Series  — 1 = 상승(롱), -1 = 하락(숏)
    """
    atr = calc_atr(df, period).to_numpy()
    high = df["high"].to_numpy()
    low  = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = np.ones(n, dtype=np.int8)
    st_line = lower.copy()

    for i in range(1, n):
        # 상단 밴드 결정
        upper[i] = (
            upper_basic[i]
            if upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]
            else upper[i - 1]
        )
        # 하단 밴드 결정
        lower[i] = (
            lower_basic[i]
            if lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]
            else lower[i - 1]
        )

        # 방향 및 Supertrend 선
        if direction[i - 1] == 1:
            if close[i] < lower[i]:
                direction[i] = -1
                st_line[i] = upper[i]
            else:
                direction[i] = 1
                st_line[i] = lower[i]
        else:
            if close[i] > upper[i]:
                direction[i] = 1
                st_line[i] = lower[i]
            else:
                direction[i] = -1
                st_line[i] = upper[i]

    idx = df.index
    return pd.Series(st_line, index=idx), pd.Series(direction.astype(int), index=idx)
