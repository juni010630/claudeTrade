"""추세 관련 지표 — 순수 함수, 사이드 이펙트 없음."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = "close",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(macd_line, signal_line, histogram) 반환."""
    fast_ema = df[col].ewm(span=fast, adjust=False).mean()
    slow_ema = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ADX(평균방향성지수). 값이 클수록 강한 추세."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = df["close"].shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = (
        pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean()
        / atr_s
        * 100
    )
    minus_di = (
        pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean()
        / atr_s
        * 100
    )

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(span=period, adjust=False).mean()
