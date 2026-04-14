"""시장 구조 지표 — OI 변화율 등."""
from __future__ import annotations

import pandas as pd


def open_interest_change(oi_series: pd.Series, lookback: int = 3) -> pd.Series:
    """OI 변화율 (%). 급증 시 캐스케이드 청산 위험 신호."""
    return oi_series.pct_change(lookback) * 100


def btc_dominance_change(dominance_series: pd.Series, lookback: int = 1) -> pd.Series:
    """BTC 도미넌스 변화량 (포인트)."""
    return dominance_series.diff(lookback)
