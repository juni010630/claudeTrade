"""드로다운 분석 — 순수 함수."""
from __future__ import annotations

import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, float("nan"))
    return float(dd.min()) if not dd.empty else 0.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max.replace(0, float("nan"))


def recovery_time(equity: pd.Series) -> pd.Timedelta | None:
    """MDD 발생 후 전고점 회복까지 소요 시간. 미회복이면 None."""
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, float("nan"))
    trough_idx = dd.idxmin()
    if trough_idx is None:
        return None
    after_trough = equity[trough_idx:]
    peak_val = rolling_max[trough_idx]
    recovered = after_trough[after_trough >= peak_val]
    if recovered.empty:
        return None
    return recovered.index[0] - trough_idx
