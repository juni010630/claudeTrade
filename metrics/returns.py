"""수익률 지표 — 순수 함수."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """연율화 Sharpe. periods_per_year=8760 (1시간봉 기준)."""
    returns = equity.pct_change().dropna()
    if returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def calmar(equity: pd.Series, periods_per_year: int = 8760) -> float:
    ann_return = cagr(equity, periods_per_year)
    mdd = max_drawdown_pct(equity)
    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return ann_return / abs(mdd)


def cagr(equity: pd.Series, periods_per_year: int = 8760) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0]
    n_years = len(equity) / periods_per_year
    return float(total_return ** (1 / n_years) - 1)


def max_drawdown_pct(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max.replace(0, float("nan"))
    return float(dd.min()) if not dd.empty else 0.0
