"""성과 지표 계산 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics.drawdown import drawdown_series, max_drawdown, recovery_time
from metrics.returns import cagr, calmar, sharpe, sortino
from metrics.trade_stats import consecutive_losses, profit_factor, win_rate


@pytest.fixture
def flat_equity() -> pd.Series:
    ts = pd.date_range("2023-01-01", periods=100, freq="1h", tz="UTC")
    return pd.Series(np.ones(100) * 100000, index=ts)


@pytest.fixture
def growing_equity() -> pd.Series:
    ts = pd.date_range("2023-01-01", periods=8760, freq="1h", tz="UTC")
    vals = 100000 * (1 + 0.0001) ** np.arange(8760)
    return pd.Series(vals, index=ts)


@pytest.fixture
def drawdown_equity() -> pd.Series:
    ts = pd.date_range("2023-01-01", periods=10, freq="1h", tz="UTC")
    vals = [100, 110, 120, 100, 90, 80, 90, 100, 110, 120]
    return pd.Series(vals, index=ts, dtype=float)


def test_max_drawdown(drawdown_equity):
    mdd = max_drawdown(drawdown_equity)
    # 최고점 120에서 80으로 → 33.3% 하락
    assert abs(mdd - (-80 / 120 + 1) * -1) < 0.01 or mdd < -0.3


def test_sharpe_flat(flat_equity):
    # 변동이 없으면 샤프 0
    assert sharpe(flat_equity) == 0.0


def test_sharpe_positive(growing_equity):
    assert sharpe(growing_equity) > 0


def test_cagr_positive(growing_equity):
    assert cagr(growing_equity) > 0


def test_win_rate():
    df = pd.DataFrame({"pnl": [100, -50, 200, -30, 50]})
    assert abs(win_rate(df) - 0.6) < 1e-9


def test_profit_factor():
    df = pd.DataFrame({"pnl": [100, -50, 200, -50]})
    assert abs(profit_factor(df) - 300 / 100) < 1e-9


def test_consecutive_losses():
    df = pd.DataFrame({"pnl": [100, -10, -20, -30, 50, -10, -10]})
    assert consecutive_losses(df) == 3


def test_recovery_time(drawdown_equity):
    rec = recovery_time(drawdown_equity)
    assert rec is not None
