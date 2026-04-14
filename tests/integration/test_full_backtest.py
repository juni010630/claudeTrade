"""합성 데이터로 end-to-end 백테스트 통합 테스트."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from data.schemas import MarketSnapshot
from engine.backtest import BacktestEngine
from strategies.ema_cross import EMACrossStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy


def _make_df(n: int = 500, trend: bool = False) -> pd.DataFrame:
    np.random.seed(0)
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    if trend:
        close = 30000.0 + np.arange(n) * 15 + np.random.randn(n) * 50
    else:
        close = 30000.0 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 80) + 50
    low = close - np.abs(np.random.randn(n) * 80) - 50
    open_ = close + np.random.randn(n) * 40
    volume = np.abs(np.random.randn(n) * 2000 + 8000)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _make_snapshots(n: int = 500) -> Iterator[MarketSnapshot]:
    df_1h = _make_df(n, trend=True)
    df_4h = _make_df(n // 4 + 10, trend=True)
    df_1d = _make_df(n // 24 + 10, trend=True)
    lookback = 300

    for i in range(lookback, n):
        ts = df_1h["timestamp"].iloc[i]
        snap = MarketSnapshot(
            timestamp=ts,
            bars={
                "BTCUSDT": {
                    "1h": df_1h.iloc[max(0, i - lookback) : i + 1].reset_index(drop=True),
                    "4h": df_4h.iloc[: max(1, i // 4)].reset_index(drop=True),
                    "1d": df_1d.iloc[: max(1, i // 24)].reset_index(drop=True),
                }
            },
            funding_rates={"BTCUSDT": 0.0001},
        )
        yield snap


def test_backtest_runs_without_error():
    strategies = [
        MomentumBreakoutStrategy({"symbols": ["BTCUSDT"]}),
        EMACrossStrategy({"symbols": ["BTCUSDT"]}),
        MeanReversionStrategy({"symbols": ["BTCUSDT"]}),
    ]
    engine = BacktestEngine(initial_capital=100_000, strategies=strategies)
    report = engine.run(_make_snapshots(500))
    assert report is not None
    assert report.initial_equity == 100_000


def test_equity_curve_tracked():
    strategies = [MomentumBreakoutStrategy({"symbols": ["BTCUSDT"]})]
    engine = BacktestEngine(initial_capital=100_000, strategies=strategies)
    engine.run(_make_snapshots(400))
    assert len(engine.equity_curve) > 0


def test_no_negative_equity():
    strategies = [
        MomentumBreakoutStrategy({"symbols": ["BTCUSDT"]}),
        MeanReversionStrategy({"symbols": ["BTCUSDT"]}),
    ]
    engine = BacktestEngine(initial_capital=100_000, strategies=strategies)
    engine.run(_make_snapshots(500))
    eq = engine.equity_curve.to_series()
    # 강제 청산 있으므로 자산이 0 아래로 가면 안 됨
    assert (eq > 0).all()
