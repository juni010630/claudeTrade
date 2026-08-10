from __future__ import annotations

import numpy as np
import pandas as pd

from data.schemas import MarketSnapshot
from signals.candidate_d import CandidateDConfig, evaluate_candidate_d


def _daily(start: str, periods: int = 300) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1D", tz="UTC")
    close = 100.0 + np.sin(np.arange(periods) / 3.0)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    })


def _funding(end: pd.Timestamp, high_last: bool = True) -> pd.DataFrame:
    ts = pd.date_range(end=end, periods=180, freq="8h", tz="UTC")
    rates = np.tile([-0.0001, 0.0, 0.0001], 60).astype(float)
    rates[-1] = 0.001 if high_last else -0.001
    return pd.DataFrame({"timestamp": ts, "rate": rates})


def test_candidate_d_blocks_only_with_valid_range_and_high_funding(monkeypatch):
    now = pd.Timestamp("2024-11-01 08:00", tz="UTC")
    bars = {
        "BTCUSDT": {"1d": _daily("2024-01-06")},
        "ETHUSDT": {"1d": _daily("2024-01-06")},
    }
    snapshot = MarketSnapshot(
        timestamp=now, bars=bars,
        funding_history={"FILUSDT": _funding(now)},
    )
    monkeypatch.setattr(
        "signals.candidate_d.trend_index",
        lambda frame, unused: pd.DataFrame({"tri": [30.0]}),
    )

    decision = evaluate_candidate_d(snapshot, "FILUSDT", CandidateDConfig(enabled=True))

    assert decision["valid"] is True
    assert decision["would_block"] is True
    assert decision["market_tri"] == 30.0
    assert decision["funding_z"] > 0.5


def test_candidate_d_is_noop_when_auxiliary_data_is_stale(monkeypatch):
    now = pd.Timestamp("2024-11-03 08:00", tz="UTC")
    bars = {
        "BTCUSDT": {"1d": _daily("2024-01-06")},
        "ETHUSDT": {"1d": _daily("2024-01-06")},
    }
    snapshot = MarketSnapshot(
        timestamp=now, bars=bars,
        funding_history={"FILUSDT": _funding(now)},
    )
    monkeypatch.setattr(
        "signals.candidate_d.trend_index",
        lambda frame, unused: pd.DataFrame({"tri": [30.0]}),
    )

    decision = evaluate_candidate_d(snapshot, "FILUSDT", CandidateDConfig(enabled=True))

    assert decision == {
        "valid": False,
        "would_block": False,
        "reason": "stale_BTCUSDT_daily_bar",
    }


def test_candidate_d_rejects_future_funding_observation(monkeypatch):
    now = pd.Timestamp("2024-11-01 08:00", tz="UTC")
    bars = {
        "BTCUSDT": {"1d": _daily("2024-01-06")},
        "ETHUSDT": {"1d": _daily("2024-01-06")},
    }
    history = _funding(now)
    history.loc[len(history)] = [now + pd.Timedelta(hours=8), 0.001]
    snapshot = MarketSnapshot(
        timestamp=now, bars=bars,
        funding_history={"FILUSDT": history},
    )
    monkeypatch.setattr(
        "signals.candidate_d.trend_index",
        lambda frame, unused: pd.DataFrame({"tri": [30.0]}),
    )

    decision = evaluate_candidate_d(snapshot, "FILUSDT", CandidateDConfig(enabled=True))

    assert decision["valid"] is False
    assert decision["would_block"] is False
    assert decision["reason"] == "future_funding_observation"
