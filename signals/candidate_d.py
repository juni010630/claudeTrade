"""Candidate D: range-market + elevated funding veto for multi-TF breakouts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.schemas import MarketSnapshot
from regime.trend_index import trend_index


@dataclass(frozen=True)
class CandidateDConfig:
    enabled: bool = False
    market_tri_max: float = 35.0
    funding_z_max: float = 0.5
    funding_window: int = 180
    funding_min_periods: int = 90
    max_daily_age_hours: float = 26.0
    max_funding_age_hours: float = 12.0
    max_funding_gap_hours: float = 24.0


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _timestamped(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "timestamp" not in data.columns:
        data = data.reset_index()
    if "timestamp" not in data.columns:
        raise ValueError("timestamp column is required")
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return data.sort_values("timestamp").reset_index(drop=True)


def evaluate_candidate_d(
    snapshot: MarketSnapshot, symbol: str, config: CandidateDConfig,
) -> dict:
    """Return a causal decision. Invalid/stale auxiliary data never vetoes a trade."""
    now = _utc(snapshot.timestamp)
    tri_values = {}
    for market_symbol in ("BTCUSDT", "ETHUSDT"):
        frame = snapshot.bars.get(market_symbol, {}).get("1d")
        if frame is None or frame.empty:
            return {"valid": False, "would_block": False,
                    "reason": f"missing_{market_symbol}_1d"}
        try:
            daily = _timestamped(frame)
            close_times = daily["timestamp"] + pd.Timedelta(days=1)
            if close_times.gt(now).any():
                return {"valid": False, "would_block": False,
                        "reason": f"future_{market_symbol}_daily_bar"}
            age = (now - close_times.iloc[-1]).total_seconds() / 3600.0
            if age < 0 or age > config.max_daily_age_hours:
                return {"valid": False, "would_block": False,
                        "reason": f"stale_{market_symbol}_daily_bar"}
            value = float(trend_index(daily, None)["tri"].iloc[-1])
        except Exception as error:
            return {"valid": False, "would_block": False,
                    "reason": f"tri_error_{market_symbol}:{type(error).__name__}"}
        if not np.isfinite(value):
            return {"valid": False, "would_block": False,
                    "reason": f"nonfinite_{market_symbol}_tri"}
        tri_values[market_symbol] = value

    frame = snapshot.funding_history.get(symbol)
    if frame is None or frame.empty:
        return {"valid": False, "would_block": False,
                "reason": "missing_funding_history"}
    try:
        history = _timestamped(frame)
    except (TypeError, ValueError) as error:
        return {"valid": False, "would_block": False,
                "reason": f"invalid_funding_history:{error}"}
    if "rate" not in history.columns:
        return {"valid": False, "would_block": False,
                "reason": "missing_funding_rate"}
    if history["timestamp"].gt(now).any():
        return {"valid": False, "would_block": False,
                "reason": "future_funding_observation"}
    history = history.drop_duplicates("timestamp", keep="last").tail(config.funding_window)
    if len(history) < config.funding_min_periods:
        return {"valid": False, "would_block": False,
                "reason": "insufficient_funding_history"}
    funding_age = (now - history["timestamp"].iloc[-1]).total_seconds() / 3600.0
    if funding_age < 0 or funding_age > config.max_funding_age_hours:
        return {"valid": False, "would_block": False,
                "reason": "stale_funding_history"}
    gaps = history["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    max_gap = float(gaps.max()) if not gaps.empty else 0.0
    if max_gap > config.max_funding_gap_hours:
        return {"valid": False, "would_block": False,
                "reason": "gapped_funding_history"}
    rates = history["rate"].astype(float)
    std = float(rates.std())
    if not np.isfinite(std) or std <= 0:
        return {"valid": False, "would_block": False, "reason": "zero_funding_std"}

    market_tri = float(np.mean(list(tri_values.values())))
    funding_z = float((rates.iloc[-1] - rates.mean()) / std)
    would_block = market_tri < config.market_tri_max and funding_z > config.funding_z_max
    return {
        "valid": True,
        "would_block": bool(would_block),
        "reason": None,
        "market_tri": market_tri,
        "funding_z": funding_z,
        "observations": len(history),
    }
