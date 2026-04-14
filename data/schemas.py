from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import pandas as pd


@dataclass(frozen=True)
class OHLCVBar:
    symbol: str
    timeframe: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class FundingRateBar:
    symbol: str
    timestamp: pd.Timestamp
    rate: float  # e.g. 0.0001 = 0.01%


@dataclass
class MarketSnapshot:
    """Per-bar 상태. 전략이 볼 수 있는 모든 데이터."""
    timestamp: pd.Timestamp
    # bars[symbol][timeframe] = DataFrame (OHLCV, 최신 봉이 마지막 행)
    bars: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    funding_rates: dict[str, float] = field(default_factory=dict)   # symbol → 현재 펀딩비
    open_interest: dict[str, float] = field(default_factory=dict)   # symbol → OI (USD)
    btc_dominance: float = 0.0
