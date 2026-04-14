"""전략에 주입되는 불변 per-bar 상태."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.schemas import MarketSnapshot
from regime.models import RegimeState
from risk.models import PortfolioState


@dataclass(frozen=True)
class BarContext:
    snapshot: MarketSnapshot
    regime: RegimeState
    portfolio_state: PortfolioState
    timestamp: pd.Timestamp
