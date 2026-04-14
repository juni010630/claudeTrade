from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    PRE_BREAKOUT = "pre_breakout"  # BB 극도 수축, 방향 불명


@dataclass
class RegimeState:
    regime: MarketRegime
    adx: float
    bb_width: float
    bb_width_pct: float       # 최근 50봉 중 백분위 (0~1)
    timestamp: pd.Timestamp
    confidence: float = 1.0   # 0~1, 향후 ML 확률 등으로 확장 가능
