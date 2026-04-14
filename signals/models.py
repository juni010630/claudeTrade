from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import pandas as pd


class LeverageTier(Enum):
    SS = "SS"       # 완벽 신호 (7점): 최고 레버리지
    S  = "S"        # 강한 신호 (5-6점)
    A  = "A"        # 중간 신호 (3-4점)
    B  = "B"        # 약한 신호 (2점)
    C  = "C"        # 최소 신호 (1점): 소규모 탐색
    NO_TRADE = "NO_TRADE"


@dataclass
class Signal:
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    entry_price: float
    tp_price: float
    sl_price: float
    timestamp: pd.Timestamp
    raw_points: dict[str, int] = field(default_factory=dict)


@dataclass
class SignalScore:
    total: int          # 0~7
    tier: LeverageTier
    signal: Signal
