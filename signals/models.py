from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import pandas as pd


class LeverageTier(Enum):
    SSS = "SSS"     # 최고 신호 (6점+): 최대 레버리지
    SS = "SS"       # 강한 신호 (5점)
    S  = "S"        # 중상 신호 (4점)
    A  = "A"        # 중간 신호 (3점)
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
