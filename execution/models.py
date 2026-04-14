from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

from signals.models import SignalScore


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    size_usd: float          # 명목 크기
    price: float             # 지정가 or 시장가 기준가
    order_type: OrderType
    leverage: int
    strategy: str
    signal_score: SignalScore
    timestamp: pd.Timestamp
    direction: Literal["long", "short"] = "long"
    tp_price: float = 0.0
    sl_price: float = 0.0


@dataclass
class Fill:
    order: Order
    fill_price: float
    commission: float
    slippage_cost: float
    timestamp: pd.Timestamp

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage_cost
