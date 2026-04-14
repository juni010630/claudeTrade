"""수수료 모델."""
from __future__ import annotations

from execution.models import OrderType


class CommissionModel:
    def __init__(
        self,
        maker_rate: float = 0.0002,  # 0.02%
        taker_rate: float = 0.0005,  # 0.05%
    ) -> None:
        self.maker_rate = maker_rate
        self.taker_rate = taker_rate

    def calculate(self, notional: float, order_type: OrderType) -> float:
        rate = self.maker_rate if order_type == OrderType.LIMIT else self.taker_rate
        return notional * rate
