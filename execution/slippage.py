"""슬리피지 모델."""
from __future__ import annotations

from execution.models import OrderType


class SlippageModel:
    def __init__(self, default_bps: float = 5.0) -> None:
        """default_bps: 기본 슬리피지 (basis points, 5bps = 0.05%)"""
        self.default_bps = default_bps

    def apply(self, price: float, order_type: OrderType, direction: str) -> float:
        """원가를 그대로 반환. 슬리피지는 cost()로만 현금 차감 (이중 계상 방지)."""
        return price

    def cost(self, notional: float, order_type: OrderType) -> float:
        if order_type == OrderType.LIMIT:
            return 0.0
        return notional * self.default_bps / 10000
