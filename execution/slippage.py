"""슬리피지 모델."""
from __future__ import annotations

from execution.models import OrderType


class SlippageModel:
    def __init__(self, default_bps: float = 5.0) -> None:
        """default_bps: 기본 슬리피지 (basis points, 5bps = 0.05%)"""
        self.default_bps = default_bps

    def apply(self, price: float, order_type: OrderType, direction: str) -> float:
        """
        시장가: 슬리피지 적용 (롱은 더 비싸게, 숏은 더 싸게 체결)
        지정가: 슬리피지 없음
        """
        if order_type == OrderType.LIMIT:
            return price
        slip = price * self.default_bps / 10000
        return price + slip if direction == "long" else price - slip

    def cost(self, notional: float, order_type: OrderType) -> float:
        if order_type == OrderType.LIMIT:
            return 0.0
        return notional * self.default_bps / 10000
