"""수수료 모델."""
from __future__ import annotations

from execution.models import OrderType


class CommissionModel:
    def __init__(
        self,
        maker_rate: float = 0.0002,  # Binance USDⓈ-M VIP0 maker 0.02%
        taker_rate: float = 0.0005,  # Binance USDⓈ-M VIP0 taker 0.05%
        bnb_discount: bool = False,  # BNB 잔고로 수수료 결제 시 10% 할인
    ) -> None:
        mult = 0.9 if bnb_discount else 1.0
        self.maker_rate = maker_rate * mult
        self.taker_rate = taker_rate * mult

    def calculate(self, notional: float, order_type: OrderType) -> float:
        rate = self.maker_rate if order_type == OrderType.LIMIT else self.taker_rate
        return notional * rate
