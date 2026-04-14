"""백테스트 브로커 — 라이브 시 LiveBroker로 교체."""
from __future__ import annotations

import pandas as pd

from execution.commission import CommissionModel
from execution.models import Fill, Order, OrderType
from execution.slippage import SlippageModel


class BacktestBroker:
    def __init__(
        self,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
    ) -> None:
        self.commission = commission_model or CommissionModel()
        self.slippage = slippage_model or SlippageModel()

    def submit(self, order: Order, current_bar: pd.Series | None = None) -> Fill:
        """
        주문을 처리하고 Fill을 반환.
        current_bar: {'open', 'high', 'low', 'close', 'volume'} Series (선택)
        """
        fill_price = self.slippage.apply(order.price, order.order_type, order.direction)

        # 지정가 주문이 현재 봉 범위 밖이면 시장가로 대체
        if order.order_type == OrderType.LIMIT and current_bar is not None:
            bar_low = float(current_bar["low"])
            bar_high = float(current_bar["high"])
            if order.direction == "long" and fill_price > bar_high:
                fill_price = float(current_bar["close"])
            elif order.direction == "short" and fill_price < bar_low:
                fill_price = float(current_bar["close"])

        commission = self.commission.calculate(order.size_usd, order.order_type)
        slip_cost = self.slippage.cost(order.size_usd, order.order_type)

        return Fill(
            order=order,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slip_cost,
            timestamp=order.timestamp,
        )
