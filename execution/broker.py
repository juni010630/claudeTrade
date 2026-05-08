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

        # 지정가 주문 체결 가능성 체크
        # LONG LIMIT: 시장가가 limit 이하로 내려와야 체결 (bar_low <= limit)
        # SHORT LIMIT: 시장가가 limit 이상으로 올라와야 체결 (bar_high >= limit)
        # 미도달 시 시장가 폴백(봉 종가) → taker 수수료/슬리피지로 재계산
        effective_type = order.order_type
        if order.order_type == OrderType.LIMIT and current_bar is not None:
            bar_low = float(current_bar["low"])
            bar_high = float(current_bar["high"])
            fallback = (
                (order.direction == "long" and fill_price < bar_low)
                or (order.direction == "short" and fill_price > bar_high)
            )
            if fallback:
                effective_type = OrderType.MARKET
                # 시장가 슬리피지를 종가에 적용
                fill_price = self.slippage.apply(
                    float(current_bar["close"]), OrderType.MARKET, order.direction
                )

        commission = self.commission.calculate(order.size_usd, effective_type)
        slip_cost = self.slippage.cost(order.size_usd, effective_type)

        return Fill(
            order=order,
            fill_price=fill_price,
            commission=commission,
            slippage_cost=slip_cost,
            timestamp=order.timestamp,
        )
