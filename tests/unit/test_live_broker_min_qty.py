"""LiveBroker.submit이 최소 주문 제약을 리스크 초과 없이 강제하는지 검증."""
from __future__ import annotations

import pandas as pd
import pytest

from execution.live_broker import LiveBroker
from execution.models import Order, OrderSide, OrderType


class _FakeExchange:
    """dry-run에 필요한 최소 인터페이스 — market limits + 정밀도만."""
    def __init__(self, min_qty: float, min_cost: float | None = None):
        self._min = min_qty
        self._min_cost = min_cost

    def market(self, symbol):
        return {
            "limits": {
                "amount": {"min": self._min},
                "cost": {"min": self._min_cost},
            }
        }

    def amount_to_precision(self, symbol, qty):
        return f"{float(qty):.3f}"


def _order(size_usd: float, price: float) -> Order:
    return Order(
        symbol="STORJUSDT", side=OrderSide.BUY, size_usd=size_usd, price=price,
        order_type=OrderType.MARKET, leverage=3, strategy="mean_reversion",
        signal_score=None, timestamp=pd.Timestamp("2026-06-09T00:00:00Z"),
        direction="long", tp_price=price * 1.1, sl_price=price * 0.95,
    )


def test_below_min_qty_is_rejected_without_upsizing():
    broker = LiveBroker(exchange=_FakeExchange(min_qty=1.0), dry_run=True)
    order = _order(size_usd=10.0, price=100.0)
    with pytest.raises(ValueError, match="최소"):
        broker.submit(order)
    assert order.size_usd == 10.0


def test_no_bump_keeps_size_usd():
    # 의도 qty = 500/100 = 5.0 >= 최소 1.0 → bump 없음 → size_usd 그대로 500
    broker = LiveBroker(exchange=_FakeExchange(min_qty=1.0), dry_run=True)
    order = _order(size_usd=500.0, price=100.0)
    fill = broker.submit(order)
    assert fill.order.size_usd == 500.0


def test_below_min_cost_is_rejected():
    broker = LiveBroker(
        exchange=_FakeExchange(min_qty=0.001, min_cost=20.0), dry_run=True
    )
    order = _order(size_usd=10.0, price=100.0)
    with pytest.raises(ValueError, match="주문 금액"):
        broker.submit(order)


def test_invalid_protection_price_is_rejected_before_exchange_call():
    broker = LiveBroker(exchange=_FakeExchange(min_qty=0.001), dry_run=False)
    order = _order(size_usd=100.0, price=100.0)
    order.tp_price = -1.0
    with pytest.raises(ValueError, match="유효하지 않은"):
        broker.submit(order)
