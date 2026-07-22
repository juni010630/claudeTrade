"""실체결가로 보정된 TP/SL이 거래소와 tracker에 동일하게 반영되는지 검증."""
from __future__ import annotations

import pandas as pd

from execution.live_broker import LiveBroker
from execution.models import Order, OrderSide, OrderType
from portfolio.ledger import Ledger
from portfolio.tracker import PortfolioTracker


class _FillExchange:
    def __init__(self):
        self.orders = []

    def set_leverage(self, leverage, symbol):
        return None

    def market(self, symbol):
        return {"limits": {"amount": {"min": 0.001}, "cost": {"min": 1.0}}}

    def amount_to_precision(self, symbol, qty):
        return str(qty)

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.orders.append((type_, side, float(amount), price, params or {}))
        if type_ == "market":
            return {"average": 105.0, "filled": float(amount), "fee": {"cost": 0.5}}
        return {}


def test_fill_shift_updates_exchange_and_tracker_protection_prices():
    exchange = _FillExchange()
    broker = LiveBroker(exchange, dry_run=False, demo=False)
    order = Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        size_usd=1000.0,
        price=100.0,
        order_type=OrderType.MARKET,
        leverage=3,
        strategy="ema_cross",
        signal_score=None,
        timestamp=pd.Timestamp("2026-07-22T00:00:00Z"),
        direction="long",
        tp_price=110.0,
        sl_price=95.0,
    )

    fill = broker.submit(order)
    tracker = PortfolioTracker(10_000.0, Ledger())
    tracker.apply_fill(fill)
    position = tracker.snapshot().positions["BTCUSDT"]

    assert fill.fill_price == 105.0
    assert position.tp_price == 115.0
    assert position.sl_price == 100.0
    assert any(kind == "limit" and price == 115.0 for kind, _, _, price, _ in exchange.orders)
    assert any(
        kind == "STOP_MARKET" and params["stopPrice"] == 100.0
        for kind, _, _, _, params in exchange.orders
    )
