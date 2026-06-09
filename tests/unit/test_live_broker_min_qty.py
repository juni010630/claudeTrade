"""LiveBroker.submit: 최소수량 보정 시 order.size_usd가 실제 노셔널로 보정되는지 검증 (F20).

회귀 대상: 최소수량 bump 시 거래소엔 큰 수량이 들어가는데 tracker는 의도(작은) size_usd를
기록해 MTM/PnL/펀딩이 잘못된 노셔널로 계산되던 버그. 보정 후 size_usd = 실제수량×체결가.
"""
from __future__ import annotations

import pandas as pd

from execution.live_broker import LiveBroker
from execution.models import Order, OrderSide, OrderType


class _FakeExchange:
    """dry-run에 필요한 최소 인터페이스 — market limits + 정밀도만."""
    def __init__(self, min_qty: float):
        self._min = min_qty

    def market(self, symbol):
        return {"limits": {"amount": {"min": self._min}}}

    def amount_to_precision(self, symbol, qty):
        return f"{float(qty):.3f}"


def _order(size_usd: float, price: float) -> Order:
    return Order(
        symbol="STORJUSDT", side=OrderSide.BUY, size_usd=size_usd, price=price,
        order_type=OrderType.MARKET, leverage=3, strategy="mean_reversion",
        signal_score=None, timestamp=pd.Timestamp("2026-06-09T00:00:00Z"),
        direction="long", tp_price=price * 1.1, sl_price=price * 0.95,
    )


def test_min_qty_bump_corrects_size_usd():
    # 의도 qty = 10/100 = 0.1 < 최소 1.0 → 1.0으로 bump → size_usd = 1.0*100 = 100
    broker = LiveBroker(exchange=_FakeExchange(min_qty=1.0), dry_run=True)
    order = _order(size_usd=10.0, price=100.0)
    fill = broker.submit(order)
    assert fill.order.size_usd == 100.0  # 의도 10이 아니라 실제 노셔널 100
    assert fill.fill_price == 100.0
    # 수수료/슬리피지도 보정된 노셔널 기준
    assert fill.commission == 100.0 * 0.0005  # taker 5bps
    assert fill.slippage_cost == 100.0 * 5 / 10000  # 5bps


def test_no_bump_keeps_size_usd():
    # 의도 qty = 500/100 = 5.0 >= 최소 1.0 → bump 없음 → size_usd 그대로 500
    broker = LiveBroker(exchange=_FakeExchange(min_qty=1.0), dry_run=True)
    order = _order(size_usd=500.0, price=100.0)
    fill = broker.submit(order)
    assert fill.order.size_usd == 500.0


def test_tp_sl_qty_matches_filled_qty_after_bump():
    """bump 후 청산 시 qty = size_usd/entry_price = 실제 체결수량이어야 함 (고아 방지)."""
    broker = LiveBroker(exchange=_FakeExchange(min_qty=1.0), dry_run=True)
    order = _order(size_usd=10.0, price=100.0)
    fill = broker.submit(order)
    # tracker가 기록할 청산 수량 = size_usd/entry = 100/100 = 1.0 = 실제 거래소 수량
    close_qty = fill.order.size_usd / fill.fill_price
    assert close_qty == 1.0
