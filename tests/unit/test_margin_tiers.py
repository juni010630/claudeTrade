"""MMR 티어 테이블 + BNB 할인 + LIMIT 폴백 수수료 회귀 테스트."""
from __future__ import annotations

import pandas as pd

from execution.broker import BacktestBroker
from execution.commission import CommissionModel
from execution.models import Order, OrderSide, OrderType
from risk.margin_tiers import ALT_BRACKETS, BTC_BRACKETS, MarginTierTable
from signals.models import LeverageTier, Signal, SignalScore


def _signal_score() -> SignalScore:
    sig = Signal(
        symbol="BTCUSDT",
        strategy="test",
        direction="long",
        entry_price=100.0,
        tp_price=110.0,
        sl_price=95.0,
        timestamp=pd.Timestamp("2024-01-01"),
    )
    return SignalScore(total=10, tier=LeverageTier.A, signal=sig)


def _order(direction: str = "long", price: float = 100.0) -> Order:
    return Order(
        symbol="BTCUSDT",
        side=OrderSide.BUY if direction == "long" else OrderSide.SELL,
        size_usd=10_000,
        price=price,
        order_type=OrderType.LIMIT,
        leverage=10,
        strategy="test",
        signal_score=_signal_score(),
        timestamp=pd.Timestamp("2024-01-01"),
        direction=direction,
        tp_price=110.0,
        sl_price=95.0,
    )


def test_btc_tier_low_notional():
    table = MarginTierTable()
    assert table.mm_rate("BTCUSDT", 10_000) == 0.004
    assert table.max_leverage("BTCUSDT", 10_000) == 125


def test_btc_tier_high_notional():
    table = MarginTierTable()
    assert table.mm_rate("BTCUSDT", 5_000_000) == 0.025


def test_alt_default_used_for_unknown_symbol():
    table = MarginTierTable()
    assert table.mm_rate("DOGEUSDT", 10_000) == 0.025


def test_bnb_discount_applies_to_both_rates():
    base = CommissionModel()
    disc = CommissionModel(bnb_discount=True)
    assert abs(disc.maker_rate - base.maker_rate * 0.9) < 1e-12
    assert abs(disc.taker_rate - base.taker_rate * 0.9) < 1e-12


def test_limit_fallback_uses_taker_fee():
    """LONG LIMIT @ 90 인데 봉 low=95 → 미체결 → close 폴백 → taker rate."""
    broker = BacktestBroker()
    order = _order(direction="long", price=90.0)
    bar = pd.Series({"open": 100, "high": 105, "low": 95, "close": 100, "volume": 1})
    fill = broker.submit(order, bar)
    # taker = 0.05% * 10_000 = 5
    assert abs(fill.commission - 5.0) < 1e-9
    # apply는 passthrough → fill_price == close
    assert fill.fill_price == 100
    # 슬리피지는 cost로만 반영
    assert fill.slippage_cost > 0


def test_limit_fill_keeps_maker_fee():
    """LONG LIMIT @ 100 이고 봉 low=95 → 체결 → maker rate."""
    broker = BacktestBroker()
    order = _order(direction="long", price=100.0)
    bar = pd.Series({"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1})
    fill = broker.submit(order, bar)
    # maker = 0.02% * 10_000 = 2
    assert abs(fill.commission - 2.0) < 1e-9


def test_alt_brackets_monotonic():
    last = -1.0
    for br in ALT_BRACKETS:
        assert br.mm_rate >= last
        last = br.mm_rate
    last = -1.0
    for br in BTC_BRACKETS:
        assert br.mm_rate >= last
        last = br.mm_rate
