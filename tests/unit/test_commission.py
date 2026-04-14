"""수수료 + 슬리피지 계산 테스트."""
from __future__ import annotations

import pytest

from execution.commission import CommissionModel
from execution.models import OrderType
from execution.slippage import SlippageModel


def test_maker_commission():
    model = CommissionModel(maker_rate=0.0002, taker_rate=0.0005)
    cost = model.calculate(10000, OrderType.LIMIT)
    assert abs(cost - 2.0) < 1e-9


def test_taker_commission():
    model = CommissionModel(maker_rate=0.0002, taker_rate=0.0005)
    cost = model.calculate(10000, OrderType.MARKET)
    assert abs(cost - 5.0) < 1e-9


def test_slippage_market_long():
    model = SlippageModel(default_bps=10)
    fill_price = model.apply(30000, OrderType.MARKET, "long")
    assert fill_price > 30000


def test_slippage_market_short():
    model = SlippageModel(default_bps=10)
    fill_price = model.apply(30000, OrderType.MARKET, "short")
    assert fill_price < 30000


def test_no_slippage_limit():
    model = SlippageModel(default_bps=10)
    fill_price = model.apply(30000, OrderType.LIMIT, "long")
    assert fill_price == 30000
    assert model.cost(10000, OrderType.LIMIT) == 0.0
