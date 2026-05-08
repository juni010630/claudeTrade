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


def test_slippage_apply_is_passthrough():
    """apply()는 원가 반환 (이중 계상 방지). 현금 비용은 cost()만 적용."""
    model = SlippageModel(default_bps=10)
    assert model.apply(30000, OrderType.MARKET, "long") == 30000
    assert model.apply(30000, OrderType.MARKET, "short") == 30000
    assert model.apply(30000, OrderType.LIMIT, "long") == 30000


def test_slippage_cost():
    model = SlippageModel(default_bps=10)
    assert abs(model.cost(10000, OrderType.MARKET) - 10.0) < 1e-9
    assert model.cost(10000, OrderType.LIMIT) == 0.0
