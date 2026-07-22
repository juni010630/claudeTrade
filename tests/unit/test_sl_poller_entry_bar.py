"""SL poller가 진입 timestamp와 같은 open time의 첫 5m 완성봉을 놓치지 않는지 검증."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from execution.commission import CommissionModel
from execution.sl_poller import SLPoller
from execution.slippage import SlippageModel
from portfolio.ledger import Ledger
from portfolio.tracker import PortfolioTracker
from risk.models import Position


class _Exchange:
    def fetch_ohlcv(self, symbol, tf, limit=5):
        ts = pd.Timestamp("2026-01-01T00:00:00Z")
        return [[int(ts.timestamp() * 1000), 100, 101, 94, 96, 1]]


class _Broker:
    def __init__(self):
        self.closed = []
        self.commission = CommissionModel()
        self.slippage = SlippageModel()

    def market_close(self, symbol, direction, qty):
        self.closed.append((symbol, direction, qty))

    def fetch_recent_fill_price(self, symbol):
        return None


def test_same_timestamp_completed_bar_can_trigger_sl():
    tracker = PortfolioTracker(10_000, Ledger())
    tracker.state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        strategy="ema_cross",
        direction="long",
        entry_price=100.0,
        size_usd=500.0,
        leverage=3,
        tp_price=110.0,
        sl_price=95.0,
        opened_at=pd.Timestamp("2026-01-01T00:00:00Z"),
    )
    broker = _Broker()
    breaker = SimpleNamespace(record_result=lambda *args, **kwargs: None)
    engine = SimpleNamespace(
        tracker=tracker,
        ledger=tracker.ledger,
        _strategy_guard_isolated={},
        circuit_breaker=breaker,
    )
    poller = SLPoller(engine, broker, _Exchange(), tf="5m")

    poller.check_once()

    assert broker.closed == [("BTCUSDT", "long", 5.0)]
    assert "BTCUSDT" not in tracker.snapshot().positions
