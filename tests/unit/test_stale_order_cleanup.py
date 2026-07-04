"""좀비 SL 정리 검증 (2026-07-04 실증: 바이낸스 Algo 주문부 분리로 SL 잔존).

두 근본 원인의 재발 방지:
①cancel_all이 일반 주문부만 지워 STOP_MARKET(트리거)이 청산 후 생존
②거래소측 TP/SL 체결(already_closed)은 반대 주문을 아예 취소 안 함
→ 잔존 SL이 같은 방향 재진입 포지션을 구 가격에 오청산 (LTC 38.36 실사례).
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from data.schemas import MarketSnapshot
from engine.backtest import BacktestEngine
from execution.commission import CommissionModel
from execution.live_broker import LiveBroker
from execution.slippage import SlippageModel
from risk.models import Position


class _FakeExchange:
    def __init__(self):
        self.cancel_calls = []  # (symbol, trigger여부)

    def cancel_all_orders(self, symbol, params=None):
        self.cancel_calls.append((symbol, bool((params or {}).get("trigger"))))

    def fetch_positions(self, symbols=None):
        return [{"contracts": 1.0}]

    def amount_to_precision(self, s, q):
        return f"{float(q):.3f}"

    def create_order(self, symbol, type_, side, qty, price=None, params=None):
        return {"id": "x", "average": 100.0, "filled": float(qty)}


def test_market_close_cancels_both_order_books():
    """market_close가 일반 + 트리거(Algo) 주문부를 모두 취소해야 함."""
    ex = _FakeExchange()
    broker = LiveBroker(exchange=ex, dry_run=False)
    broker.market_close("LTCUSDT", "long", 1.0)
    assert ("LTCUSDT", False) in ex.cancel_calls  # 일반 주문부
    assert ("LTCUSDT", True) in ex.cancel_calls   # Algo(트리거) 주문부


def test_public_cancel_all_covers_trigger_book():
    ex = _FakeExchange()
    broker = LiveBroker(exchange=ex, dry_run=False)
    broker.cancel_all_orders("ADAUSDT")
    assert ("ADAUSDT", True) in ex.cancel_calls


class _FakeBroker:
    """거래소측 TP 체결로 포지션이 이미 사라진 상황의 브로커."""

    def __init__(self):
        self.commission = CommissionModel()
        self.slippage = SlippageModel()
        self.cancel_calls = []

    def fetch_open_symbols(self):
        return set()  # 포지션 이미 청산됨 → already_closed 분기

    def market_close(self, *a, **k):
        raise AssertionError("already_closed면 market_close 호출 금지")

    def cancel_all_orders(self, symbol):
        self.cancel_calls.append(symbol)


def test_already_closed_cancels_leftover_orders():
    """거래소측 TP/SL 체결 감지 시 잔존 반대 주문(SL/TP) 취소해야 함."""
    broker = _FakeBroker()
    engine = BacktestEngine(initial_capital=100_000, strategies=[], broker=broker)
    ts = pd.Timestamp("2026-07-04 12:00", tz="UTC")
    engine.tracker.state.positions["ADAUSDT"] = Position(
        symbol="ADAUSDT", strategy="multi_tf_breakout", direction="short",
        entry_price=0.15, size_usd=1000.0, leverage=10,
        tp_price=0.14, sl_price=0.16, opened_at=ts - pd.Timedelta(hours=5),
    )
    pos = engine.tracker.state.positions["ADAUSDT"]
    snap = MarketSnapshot(timestamp=ts, bars={}, funding_rates={})
    engine._close_with_reason("ADAUSDT", pos, 0.14, "tp", snap,
                              SimpleNamespace(regime=None))
    assert broker.cancel_calls == ["ADAUSDT"]  # 잔존 SL 정리됨
    assert "ADAUSDT" not in engine.tracker.state.positions
