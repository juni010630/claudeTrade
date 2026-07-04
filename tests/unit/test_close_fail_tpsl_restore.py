"""market_close 실패 시 TP/SL 재등록 검증 (2026-07-04 2차 리뷰 MEDIUM①).

회귀 대상: market_close는 SL/TP를 선취소한 뒤 청산하는데, 청산 시장가가 실패하면
취소된 SL/TP를 재등록하지 않아 조건 재발까지 거래소측 보호가 전무했음.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from data.schemas import MarketSnapshot
from engine.backtest import BacktestEngine
from execution.commission import CommissionModel
from execution.slippage import SlippageModel
from risk.models import Position


class _FakeBroker:
    """market_close 실패를 시뮬레이션하는 라이브 유사 브로커."""

    def __init__(self, close_fails: bool):
        self.close_fails = close_fails
        self.commission = CommissionModel()
        self.slippage = SlippageModel()
        self.refresh_calls = []

    def fetch_open_symbols(self):
        return {"BTCUSDT"}  # 거래소에 아직 열려 있음 → market_close 경로 진입

    def market_close(self, symbol, direction, qty, allow_maker=False):
        if self.close_fails:
            raise RuntimeError("rate limit")

    def refresh_tp_sl_after_add(self, symbol, direction, qty_total, tp_price, sl_price):
        self.refresh_calls.append((symbol, direction, qty_total, tp_price, sl_price))


def _setup(close_fails: bool):
    broker = _FakeBroker(close_fails)
    engine = BacktestEngine(initial_capital=100_000, strategies=[], broker=broker)
    ts = pd.Timestamp("2026-07-04 12:00", tz="UTC")
    engine.tracker.state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT", strategy="test_strat", direction="long",
        entry_price=30000.0, size_usd=1000.0, leverage=10,
        tp_price=40000.0, sl_price=27000.0,
        opened_at=ts - pd.Timedelta(hours=5),
    )
    snap = MarketSnapshot(timestamp=ts, bars={}, funding_rates={})
    regime = SimpleNamespace(regime=None)
    return engine, broker, snap, regime


def test_close_fail_restores_tp_sl():
    engine, broker, snap, regime = _setup(close_fails=True)
    pos = engine.tracker.state.positions["BTCUSDT"]
    engine._close_with_reason("BTCUSDT", pos, 27000.0, "sl", snap, regime)
    # 청산 실패 → 포지션 유지 + TP/SL 재등록 호출
    assert "BTCUSDT" in engine.tracker.state.positions
    (call,) = broker.refresh_calls
    assert call[0] == "BTCUSDT" and call[1] == "long"
    assert call[2] == pytest.approx(1000.0 / 30000.0)  # qty
    assert call[3] == pytest.approx(40000.0) and call[4] == pytest.approx(27000.0)


def test_close_success_no_refresh():
    engine, broker, snap, regime = _setup(close_fails=False)
    pos = engine.tracker.state.positions["BTCUSDT"]
    engine._close_with_reason("BTCUSDT", pos, 27000.0, "sl", snap, regime)
    # 정상 청산 → 재등록 없음 + 포지션 정리
    assert broker.refresh_calls == []
    assert "BTCUSDT" not in engine.tracker.state.positions
