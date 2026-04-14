"""리스크 가드 + 서킷브레이커 단위 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from risk.circuit_breaker import BreakerStatus, CircuitBreaker
from risk.guards import DrawdownAction, RiskGuards
from risk.models import PortfolioState, Position
from signals.models import LeverageTier, Signal, SignalScore


def _make_state(equity: float, daily_start: float, positions: dict | None = None) -> PortfolioState:
    state = PortfolioState(equity=equity, cash=equity, daily_start_equity=daily_start)
    if positions:
        state.positions = positions
    return state


def _make_signal(sym: str = "BTCUSDT", direction: str = "long") -> Signal:
    ts = pd.Timestamp("2023-01-01", tz="UTC")
    return Signal(sym, "test", direction, 30000, 31000, 29000, ts)


def test_daily_drawdown_ok():
    guards = RiskGuards()
    state = _make_state(99000, 100000)
    assert guards.check_daily_drawdown(state) == DrawdownAction.OK


def test_daily_drawdown_pause():
    guards = RiskGuards()
    state = _make_state(94000, 100000)  # -6%
    assert guards.check_daily_drawdown(state) == DrawdownAction.PAUSE


def test_daily_drawdown_stop():
    guards = RiskGuards()
    state = _make_state(91000, 100000)  # -9%
    assert guards.check_daily_drawdown(state) == DrawdownAction.STOP


def test_max_positions_limit():
    guards = RiskGuards(max_positions=2)
    pos = Position("BTCUSDT", "t", "long", 30000, 1000, 3, 31000, 29000, pd.Timestamp.now())
    pos2 = Position("ETHUSDT", "t", "long", 2000, 1000, 3, 2100, 1900, pd.Timestamp.now())
    state = _make_state(100000, 100000, {"BTCUSDT": pos, "ETHUSDT": pos2})
    assert not guards.check_max_positions(state)


def test_circuit_breaker_pause():
    cb = CircuitBreaker(strategy_pause_losses=3, global_stop_losses=10, pause_duration_hours=48)
    now = pd.Timestamp("2023-01-01", tz="UTC")
    for _ in range(3):
        cb.record_result("momentum_breakout", is_win=False)
    status = cb.get_status("momentum_breakout", now)
    assert status == BreakerStatus.PAUSED


def test_circuit_breaker_global_stop():
    cb = CircuitBreaker(strategy_pause_losses=5, global_stop_losses=5)
    now = pd.Timestamp("2023-01-01", tz="UTC")
    for _ in range(5):
        cb.record_result("any_strategy", is_win=False)
    status = cb.get_status("any_strategy", now)
    assert status == BreakerStatus.STOPPED


def test_circuit_breaker_resets_on_win():
    cb = CircuitBreaker(strategy_pause_losses=3)
    now = pd.Timestamp("2023-01-01", tz="UTC")
    for _ in range(2):
        cb.record_result("s1", is_win=False)
    cb.record_result("s1", is_win=True)
    assert cb._strategy_losses.get("s1", 0) == 0
