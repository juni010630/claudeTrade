"""라이브 거래소-sync 청산이 서킷브레이커에 승/패를 기록하는지 검증.

회귀 대상 버그: 메인넷에서 거래소측 TP/SL 체결로 사라진 포지션을 _process_bar의
sync 경로(engine/backtest.py)가 정리할 때 circuit_breaker.record_result를 호출하지
않아, 백테(_close_with_reason)와 달리 CB 연속손절 카운터가 동결되던 패리티 결함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.schemas import MarketSnapshot
from engine.backtest import BacktestEngine
from execution.commission import CommissionModel
from execution.slippage import SlippageModel
from risk.models import Position


class _FakeLiveBroker:
    """fetch_open_symbols를 가진 라이브 유사 브로커 — 엔진의 sync 경로를 트리거한다."""

    def __init__(self, open_syms, fill_price):
        self._open_syms = set(open_syms)
        self._fill_price = fill_price
        self.commission = CommissionModel()
        self.slippage = SlippageModel()

    def fetch_open_symbols(self):
        return self._open_syms

    def fetch_recent_fill_price(self, sym):
        return self._fill_price

    def cancel_all_orders(self, sym):
        pass


def _make_df(n: int = 320) -> pd.DataFrame:
    np.random.seed(0)
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 30000.0 + np.cumsum(np.random.randn(n) * 50)
    high = close + np.abs(np.random.randn(n) * 80) + 50
    low = close - np.abs(np.random.randn(n) * 80) - 50
    open_ = close + np.random.randn(n) * 40
    volume = np.abs(np.random.randn(n) * 2000 + 8000)
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _snapshot() -> MarketSnapshot:
    df_1h = _make_df(320)
    df_4h = _make_df(60)
    df_1d = _make_df(30)
    return MarketSnapshot(
        timestamp=df_1h["timestamp"].iloc[-1],
        bars={"BTCUSDT": {"1h": df_1h, "4h": df_4h, "1d": df_1d}},
        funding_rates={"BTCUSDT": 0.0},
    )


def _engine_with_stale_position(fill_price: float, entry_price: float = 30000.0) -> BacktestEngine:
    """BTCUSDT 롱을 tracker에 주입하고, 거래소엔 없는(=sync로 청산될) 상태로 만든다."""
    broker = _FakeLiveBroker(open_syms=set(), fill_price=fill_price)  # 거래소에 포지션 없음
    engine = BacktestEngine(initial_capital=100_000, strategies=[], broker=broker)
    snap = _snapshot()
    engine.tracker.state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        strategy="test_strat",
        direction="long",
        entry_price=entry_price,
        size_usd=1000.0,
        leverage=10,
        tp_price=40000.0,   # fill_price와 멀어 external_close로 분류됨 (SL 슬리피지 케이스 모사)
        sl_price=27000.0,
        opened_at=snap.timestamp - pd.Timedelta(hours=5),
    )
    return engine, snap


def test_sync_close_loss_increments_circuit_breaker():
    # fill 28000 < entry 30000 → 손실. 백테와 동일하게 CB에 손실로 기록되어야 함.
    engine, snap = _engine_with_stale_position(fill_price=28000.0)
    engine._process_bar(snap)

    # 포지션은 sync로 청산됨
    assert "BTCUSDT" not in engine.tracker.snapshot().positions
    # ledger에 손실 거래 기록
    assert engine.ledger._records[-1].pnl < 0
    # CB 연속손절 카운터 증가 (이전엔 동결되어 0이던 회귀 버그)
    assert engine.circuit_breaker._global_losses == 1
    assert engine.circuit_breaker._strategy_losses["test_strat"] == 1


def test_sync_close_win_resets_circuit_breaker():
    # fill 33000 > entry 30000 → 이익. 백테와 동일하게 CB 카운터가 리셋되어야 함.
    engine, snap = _engine_with_stale_position(fill_price=33000.0)
    # 사전에 연속손절 누적 상태를 만들어 둠
    engine.circuit_breaker._global_losses = 3
    engine.circuit_breaker._strategy_losses["test_strat"] = 3

    engine._process_bar(snap)

    assert engine.ledger._records[-1].pnl > 0
    assert engine.circuit_breaker._global_losses == 0
    assert engine.circuit_breaker._strategy_losses["test_strat"] == 0
