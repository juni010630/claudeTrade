"""CircuitBreaker 연속손절·정지 타이머와 TP 쿨다운이 라이브 재기동(state_store)에서 보존되는지 검증.

회귀 대상(F37/F33): save/load가 PortfolioState만 직렬화하고 CB/쿨다운은 인메모리라
systemd Restart=always 재기동마다 손실 방어 가드가 0으로 리셋되던 버그.
"""
from __future__ import annotations

import pandas as pd

from portfolio import state_store
from risk.circuit_breaker import BreakerStatus, CircuitBreaker
from risk.guards import RiskGuards
from risk.models import PortfolioState


class _FakeEngine:
    def __init__(self, cb, guards):
        self.circuit_breaker = cb
        self.guards = guards


def test_losses_and_cooldown_persist_across_restart(tmp_path):
    now = pd.Timestamp("2026-06-09T03:00:00Z")
    cb = CircuitBreaker(strategy_pause_losses=5, global_stop_losses=10, pause_duration_hours=48)
    for _ in range(4):  # 4연패 (아직 PAUSE 직전)
        cb.record_result("ema_cross", is_win=False)
    guards = RiskGuards(tp_cooldown_hours=6.0)
    guards.record_tp("BTCUSDT", "ema_cross", now)

    state = PortfolioState(equity=7400.0, cash=7400.0, daily_start_equity=7400.0, positions={})
    path = tmp_path / "state.json"
    state_store.save(state, path, engine=_FakeEngine(cb, guards))

    # 재기동: 빈 컴포넌트로 새 엔진 → restore_runtime
    cb2 = CircuitBreaker(strategy_pause_losses=5, global_stop_losses=10, pause_duration_hours=48)
    guards2 = RiskGuards(tp_cooldown_hours=6.0)
    state_store.restore_runtime(_FakeEngine(cb2, guards2), path)

    # 연속손절 카운터 보존 → 1번 더 지면 즉시 PAUSE (리셋됐다면 아직 1패라 ACTIVE였을 것)
    assert cb2._strategy_losses["ema_cross"] == 4
    assert cb2._global_losses == 4
    cb2.record_result("ema_cross", is_win=False)  # 5연패
    assert cb2.get_status("ema_cross", now) == BreakerStatus.PAUSED

    # TP 쿨다운 보존 → 6h 내 재진입 차단, 6h 후 해제
    assert guards2.is_cooldown_active("BTCUSDT", "ema_cross", now + pd.Timedelta(hours=1))
    assert not guards2.is_cooldown_active("BTCUSDT", "ema_cross", now + pd.Timedelta(hours=7))


def test_pause_timer_persists_across_restart(tmp_path):
    now = pd.Timestamp("2026-06-09T03:00:00Z")
    cb = CircuitBreaker(strategy_pause_losses=5, global_stop_losses=10, pause_duration_hours=48)
    for _ in range(5):
        cb.record_result("multi_tf_breakout", is_win=False)
    assert cb.get_status("multi_tf_breakout", now) == BreakerStatus.PAUSED  # 48h 타이머 무장

    path = tmp_path / "s.json"
    state_store.save(
        PortfolioState(equity=100.0, cash=100.0, daily_start_equity=100.0, positions={}),
        path, engine=_FakeEngine(cb, RiskGuards()),
    )

    cb2 = CircuitBreaker(strategy_pause_losses=5, global_stop_losses=10, pause_duration_hours=48)
    state_store.restore_runtime(_FakeEngine(cb2, RiskGuards()), path)
    # 48h 정지 타이머 보존 → 1h 뒤 여전히 PAUSED, 49h 뒤 자동 해제
    assert cb2.get_status("multi_tf_breakout", now + pd.Timedelta(hours=1)) == BreakerStatus.PAUSED
    assert cb2.get_status("multi_tf_breakout", now + pd.Timedelta(hours=49)) == BreakerStatus.ACTIVE


def test_restore_runtime_noop_on_legacy_state(tmp_path):
    """구버전 state.json(CB 필드 없음) 복원 시 no-op — 빈 상태 유지, 예외 없음."""
    state_store.save(
        PortfolioState(equity=100.0, cash=100.0, daily_start_equity=100.0, positions={}),
        tmp_path / "legacy.json",  # engine 미전달 → CB/guards 키 없음
    )
    cb = CircuitBreaker()
    guards = RiskGuards()
    state_store.restore_runtime(_FakeEngine(cb, guards), tmp_path / "legacy.json")
    assert cb._global_losses == 0
    assert guards._last_tp_times == {}
