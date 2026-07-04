"""FundingRateSimulator 정산 ts 전진 감지 검증 (2026-07-04 MEDIUM③: 4h 정산 심볼).

핵심: funding_ts(직전 정산 시각)가 전진하면 = 새 정산 발생 → 그 rate 1회 부과.
심볼별 4h/8h 주기·과거 전환 이력을 데이터로 자동 추적, ts 없으면 기존 8h 버킷 폴백.
"""
from __future__ import annotations

import pandas as pd
import pytest

from execution.funding import FundingRateSimulator
from risk.models import PortfolioState, Position


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _state(direction: str = "long", opened: str = "2026-01-01 10:00") -> PortfolioState:
    pos = Position(
        symbol="LPTUSDT", strategy="s", direction=direction,
        entry_price=100.0, size_usd=1000.0, leverage=3,
        tp_price=110.0, sl_price=90.0, opened_at=_ts(opened),
    )
    return PortfolioState(equity=10_000, cash=9_000, positions={"LPTUSDT": pos})


def _run_hours(sim, state, start, hours, settle_every, rate=0.0001):
    """1h 봉 루프 시뮬 — settle_every 시간마다 정산 행이 나타나는 심볼."""
    charged = []
    t0 = _ts(start)
    for h in range(1, hours + 1):
        now = t0 + pd.Timedelta(hours=h)
        # 직전 정산 = now 이하의 가장 최근 settle_every 배수 시각 (as-of 룩업 모사)
        fts = now.floor(f"{settle_every}h")
        acc = sim.accrue(state, now, {"LPTUSDT": rate},
                         funding_ts={"LPTUSDT": fts})
        if acc:
            charged.append((now, acc["LPTUSDT"]))
    return charged


def test_4h_symbol_six_settlements_per_day():
    """4h 정산 심볼 → 하루 6회 부과 (기존 8h 버킷은 3회만 잡던 버그)."""
    sim = FundingRateSimulator()
    state = _state(opened="2026-01-01 00:00")
    charged = _run_hours(sim, state, "2026-01-01 00:00", 24, settle_every=4)
    assert len(charged) == 6
    assert [c[0].hour for c in charged] == [4, 8, 12, 16, 20, 0]
    # 롱 + rate 양수 = 지불(양수), notional 1000 × 0.0001 = 0.1
    assert all(c[1] == pytest.approx(0.1) for c in charged)


def test_8h_symbol_three_settlements_per_day():
    sim = FundingRateSimulator()
    state = _state(direction="short", opened="2026-01-01 00:00")
    charged = _run_hours(sim, state, "2026-01-01 00:00", 24, settle_every=8)
    assert len(charged) == 3
    assert all(c[1] == pytest.approx(-0.1) for c in charged)  # 숏 = 수취(음수)


def test_interval_transition_8h_to_4h():
    """8h→4h 전환(바이낸스 실제 이력) — 전환 지점부터 자동으로 6회/일."""
    sim = FundingRateSimulator()
    state = _state(opened="2026-01-01 00:00")
    day1 = _run_hours(sim, state, "2026-01-01 00:00", 24, settle_every=8)
    day2 = _run_hours(sim, state, "2026-01-02 00:00", 24, settle_every=4)
    assert len(day1) == 3 and len(day2) == 6


def test_no_charge_before_entry():
    """진입 이전(또는 진입 당봉) 정산은 미부과 — 재진입 시 과거 정산 소급 금지."""
    sim = FundingRateSimulator()
    state = _state(opened="2026-01-01 10:00")
    # 08:00 정산(진입 전, ms 지터 포함), 진입 10:00 → 11:00 봉 첫 관측: 부과 없음
    acc = sim.accrue(state, _ts("2026-01-01 11:00"), {"LPTUSDT": 0.0001},
                     funding_ts={"LPTUSDT": _ts("2026-01-01 08:00:00.005")})
    assert acc == {}
    # 16:00 정산(보유 중, 지터로 17:00 봉에서 관측) → 부과
    acc = sim.accrue(state, _ts("2026-01-01 17:00"), {"LPTUSDT": 0.0001},
                     funding_ts={"LPTUSDT": _ts("2026-01-01 16:00:00.005")})
    assert acc["LPTUSDT"] == pytest.approx(0.1)


def test_restart_sync_no_double_charge():
    """재기동 직후 관측되는 과거 정산은 잔고 재앵커에 이미 반영 → 중복부과 금지."""
    sim = FundingRateSimulator()
    sim.sync_to(_ts("2026-01-01 16:03"))
    state = _state(opened="2026-01-01 02:00")  # 재기동 전부터 보유
    acc = sim.accrue(state, _ts("2026-01-01 17:00"), {"LPTUSDT": 0.0001},
                     funding_ts={"LPTUSDT": _ts("2026-01-01 16:00:00.005")})
    assert acc == {}  # 16:00 정산은 재기동(16:03) 이전
    acc = sim.accrue(state, _ts("2026-01-01 20:00"), {"LPTUSDT": 0.0001},
                     funding_ts={"LPTUSDT": _ts("2026-01-01 20:00")})
    assert acc["LPTUSDT"] == pytest.approx(0.1)  # 재기동 후 첫 정산부터 정상


def test_fallback_bucket_without_ts():
    """funding_ts 없는 심볼(라이브 API 폴백) → 기존 8h 버킷 동작 유지."""
    sim = FundingRateSimulator()
    # 엔진 시작 모사: 첫 봉엔 포지션 없음 (첫 호출 = 버킷 초기화, 기존과 동일)
    sim.accrue(PortfolioState(equity=10_000, cash=10_000), _ts("2026-01-01 00:00"), {})
    state = _state(opened="2026-01-01 00:00")
    charged = []
    for h in range(1, 25):
        now = _ts("2026-01-01 00:00") + pd.Timedelta(hours=h)
        acc = sim.accrue(state, now, {"LPTUSDT": 0.0001})  # ts 미제공
        if acc:
            charged.append(now)
    assert [t.hour for t in charged] == [8, 16, 0]  # UTC 8h 버킷 3회


def test_same_settlement_charged_once():
    sim = FundingRateSimulator()
    state = _state(opened="2026-01-01 00:00")
    fts = {"LPTUSDT": _ts("2026-01-01 08:00:00.005")}
    acc1 = sim.accrue(state, _ts("2026-01-01 09:00"), {"LPTUSDT": 0.0001}, funding_ts=fts)
    acc2 = sim.accrue(state, _ts("2026-01-01 10:00"), {"LPTUSDT": 0.0001}, funding_ts=fts)
    assert acc1["LPTUSDT"] == pytest.approx(0.1) and acc2 == {}
