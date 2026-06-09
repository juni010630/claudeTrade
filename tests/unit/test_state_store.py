"""state_store 저장/복원 라운드트립 — Position 전 필드 보존 검증.

회귀 대상: save()/load()가 adds_done·order_price를 누락해 재기동 시 피라미딩 카운트와
슬리피지 기준가가 0으로 리셋되던 버그.
"""
from __future__ import annotations

import pandas as pd

from portfolio import state_store
from risk.models import PortfolioState, Position


def test_roundtrip_preserves_all_position_fields(tmp_path):
    pos = Position(
        symbol="BTCUSDT", strategy="ema_cross", direction="long",
        entry_price=30000.0, size_usd=1000.0, leverage=10,
        tp_price=33000.0, sl_price=29000.0,
        opened_at=pd.Timestamp("2026-06-09T00:00:00Z"),
        unrealized_pnl=12.5, funding_paid=0.3, peak_price=31000.0,
        initial_sl_price=28800.0, entry_commission=0.5, entry_slippage=0.2,
        confluence_score=5, adds_done=2, order_price=29990.0,
    )
    state = PortfolioState(equity=7451.8, cash=6000.0, daily_start_equity=7400.0,
                           positions={"BTCUSDT": pos})

    path = tmp_path / "state.json"
    state_store.save(state, path)
    loaded = state_store.load(path)

    assert loaded is not None
    rp = loaded.positions["BTCUSDT"]
    # 이전 회귀에서 0으로 리셋되던 두 필드
    assert rp.adds_done == 2
    assert rp.order_price == 29990.0
    # 나머지 필드도 정확 복원
    assert rp.entry_price == 30000.0
    assert rp.size_usd == 1000.0
    assert rp.confluence_score == 5
    assert loaded.daily_start_equity == 7400.0
