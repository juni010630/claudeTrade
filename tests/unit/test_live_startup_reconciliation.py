"""라이브 기동 시 state↔거래소 포지션 교차 검증."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from risk.models import Position
from scripts.live_trade import reconcile_saved_positions


def _position(direction="long", entry=100.0, size=500.0):
    return Position(
        symbol="BTCUSDT",
        strategy="ema_cross",
        direction=direction,
        entry_price=entry,
        size_usd=size,
        leverage=3,
        tp_price=110.0,
        sl_price=95.0,
        opened_at=pd.Timestamp("2026-07-22T00:00:00Z"),
    )


def test_matching_position_is_restored():
    pos = _position()
    saved = SimpleNamespace(positions={"BTCUSDT": pos})
    live = {
        "BTCUSDT": {
            "direction": "long",
            "contracts": 5.0,
            "entry_price": 100.0,
            "unrealized_pnl": 0.0,
        }
    }
    assert reconcile_saved_positions(saved, live) == {"BTCUSDT": pos}


@pytest.mark.parametrize(
    "live, message",
    [
        ({"ETHUSDT": {"direction": "long", "contracts": 1, "entry_price": 10}}, "state에 없는"),
        ({"BTCUSDT": {"direction": "short", "contracts": 5, "entry_price": 100}}, "방향"),
        ({"BTCUSDT": {"direction": "long", "contracts": 4, "entry_price": 100}}, "수량"),
        ({"BTCUSDT": {"direction": "long", "contracts": 5, "entry_price": 105}}, "평단"),
    ],
)
def test_orphan_or_material_mismatch_aborts(live, message):
    saved = SimpleNamespace(positions={"BTCUSDT": _position()})
    with pytest.raises(RuntimeError, match=message):
        reconcile_saved_positions(saved, live)


def test_stale_saved_position_is_not_restored_when_exchange_is_flat():
    saved = SimpleNamespace(positions={"BTCUSDT": _position()})
    assert reconcile_saved_positions(saved, {}) == {}
