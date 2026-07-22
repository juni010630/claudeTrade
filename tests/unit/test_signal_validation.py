"""실거래에 등록할 수 없는 TP/SL 가격을 시그널 단계에서 차단한다."""
from __future__ import annotations

import pandas as pd
import pytest

from signals.models import Signal
from signals.validators import validate


def _signal(direction="short", tp=90.0, entry=100.0, sl=110.0):
    return Signal(
        symbol="BTCUSDT",
        strategy="mean_reversion",
        direction=direction,
        entry_price=entry,
        tp_price=tp,
        sl_price=sl,
        timestamp=pd.Timestamp("2026-07-22T00:00:00Z"),
    )


def test_valid_long_and_short_geometry():
    assert validate(_signal())
    assert validate(_signal(direction="long", tp=110.0, sl=90.0))


@pytest.mark.parametrize(
    "signal",
    [
        _signal(tp=-1.0),
        _signal(sl=0.0),
        _signal(tp=105.0, entry=100.0, sl=110.0),
        _signal(direction="long", tp=90.0, entry=100.0, sl=80.0),
    ],
)
def test_nonpositive_or_wrong_side_protection_is_rejected(signal):
    assert not validate(signal)
