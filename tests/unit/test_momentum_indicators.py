"""RSI 경계값 회귀 테스트."""
from __future__ import annotations

import pandas as pd

from indicators.momentum import rsi


def test_rsi_monotonic_up_down_and_flat_boundaries():
    up = pd.DataFrame({"close": list(range(1, 40))})
    down = pd.DataFrame({"close": list(range(40, 1, -1))})
    flat = pd.DataFrame({"close": [10.0] * 40})

    assert rsi(up).iloc[-1] == 100.0
    assert rsi(down).iloc[-1] == 0.0
    assert rsi(flat).iloc[-1] == 50.0
