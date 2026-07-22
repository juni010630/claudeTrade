"""운영 config의 핵심 키·전략 오타를 fail-closed로 잡는지 검증."""
from __future__ import annotations

import pytest

from scripts.run_backtest import build_engine, validate_params


def _params():
    return {
        "symbols": ["BTCUSDT"],
        "timeframes": ["1h"],
        "primary_timeframe": "1h",
        "strategies": {"ema_cross": {"enabled": False}},
    }


@pytest.mark.parametrize("key", ["symbols", "timeframes", "strategies"])
def test_required_core_config_is_validated(key):
    params = _params()
    del params[key]
    with pytest.raises(ValueError):
        validate_params(params)


def test_unknown_enabled_strategy_is_rejected():
    params = _params()
    params["strategies"]["ema_corss"] = {"enabled": True}
    with pytest.raises(ValueError, match="알 수 없는 활성 전략"):
        build_engine(params, initial_capital=100.0)
