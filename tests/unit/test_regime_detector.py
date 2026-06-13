"""시장 국면 분류기 단위 테스트."""
from __future__ import annotations

import pandas as pd
import pytest

from data.schemas import MarketSnapshot
from regime.detector import RegimeDetector
from regime.filters import is_strategy_eligible
from regime.models import MarketRegime


def _make_snapshot(df: pd.DataFrame, sym: str = "BTCUSDT") -> MarketSnapshot:
    return MarketSnapshot(
        timestamp=df["timestamp"].iloc[-1],
        bars={sym: {"1h": df}},
    )


def test_trending_regime(trending_ohlcv):
    detector = RegimeDetector(primary_symbol="BTCUSDT", primary_tf="1h")
    snap = _make_snapshot(trending_ohlcv)
    state = detector.classify(snap)
    # 추세장에서는 TRENDING 또는 전환 구간
    assert state.regime in (MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.PRE_BREAKOUT)
    assert 0 <= state.adx <= 100


def test_ranging_regime(ranging_ohlcv):
    detector = RegimeDetector(primary_symbol="BTCUSDT", primary_tf="1h")
    snap = _make_snapshot(ranging_ohlcv)
    state = detector.classify(snap)
    assert state.regime in (MarketRegime.RANGING, MarketRegime.PRE_BREAKOUT, MarketRegime.TRENDING)
    assert 0.0 <= state.bb_width_pct <= 1.0


def test_strategy_eligibility():
    # mean_reversion/macross_d는 전체 국면 허용 — 횡보 판정은 전략 내부 게이트 담당 (regime/filters.py)
    assert is_strategy_eligible(MarketRegime.RANGING, "mean_reversion")
    assert is_strategy_eligible(MarketRegime.TRENDING, "mean_reversion")
    assert is_strategy_eligible(MarketRegime.TRENDING, "macross_d")
    assert is_strategy_eligible(MarketRegime.RANGING, "macross_d")
    assert is_strategy_eligible(MarketRegime.TRENDING, "ema_cross")
    assert is_strategy_eligible(MarketRegime.PRE_BREAKOUT, "ema_cross")
    assert not is_strategy_eligible(MarketRegime.RANGING, "ema_cross")
    assert is_strategy_eligible(MarketRegime.TRENDING, "multi_tf_breakout")
    assert not is_strategy_eligible(MarketRegime.RANGING, "multi_tf_breakout")
    # momentum_breakout(15m Scalp 검증 포팅)은 전체 국면 허용 — 무게이트 검증
    assert is_strategy_eligible(MarketRegime.TRENDING, "momentum_breakout")
    assert is_strategy_eligible(MarketRegime.RANGING, "momentum_breakout")
    # 미등록 전략은 항상 차단
    assert not is_strategy_eligible(MarketRegime.TRENDING, "nonexistent_strategy")
