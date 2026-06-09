"""실시간 국면 스위치 로직 검증 — 합성 추세/횡보 데이터로 분류 정확성 확인.

백테스트가 아니라 '강한 추세 데이터 → 추세장', '진동 데이터 → 횡보장'을 맞추는지
보는 로직 회귀 테스트.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime.realtime_switch import (
    RealtimeRegimeSwitch,
    choppiness_index,
    classify_regime,
    efficiency_ratio,
    wilder_adx,
)


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    wig = np.abs(rng.normal(0, 0.003, n)) * close + 0.05  # 현실적 봉내 변동폭(~0.3%)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + wig,
        "low": close - wig,
        "close": close,
        "volume": np.abs(rng.normal(1000, 100, n)),
    })


def _trending(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 100.0 + np.arange(n) * 0.8 + rng.normal(0, 1.0, n)  # 강한 상승 추세(현실적 노이즈)
    return _ohlcv(close)


def _ranging(n: int = 160) -> pd.DataFrame:
    # 평균회귀 + 노이즈 = 실제 횡보장처럼 좁은 밴드 안에서 톱니(choppy)로 움직임
    rng = np.random.default_rng(2)
    close = np.empty(n)
    close[0] = 100.0
    for i in range(1, n):
        close[i] = close[i - 1] + 0.3 * (100.0 - close[i - 1]) + rng.normal(0, 1.2)
    return _ohlcv(close)


def test_trending_data_classified_as_trend():
    v = classify_regime(_trending())
    assert v.regime == "추세장", f"got {v.regime} score={v.score}"
    assert v.direction == "상승"
    assert v.score >= 1.5
    assert v.confidence >= 0.66


def test_ranging_data_classified_as_range():
    v = classify_regime(_ranging())
    assert v.regime == "횡보장", f"got {v.regime} score={v.score}"
    assert v.score <= -1.5


def test_indicator_thresholds_sane():
    # 추세 데이터: ADX 높고, Choppiness 낮고, ER 높아야 함
    t = _trending()
    assert wilder_adx(t).iloc[-1] > 25
    assert choppiness_index(t).iloc[-1] < 50
    assert efficiency_ratio(t).iloc[-1] > 0.5
    # 횡보 데이터: 반대
    r = _ranging()
    assert wilder_adx(r).iloc[-1] < 25
    assert efficiency_ratio(r).iloc[-1] < 0.3


def test_switch_hysteresis_holds_on_weak_signal():
    sw = RealtimeRegimeSwitch()
    # 강한 추세로 한번 전환
    assert sw.update(_trending()) == "추세장"
    # 짧은 데이터(판별 불가, score 0)에선 직전 상태 유지
    short = _trending(30)
    assert sw.update(short) == "추세장"  # 깜빡이지 않음


def test_insufficient_data_returns_neutral():
    v = classify_regime(_trending(20))
    assert v.regime == "전환/중립"
    assert v.confidence == 0.0
