"""multi_tf 1d EMA 필터의 빈 프레임 fail-closed 검증 (2026-07-04 2차 리뷰 MEDIUM②).

회귀 대상: 라이브 1d 피드 저하 시 live_feed가 프레임을 비우는데(iloc[0:0]),
1d 필터가 조용히 꺼진 채 신호가 통과(fail-open)했음. 백테 동작(None=1d 미로드,
상장 초기 len<period=필터 생략)은 변경 없어야 함.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from data.schemas import MarketSnapshot
from regime.models import MarketRegime
from strategies.multi_tf_breakout import MultiTFBreakoutStrategy


def _flat_breakout_df(n: int = 60) -> pd.DataFrame:
    """준평탄(±0.3 교대 — RSI loss=0 NaN 방지) 후 마지막 봉 돌파 + 볼륨 5x."""
    ts = pd.date_range("2026-06-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + 0.3 * (-1.0) ** np.arange(n)
    high = close + 0.5
    low = close - 0.5
    volume = np.full(n, 1000.0)
    close[-1], high[-1], volume[-1] = 120.0, 121.0, 5000.0
    return pd.DataFrame({"timestamp": ts, "open": np.full(n, 100.0),
                         "high": high, "low": low, "close": close, "volume": volume})


def _signals(df_1d) -> list:
    strat = MultiTFBreakoutStrategy({"symbols": ["BTCUSDT"]})
    bars = {"1h": _flat_breakout_df(), "4h": _flat_breakout_df()}
    if df_1d is not None:
        bars["1d"] = df_1d
    snap = MarketSnapshot(
        timestamp=pd.Timestamp("2026-07-04 12:00", tz="UTC"),
        bars={"BTCUSDT": bars}, funding_rates={},
    )
    regime = SimpleNamespace(regime=MarketRegime.TRENDING)
    return strat.generate_signals(snap, regime)


def _daily_df(n: int, start: float = 100.0, end: float = 100.0) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    close = np.linspace(start, end, n)
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": np.full(n, 1000.0)})


def test_empty_1d_frame_fails_closed():
    """빈 1d 프레임(라이브 피드 저하) → 신호 억제."""
    assert _signals(_daily_df(250).iloc[0:0]) == []


def test_missing_1d_unchanged():
    """1d 미로드(None) → 기존 동작 유지 = 필터 없이 신호."""
    (sig,) = _signals(None)
    assert sig.direction == "long"


def test_short_1d_history_unchanged():
    """상장 초기(len < daily_ema_period) → 기존 동작 유지 = 필터 생략."""
    (sig,) = _signals(_daily_df(50))
    assert sig.direction == "long"


def test_long_downtrend_1d_still_filters():
    """1d 하락추세(close < EMA200) 롱 필터 — 기존 필터 동작 sanity."""
    assert _signals(_daily_df(250, start=300.0, end=100.0)) == []
