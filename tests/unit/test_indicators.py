"""지표 함수 단위 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.momentum import rsi, volume_ratio
from indicators.trend import adx, atr, ema, macd
from indicators.volatility import bb_width, bollinger_bands


def test_ema_length(sample_ohlcv):
    result = ema(sample_ohlcv, 20)
    assert len(result) == len(sample_ohlcv)


def test_ema_smoothing(sample_ohlcv):
    e9 = ema(sample_ohlcv, 9)
    e21 = ema(sample_ohlcv, 21)
    # 빠른 EMA가 느린 EMA보다 변동성 큼
    assert e9.std() >= e21.std()


def test_macd_returns_tuple(sample_ohlcv):
    macd_line, signal_line, hist = macd(sample_ohlcv)
    assert len(macd_line) == len(sample_ohlcv)
    assert len(signal_line) == len(sample_ohlcv)
    # 히스토그램 = macd - signal
    pd.testing.assert_series_equal(hist, macd_line - signal_line, check_names=False)


def test_atr_positive(sample_ohlcv):
    result = atr(sample_ohlcv)
    assert (result.dropna() > 0).all()


def test_adx_range(sample_ohlcv):
    result = adx(sample_ohlcv)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_bollinger_bands_order(sample_ohlcv):
    upper, mid, lower = bollinger_bands(sample_ohlcv)
    valid = upper.dropna().index
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_bb_width_positive(sample_ohlcv):
    bw = bb_width(sample_ohlcv)
    assert (bw.dropna() > 0).all()


def test_rsi_range(sample_ohlcv):
    result = rsi(sample_ohlcv)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_volume_ratio(sample_ohlcv):
    vr = volume_ratio(sample_ohlcv)
    valid = vr.dropna()
    assert (valid > 0).all()
