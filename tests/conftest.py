"""공유 pytest 픽스처."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """300봉짜리 합성 OHLCV 데이터."""
    np.random.seed(42)
    n = 300
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 30000.0 + np.cumsum(np.random.randn(n) * 100)
    high = close + np.abs(np.random.randn(n) * 50)
    low = close - np.abs(np.random.randn(n) * 50)
    open_ = close + np.random.randn(n) * 30
    volume = np.abs(np.random.randn(n) * 1000 + 5000)
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """명확한 상승 추세 OHLCV."""
    n = 300
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 30000.0 + np.arange(n) * 20.0  # 꾸준한 상승
    high = close + 100
    low = close - 100
    open_ = close - 10
    volume = np.ones(n) * 5000
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.fixture
def ranging_ohlcv() -> pd.DataFrame:
    """횡보 OHLCV."""
    np.random.seed(7)
    n = 300
    timestamps = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 30000.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 500
    high = close + 50
    low = close - 50
    open_ = close + np.random.randn(n) * 10
    volume = np.ones(n) * 3000
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )
