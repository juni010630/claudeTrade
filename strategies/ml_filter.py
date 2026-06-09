"""ML 메타라벨 필터 — López de Prado 메타라벨링 패턴.

scorer.py에서 compute_features()와 MLSignalFilter.predict()를 호출.
학습 시에도 동일한 compute_features()를 사용해 피처 일관성 보장.

피처 8종 (bars 기반, 무누수):
  adx, bb_width_pct, rsi_1h, vol_ratio, atr_pct,
  dist_ema200_1d, hour_sin, hour_cos

컨텍스트 피처 3종 (caller가 feat dict에 추가):
  direction_long, strategy_ema, funding
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from indicators.trend import ema as calc_ema, atr as calc_atr, adx as calc_adx
from indicators.momentum import rsi as calc_rsi

# 최종 피처 순서 (학습·추론 동일 보장)
FEATURE_COLS = [
    "adx", "bb_width_pct", "rsi_1h", "vol_ratio", "atr_pct",
    "dist_ema200_1d", "hour_sin", "hour_cos",
    "direction_long", "strategy_ema", "funding",
]


def compute_features(
    bars_1h: pd.DataFrame,
    bars_4h: pd.DataFrame,
    bars_1d: pd.DataFrame,
    idx: int,
) -> dict | None:
    """진입봉 idx까지의 데이터로 피처 계산. 데이터 부족 시 None.

    bars_*: reset_index() 된 DataFrame (timestamp 컬럼 존재).
    idx: 진입봉 인덱스 (0-based). 내부에서 bars_1h.iloc[:idx+1]만 사용.
    반환값에 direction_long / strategy_ema / funding은 포함 안 함 — caller가 추가.
    """
    if len(bars_1h) < 30 or idx < 29:
        return None

    bars = bars_1h.iloc[:idx + 1]

    close = bars["close"]
    n = len(bars)

    # 1. ADX(14)
    adx_val = 0.0
    if n >= 14:
        v = calc_adx(bars, 14).iloc[-1]
        adx_val = float(v) if np.isfinite(v) else 0.0

    # 2. BB 밴드폭 % = 4σ / mid × 100 (period=25)
    bb_width_pct = 0.0
    if n >= 25:
        mid = close.rolling(25).mean().iloc[-1]
        std = close.rolling(25).std().iloc[-1]
        if mid > 0 and np.isfinite(std):
            bb_width_pct = float(4.0 * std / mid * 100)

    # 3. RSI(14)
    rsi_val = 50.0
    if n >= 15:
        v = calc_rsi(bars, 14).iloc[-1]
        rsi_val = float(v) if np.isfinite(v) else 50.0

    # 4. 거래량 비율 (현재 / 20봉 평균)
    vol_ratio = 1.0
    if n >= 21:
        vol_avg = bars["volume"].iloc[-21:-1].mean()
        vol_cur = bars["volume"].iloc[-1]
        if vol_avg > 0:
            vol_ratio = float(vol_cur / vol_avg)

    # 5. ATR% = ATR(14) / close
    atr_pct = 0.0
    if n >= 14:
        av = calc_atr(bars, 14).iloc[-1]
        cv = float(close.iloc[-1])
        if cv > 0 and np.isfinite(av):
            atr_pct = float(av / cv)

    # 6. 일봉 EMA200 거리 (부호 있음: 위=양수)
    dist_ema200_1d = 0.0
    if len(bars_1d) >= 200:
        ev = calc_ema(bars_1d, 200).iloc[-1]
        cv = float(bars_1d["close"].iloc[-1])
        if cv > 0 and np.isfinite(ev):
            dist_ema200_1d = float((cv - ev) / cv)

    # 7. UTC 시간대 (순환 인코딩)
    if "timestamp" in bars.columns:
        last_ts = bars["timestamp"].iloc[-1]
    else:
        last_ts = bars.index[-1]
    hour = pd.Timestamp(last_ts).hour
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))

    return {
        "adx": adx_val,
        "bb_width_pct": bb_width_pct,
        "rsi_1h": rsi_val,
        "vol_ratio": vol_ratio,
        "atr_pct": atr_pct,
        "dist_ema200_1d": dist_ema200_1d,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
    }


class MLModels:
    """학습된 모델 + 전처리기 번들. 직렬화·역직렬화 담당."""

    def __init__(
        self,
        clf: Any,
        scaler: Any,
        feature_cols: list[str] | None = None,
        model_type: str = "unknown",
    ) -> None:
        self.clf = clf
        self.scaler = scaler
        self.feature_cols = feature_cols or FEATURE_COLS
        self.model_type = model_type

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MLModels":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"pkl이 MLModels 타입이 아님: {type(obj)}")
        return obj


class MLSignalFilter:
    """진입 후보에 대한 P(win) 예측기.

    scorer.py 인터페이스:
        pred = ml_filter.predict(feat)   # feat: dict (compute_features 반환값 + 컨텍스트)
        clf_prob = pred["clf_prob"]      # float [0, 1]
    """

    def __init__(self, models: MLModels, clf_threshold: float = 0.0) -> None:
        self.models = models
        self.clf_threshold = clf_threshold  # hardcut 임계값 (0.0 = 비활성)

    def predict(self, feat: dict) -> dict:
        """feat dict → {"clf_prob": float}."""
        cols = self.models.feature_cols
        # 누락 피처는 0으로 채움
        X = np.array([[feat.get(c, 0.0) for c in cols]], dtype=float)
        X = self.models.scaler.transform(X)
        prob = float(self.models.clf.predict_proba(X)[0, 1])
        return {"clf_prob": prob}
