"""시장 국면 분류기."""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.trend import adx as calc_adx
from indicators.volatility import bb_width as calc_bb_width
from regime.models import MarketRegime, RegimeState


class RegimeDetector:
    def __init__(
        self,
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        bb_period: int = 20,
        bb_std: float = 2.0,
        bb_width_lookback: int = 50,
        bb_width_squeeze_pct: float = 0.2,  # 하위 20% → 수렴
        primary_symbol: str = "BTCUSDT",
        primary_tf: str = "1h",
    ) -> None:
        self.adx_period = adx_period
        self.adx_trending = adx_trending_threshold
        self.adx_ranging = adx_ranging_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_lookback = bb_width_lookback
        self.bb_width_squeeze_pct = bb_width_squeeze_pct
        self.primary_symbol = primary_symbol
        self.primary_tf = primary_tf

    def classify(self, snapshot: MarketSnapshot) -> RegimeState:
        df = snapshot.bars[self.primary_symbol][self.primary_tf]

        adx_series = calc_adx(df, self.adx_period)
        bw_series = calc_bb_width(df, self.bb_period, self.bb_std)

        current_adx = float(adx_series.iloc[-1])
        current_bw = float(bw_series.iloc[-1])

        # BB 폭의 최근 N봉 중 백분위 계산
        recent_bw = bw_series.iloc[-self.bb_width_lookback :]
        pct_rank = float((recent_bw < current_bw).mean())  # 0~1

        # 국면 분류
        if current_adx > self.adx_trending:
            regime = MarketRegime.TRENDING
        elif current_adx < self.adx_ranging and pct_rank < self.bb_width_squeeze_pct:
            regime = MarketRegime.PRE_BREAKOUT
        elif current_adx < self.adx_ranging:
            regime = MarketRegime.RANGING
        else:
            # 전환 구간 (20~25): 추세장으로 처리
            regime = MarketRegime.TRENDING

        return RegimeState(
            regime=regime,
            adx=current_adx,
            bb_width=current_bw,
            bb_width_pct=pct_rank,
            timestamp=snapshot.timestamp,
        )
