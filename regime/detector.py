"""시장 국면 분류기."""
from __future__ import annotations

import math
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
        self._last_state: RegimeState | None = None

    def classify(self, snapshot: MarketSnapshot) -> RegimeState:
        df = snapshot.bars.get(self.primary_symbol, {}).get(self.primary_tf)
        if df is None or len(df) <= self.adx_period + 1:
            # 기준 심볼(ETH) 프레임 부재/부족 (라이브 fetch 실패로 심볼 제외 등) —
            # iloc[-1] 크래시로 봇 전체가 죽는 대신 직전 국면 유지, 콜드스타트는
            # RANGING(추세 진입 차단 = 보수) 폴백. 백테 로더는 항상 프레임 제공 → 미발동.
            if self._last_state is not None:
                return self._last_state
            return RegimeState(
                regime=MarketRegime.RANGING, adx=0.0, bb_width=0.0,
                bb_width_pct=1.0, timestamp=snapshot.timestamp,
            )

        adx_series = calc_adx(df, self.adx_period)
        bw_series = calc_bb_width(df, self.bb_period, self.bb_std)

        current_adx = float(adx_series.iloc[-1])
        current_bw = float(bw_series.iloc[-1])

        # 지표 NaN/무한대는 비교식을 모두 False로 만들어 기존 else의
        # TRENDING으로 새었다. 불완전 데이터에서는 진입을 줄이도록 fail-closed.
        if not math.isfinite(current_adx) or not math.isfinite(current_bw):
            state = RegimeState(
                regime=MarketRegime.RANGING,
                adx=current_adx if math.isfinite(current_adx) else 0.0,
                bb_width=current_bw if math.isfinite(current_bw) else 0.0,
                bb_width_pct=1.0,
                timestamp=snapshot.timestamp,
            )
            self._last_state = state
            return state

        # BB 폭의 최근 N봉 중 백분위 계산
        recent_bw = bw_series.iloc[-self.bb_width_lookback :]
        recent_bw = recent_bw.dropna()
        pct_rank = float((recent_bw < current_bw).mean()) if len(recent_bw) else 1.0

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

        state = RegimeState(
            regime=regime,
            adx=current_adx,
            bb_width=current_bw,
            bb_width_pct=pct_rank,
            timestamp=snapshot.timestamp,
        )
        self._last_state = state
        return state
