"""컨플루언스 점수 계산기."""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.trend import ema as calc_ema
from indicators.momentum import rsi as calc_rsi
from regime.models import RegimeState
from signals.models import LeverageTier, Signal, SignalScore


class ConfluenceScorer:
    """
    7개 항목 각 +1점, 총점으로 레버리지 등급 결정.
    등급 기준 점수는 params.yaml scorer 섹션에서 제어합니다.
    """

    def __init__(
        self,
        volume_ratio_threshold: float = 1.5,
        rsi_long_max: float = 65.0,
        rsi_short_min: float = 35.0,
        funding_long_max: float = 0.0003,
        funding_short_min: float = -0.0003,
        daily_ema_period: int = 200,
        tier_ss_min_score: int = 7,   # 완벽한 신호 (최고점)
        tier_s_min_score: int = 5,
        tier_a_min_score: int = 3,
        tier_b_min_score: int = 2,
        tier_c_min_score: int = 1,    # 최소 신호 탐색
    ) -> None:
        self.volume_ratio_threshold = volume_ratio_threshold
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.funding_long_max = funding_long_max
        self.funding_short_min = funding_short_min
        self.daily_ema_period = daily_ema_period
        self.tier_ss_min_score = tier_ss_min_score
        self.tier_s_min_score  = tier_s_min_score
        self.tier_a_min_score  = tier_a_min_score
        self.tier_b_min_score  = tier_b_min_score
        self.tier_c_min_score  = tier_c_min_score

    def _map_tier(self, total: int) -> LeverageTier:
        if total >= self.tier_ss_min_score:
            return LeverageTier.SS
        if total >= self.tier_s_min_score:
            return LeverageTier.S
        if total >= self.tier_a_min_score:
            return LeverageTier.A
        if total >= self.tier_b_min_score:
            return LeverageTier.B
        if total >= self.tier_c_min_score:
            return LeverageTier.C
        return LeverageTier.NO_TRADE

    def score(
        self,
        signal: Signal,
        snapshot: MarketSnapshot,
        regime: RegimeState,
    ) -> SignalScore:
        pts: dict[str, int] = {}
        sym = signal.symbol
        direction = signal.direction

        # 1. 국면 강도 (ADX > 25 이면 +1)
        pts["regime_strength"] = 1 if regime.adx > 25 else 0

        # 2. 거래량 확인
        bars_1h = snapshot.bars.get(sym, {}).get("1h", pd.DataFrame())
        if not bars_1h.empty and len(bars_1h) >= 21:
            vol_avg = bars_1h["volume"].iloc[-21:-1].mean()
            current_vol = bars_1h["volume"].iloc[-1]
            pts["volume"] = 1 if (vol_avg > 0 and current_vol / vol_avg >= self.volume_ratio_threshold) else 0
        else:
            pts["volume"] = 0

        # 3. BTC 방향 일치 (BTC가 심볼이 아닐 때만)
        btc_bars = snapshot.bars.get("BTCUSDT", {}).get("1h", pd.DataFrame())
        if sym != "BTCUSDT" and not btc_bars.empty and len(btc_bars) >= 2:
            btc_change = btc_bars["close"].iloc[-1] - btc_bars["close"].iloc[-2]
            aligned = (direction == "long" and btc_change > 0) or (
                direction == "short" and btc_change < 0
            )
            pts["btc_alignment"] = 1 if aligned else 0
        else:
            pts["btc_alignment"] = 0

        # 4. 일봉 EMA 방향
        bars_1d = snapshot.bars.get(sym, {}).get("1d", pd.DataFrame())
        if not bars_1d.empty and len(bars_1d) >= self.daily_ema_period:
            daily_ema = calc_ema(bars_1d, self.daily_ema_period)
            price_vs_ema = bars_1d["close"].iloc[-1] - daily_ema.iloc[-1]
            aligned_ema = (direction == "long" and price_vs_ema > 0) or (
                direction == "short" and price_vs_ema < 0
            )
            pts["daily_ema"] = 1 if aligned_ema else 0
        else:
            pts["daily_ema"] = 0

        # 5. 펀딩비 방향
        funding = snapshot.funding_rates.get(sym, 0.0)
        if direction == "long":
            pts["funding_rate"] = 1 if funding <= self.funding_long_max else 0
        else:
            pts["funding_rate"] = 1 if funding >= self.funding_short_min else 0

        # 6. RSI 과매수/과매도 아님
        if not bars_1h.empty and len(bars_1h) >= 15:
            rsi_val = float(calc_rsi(bars_1h).iloc[-1])
            if direction == "long":
                pts["rsi"] = 1 if rsi_val <= self.rsi_long_max else 0
            else:
                pts["rsi"] = 1 if rsi_val >= self.rsi_short_min else 0
        else:
            pts["rsi"] = 0

        # 7. 4시간봉 방향 일치
        bars_4h = snapshot.bars.get(sym, {}).get("4h", pd.DataFrame())
        if not bars_4h.empty and len(bars_4h) >= 10:
            change_4h = bars_4h["close"].iloc[-1] - bars_4h["close"].iloc[-5]
            aligned_4h = (direction == "long" and change_4h > 0) or (
                direction == "short" and change_4h < 0
            )
            pts["tf_4h"] = 1 if aligned_4h else 0
        else:
            pts["tf_4h"] = 0

        total = sum(pts.values())
        signal.raw_points = pts
        return SignalScore(total=total, tier=self._map_tier(total), signal=signal)
