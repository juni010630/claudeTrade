"""전략 5: Donchian Channel 브레이크아웃 (터틀 트레이딩 기반).

진입:
  롱: 현재 종가 > 이전 N봉 최고가 (새 고점 돌파) + 거래량 확인
  숏: 현재 종가 < 이전 N봉 최저가 (새 저점 돌파) + 거래량 확인

필터:
  - 4h 중간선(mid) 위/아래 → 방향 필터
  - 1d EMA200 (옵션)

TP/SL:
  ATR 배수 기반 (SL은 진입 반대 채널선 또는 ATR × sl_mult 중 작은 쪽)
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.donchian import donchian_channels
from indicators.trend import atr as calc_atr, ema as calc_ema
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian 채널 돌파 + 4h 방향 필터 + 거래량 확인.
    BB 전략과 달리 변동성이 아닌 실제 가격 고/저점을 기준으로 하므로
    변동성 급등 시 가짜 돌파에 덜 취약합니다.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.period: int             = cfg.get("period", 20)
        self.confirm_period: int     = cfg.get("confirm_period", 10)  # 4h 필터용
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.5)
        self.volume_lookback: int    = cfg.get("volume_lookback", 20)
        self.atr_period: int         = cfg.get("atr_period", 14)
        self.atr_tp_mult: float      = cfg.get("atr_tp_mult", 3.0)
        self.atr_sl_mult: float      = cfg.get("atr_sl_mult", 1.0)
        self.daily_ema_period: int   = cfg.get("daily_ema_period", 200)
        self.symbols: list[str]      = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str          = cfg.get("signal_tf", "1h")
        self.confirm_tf: str         = cfg.get("confirm_tf", "4h")
        self.filter_tf: str          = cfg.get("filter_tf", "1d")

    @property
    def name(self) -> str:
        return "donchian_breakout"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if regime.regime not in (MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT):
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df_1h = snapshot.bars.get(sym, {}).get(self.signal_tf)
            df_4h = snapshot.bars.get(sym, {}).get(self.confirm_tf)
            df_1d = snapshot.bars.get(sym, {}).get(self.filter_tf)

            min_bars = max(self.period, self.atr_period) + 5
            if df_1h is None or len(df_1h) < min_bars:
                continue

            try:
                # ── 4h 방향 필터 ──
                bull_bias = True
                bear_bias = True
                if df_4h is not None and len(df_4h) >= self.confirm_period + 2:
                    upper_4h, mid_4h, lower_4h = donchian_channels(df_4h, self.confirm_period)
                    close_4h = float(df_4h["close"].iloc[-1])
                    mid_4h_val = float(mid_4h.iloc[-1])
                    bull_bias = close_4h > mid_4h_val
                    bear_bias = close_4h < mid_4h_val

                # ── 1d EMA 필터 ──
                if self.daily_ema_period > 0 and df_1d is not None and len(df_1d) >= self.daily_ema_period:
                    ema_1d = calc_ema(df_1d, self.daily_ema_period)
                    close_1d = float(df_1d["close"].iloc[-1])
                    ema_1d_val = float(ema_1d.iloc[-1])
                    if bull_bias and close_1d < ema_1d_val:
                        bull_bias = False
                    if bear_bias and close_1d > ema_1d_val:
                        bear_bias = False

                if not bull_bias and not bear_bias:
                    continue

                # ── 1h 돌파 신호 ──
                upper, _, lower = donchian_channels(df_1h, self.period)
                atr = calc_atr(df_1h, self.atr_period)

                curr_close = float(df_1h["close"].iloc[-1])
                prev_close = float(df_1h["close"].iloc[-2])
                # 이전 봉 기준 채널 (look-ahead 방지: iloc[-2] 사용)
                prev_upper = float(upper.iloc[-2])
                prev_lower = float(lower.iloc[-2])
                curr_atr   = float(atr.iloc[-1])

                if curr_atr <= 0:
                    continue

                curr_vol = float(df_1h["volume"].iloc[-1])
                avg_vol  = float(df_1h["volume"].iloc[-self.volume_lookback - 1:-1].mean())
                vol_ok   = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

                # 롱: 이전 봉이 채널 내 → 현재 봉이 채널 위 돌파
                if bull_bias and prev_close <= prev_upper and curr_close > prev_upper and vol_ok:
                    entry = curr_close
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="long",
                        entry_price=entry,
                        tp_price=entry + curr_atr * self.atr_tp_mult,
                        sl_price=entry - curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    ))
                # 숏: 이전 봉이 채널 내 → 현재 봉이 채널 아래 돌파
                elif bear_bias and prev_close >= prev_lower and curr_close < prev_lower and vol_ok:
                    entry = curr_close
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="short",
                        entry_price=entry,
                        tp_price=entry - curr_atr * self.atr_tp_mult,
                        sl_price=entry + curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    ))

            except Exception:
                continue

        return signals
