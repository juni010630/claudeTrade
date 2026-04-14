"""전략 1: 볼린저밴드 돌파 + 거래량 + ATR TP/SL (추세장 전용)."""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.trend import atr as calc_atr
from indicators.volatility import bollinger_bands
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    """
    조건:
      롱: 이전 봉 종가 < BB 상단, 현재 봉 종가 > BB 상단 + 거래량 확인
      숏: 이전 봉 종가 > BB 하단, 현재 봉 종가 < BB 하단 + 거래량 확인
    TP/SL: ATR 배수 기반
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std: float = cfg.get("bb_std", 2.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.5)
        self.volume_lookback: int = cfg.get("volume_lookback", 20)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 2.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "1h")
        # 4h 추세 확인 (옵션) — 설정 시 4h BB 방향과 일치하는 신호만 허용
        self.confirm_tf: str | None = cfg.get("confirm_tf", None)
        self.confirm_bb_std: float = cfg.get("confirm_bb_std", 2.0)
        # 1d EMA 필터 (옵션)
        self.daily_ema_period: int = cfg.get("daily_ema_period", 0)  # 0 = 비활성

    @property
    def name(self) -> str:
        return "momentum_breakout"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if regime.regime != MarketRegime.TRENDING:
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.timeframe)
            if df is None or len(df) < max(self.bb_period, self.atr_period) + 2:
                continue

            # ── 4h 추세 확인 (옵션) ──
            bull_bias = True   # 기본: 양방향 허용
            bear_bias = True
            if self.confirm_tf:
                df_c = snapshot.bars.get(sym, {}).get(self.confirm_tf)
                if df_c is not None and len(df_c) >= self.bb_period + 2:
                    upper_c, mid_c, lower_c = bollinger_bands(df_c, self.bb_period, self.confirm_bb_std)
                    close_c = float(df_c["close"].iloc[-1])
                    mid_c_val = float(mid_c.iloc[-1])
                    bull_bias = close_c > mid_c_val   # 4h 중간밴드 위 = 강세
                    bear_bias = close_c < mid_c_val   # 4h 중간밴드 아래 = 약세

            # ── 1d EMA 필터 (옵션) ──
            if self.daily_ema_period > 0:
                df_1d = snapshot.bars.get(sym, {}).get("1d")
                if df_1d is not None and len(df_1d) >= self.daily_ema_period:
                    from indicators.trend import ema as calc_ema
                    ema_1d = calc_ema(df_1d, self.daily_ema_period)
                    close_1d = float(df_1d["close"].iloc[-1])
                    ema_1d_val = float(ema_1d.iloc[-1])
                    if bull_bias and close_1d < ema_1d_val:
                        bull_bias = False
                    if bear_bias and close_1d > ema_1d_val:
                        bear_bias = False

            upper, _, lower = bollinger_bands(df, self.bb_period, self.bb_std)
            atr_series = calc_atr(df, self.atr_period)

            prev_close = float(df["close"].iloc[-2])
            curr_close = float(df["close"].iloc[-1])
            curr_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-self.volume_lookback - 1:-1].mean())
            curr_atr = float(atr_series.iloc[-1])
            curr_upper = float(upper.iloc[-1])
            curr_lower = float(lower.iloc[-1])
            prev_upper = float(upper.iloc[-2])
            prev_lower = float(lower.iloc[-2])

            vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

            # 롱
            if bull_bias and prev_close <= prev_upper and curr_close > curr_upper and vol_ok:
                entry = curr_close
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="long",
                    entry_price=entry,
                    tp_price=entry + curr_atr * self.atr_tp_mult,
                    sl_price=entry - curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
            # 숏
            elif bear_bias and prev_close >= prev_lower and curr_close < curr_lower and vol_ok:
                entry = curr_close
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="short",
                    entry_price=entry,
                    tp_price=entry - curr_atr * self.atr_tp_mult,
                    sl_price=entry + curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))

        return signals
