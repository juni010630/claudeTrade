"""전략 4: 멀티 타임프레임 모멘텀 브레이크아웃 (4h 방향 + 1h 진입).

4h BB 방향 확인:
  - 4h 종가 > 4h BB 상단  → 강세 국면
  - 4h 종가 < 4h BB 하단  → 약세 국면

1h 진입 조건 (방향 일치):
  - 롱: 1h 이전 종가 ≤ 1h BB 상단, 현재 종가 > 상단 (돌파) + 거래량 확인
  - 숏: 1h 이전 종가 ≥ 1h BB 하단, 현재 종가 < 하단 (돌파) + 거래량 확인

추가 필터:
  - 1d EMA200 방향 (롱: 종가 > EMA200, 숏: 종가 < EMA200)
  - 4h RSI 모멘텀 (롱: RSI > 50, 숏: RSI < 50)

TP/SL: ATR 기반
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr, ema as calc_ema
from indicators.volatility import bollinger_bands
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MultiTFBreakoutStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std_4h: float = cfg.get("bb_std_4h", 2.0)
        self.bb_std_1h: float = cfg.get("bb_std_1h", 2.0)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 50.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 50.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.5)
        self.volume_lookback: int = cfg.get("volume_lookback", 20)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 3.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.daily_ema_period: int = cfg.get("daily_ema_period", 200)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str = cfg.get("signal_tf", "1h")
        self.confirm_tf: str = cfg.get("confirm_tf", "4h")
        self.filter_tf: str = cfg.get("filter_tf", "1d")

    @property
    def name(self) -> str:
        return "multi_tf_breakout"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        # 추세장 + 브레이크아웃 국면에서만
        if regime.regime not in (MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT):
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df_1h = snapshot.bars.get(sym, {}).get(self.signal_tf)
            df_4h = snapshot.bars.get(sym, {}).get(self.confirm_tf)
            df_1d = snapshot.bars.get(sym, {}).get(self.filter_tf)

            min_bars = max(self.bb_period, self.atr_period, self.rsi_period) + 5
            if df_1h is None or len(df_1h) < min_bars:
                continue
            if df_4h is None or len(df_4h) < min_bars:
                continue

            try:
                # ── 4h 방향 확인 ──
                upper_4h, mid_4h, lower_4h = bollinger_bands(df_4h, self.bb_period, self.bb_std_4h)
                rsi_4h = calc_rsi(df_4h, self.rsi_period)
                close_4h = float(df_4h["close"].iloc[-1])
                rsi_4h_val = float(rsi_4h.iloc[-1])
                bull_4h = close_4h > float(upper_4h.iloc[-1]) and rsi_4h_val >= self.rsi_long_min
                bear_4h = close_4h < float(lower_4h.iloc[-1]) and rsi_4h_val <= self.rsi_short_max

                if not bull_4h and not bear_4h:
                    continue

                # ── 1d EMA 필터 ──
                if df_1d is not None and len(df_1d) >= self.daily_ema_period:
                    ema_1d = calc_ema(df_1d, self.daily_ema_period)
                    close_1d = float(df_1d["close"].iloc[-1])
                    ema_1d_val = float(ema_1d.iloc[-1])
                    if bull_4h and close_1d < ema_1d_val:
                        continue  # 롱인데 1d 하락 추세
                    if bear_4h and close_1d > ema_1d_val:
                        continue  # 숏인데 1d 상승 추세

                # ── 1h 진입 신호 ──
                upper_1h, _, lower_1h = bollinger_bands(df_1h, self.bb_period, self.bb_std_1h)
                atr_1h = calc_atr(df_1h, self.atr_period)
                curr_close = float(df_1h["close"].iloc[-1])
                prev_close = float(df_1h["close"].iloc[-2])
                curr_upper = float(upper_1h.iloc[-1])
                curr_lower = float(lower_1h.iloc[-1])
                prev_upper = float(upper_1h.iloc[-2])
                prev_lower = float(lower_1h.iloc[-2])
                curr_atr = float(atr_1h.iloc[-1])

                if curr_atr <= 0:
                    continue

                curr_vol = float(df_1h["volume"].iloc[-1])
                avg_vol = float(df_1h["volume"].iloc[-self.volume_lookback - 1:-1].mean())
                vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

                if bull_4h and prev_close <= prev_upper and curr_close > curr_upper and vol_ok:
                    entry = curr_close
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="long",
                        entry_price=entry,
                        tp_price=entry + curr_atr * self.atr_tp_mult,
                        sl_price=entry - curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    ))
                elif bear_4h and prev_close >= prev_lower and curr_close < curr_lower and vol_ok:
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
