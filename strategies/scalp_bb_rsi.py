"""스캘핑 전략 3: 볼린저밴드 스퀴즈 → 브레이크아웃 + RSI (5m 봉 기반).

스퀴즈 감지:
  - BB 폭(상단-하단)이 최근 N봉 하위 squeeze_pct 퍼센타일 이하

브레이크아웃:
  - 종가가 BB 상단 돌파 (롱) / 하단 하락 (숏)
  - RSI 모멘텀 확인
  - 거래량 급등 확인

TP/SL: ATR 배수 기반
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr
from indicators.volatility import bollinger_bands
from regime.models import RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class ScalpBBRSIStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std: float = cfg.get("bb_std", 2.0)
        self.squeeze_pct: float = cfg.get("squeeze_pct", 0.25)
        self.squeeze_lookback: int = cfg.get("squeeze_lookback", 50)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 52.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 48.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.5)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 1.3)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 0.5)
        self.tp_pct: float = cfg.get("tp_pct", 0.0)
        self.sl_pct: float = cfg.get("sl_pct", 0.0)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "5m")

    @property
    def name(self) -> str:
        return "scalp_bb_rsi"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        signals: list[Signal] = []

        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.timeframe)
            min_bars = max(self.bb_period, self.rsi_period, self.squeeze_lookback) + 5
            if df is None or len(df) < min_bars:
                continue

            try:
                upper, mid, lower = bollinger_bands(df, self.bb_period, self.bb_std)
                rsi_s = calc_rsi(df, self.rsi_period)
                atr_s = calc_atr(df, self.atr_period)
            except Exception:
                continue

            # BB 폭 계산
            bb_width = upper - lower
            # 최근 squeeze_lookback 봉 중 현재 폭이 하위 squeeze_pct 이하면 스퀴즈
            width_history = bb_width.iloc[-self.squeeze_lookback:]
            threshold = width_history.quantile(self.squeeze_pct)
            curr_width = float(bb_width.iloc[-2])  # 직전 봉 기준 스퀴즈
            was_squeeze = curr_width <= threshold

            if not was_squeeze:
                continue

            curr_close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2])
            curr_upper = float(upper.iloc[-1])
            curr_lower = float(lower.iloc[-1])
            prev_upper = float(upper.iloc[-2])
            prev_lower = float(lower.iloc[-2])
            curr_rsi = float(rsi_s.iloc[-1])
            curr_atr = float(atr_s.iloc[-1])

            if curr_atr <= 0:
                continue

            curr_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-21:-1].mean())
            vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

            # 롱: 스퀴즈 직후 BB 상단 상향 돌파
            long_break = prev_close <= prev_upper and curr_close > curr_upper
            if long_break and curr_rsi >= self.rsi_long_min and vol_ok:
                entry = curr_close
                if self.tp_pct > 0:
                    tp = entry * (1 + self.tp_pct)
                    sl = entry * (1 - self.sl_pct)
                else:
                    tp = entry + curr_atr * self.atr_tp_mult
                    sl = entry - curr_atr * self.atr_sl_mult
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="long",
                    entry_price=entry, tp_price=tp, sl_price=sl,
                    timestamp=snapshot.timestamp,
                ))
                continue

            short_break = prev_close >= prev_lower and curr_close < curr_lower
            if short_break and curr_rsi <= self.rsi_short_max and vol_ok:
                entry = curr_close
                if self.tp_pct > 0:
                    tp = entry * (1 - self.tp_pct)
                    sl = entry * (1 + self.sl_pct)
                else:
                    tp = entry - curr_atr * self.atr_tp_mult
                    sl = entry + curr_atr * self.atr_sl_mult
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="short",
                    entry_price=entry, tp_price=tp, sl_price=sl,
                    timestamp=snapshot.timestamp,
                ))

        return signals
