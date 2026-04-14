"""스캘핑 전략 1: VWAP 크로스 + RSI + 거래량 (5m 봉 기반).

롱 조건:
  - 이전 봉 종가 < VWAP, 현재 봉 종가 > VWAP (상향 크로스)
  - RSI > rsi_long_min 이고 RSI 상승 중
  - 거래량 평균 대비 1.3× 이상

숏 조건:
  - 이전 봉 종가 > VWAP, 현재 봉 종가 < VWAP (하향 크로스)
  - RSI < rsi_short_max 이고 RSI 하락 중
  - 거래량 확인

TP/SL: ATR 배수 기반 (스캘핑 타이트 설정)
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr
from indicators.vwap import vwap as calc_vwap
from regime.models import RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class ScalpVWAPStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.rsi_period: int = cfg.get("rsi_period", 9)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 45.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 55.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.3)
        self.volume_lookback: int = cfg.get("volume_lookback", 20)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 1.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 0.5)
        # 퍼센트 기반 TP/SL (> 0 이면 ATR 방식 대신 사용)
        self.tp_pct: float = cfg.get("tp_pct", 0.0)   # 예: 0.004 = 0.4%
        self.sl_pct: float = cfg.get("sl_pct", 0.0)   # 예: 0.002 = 0.2%
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "5m")

    @property
    def name(self) -> str:
        return "scalp_vwap"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        signals: list[Signal] = []

        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.timeframe)
            min_bars = max(self.atr_period, self.rsi_period, self.volume_lookback) + 5
            if df is None or len(df) < min_bars:
                continue

            try:
                vwap_s = calc_vwap(df)
                rsi_s = calc_rsi(df, self.rsi_period)
                atr_s = calc_atr(df, self.atr_period)
            except Exception:
                continue

            curr_close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2])
            curr_vwap = float(vwap_s.iloc[-1])
            prev_vwap = float(vwap_s.iloc[-2])
            curr_rsi = float(rsi_s.iloc[-1])
            prev_rsi = float(rsi_s.iloc[-2])
            curr_atr = float(atr_s.iloc[-1])

            if curr_atr <= 0 or pd.isna(curr_vwap) or pd.isna(prev_vwap):
                continue

            curr_vol = float(df["volume"].iloc[-1])
            avg_vol = float(df["volume"].iloc[-self.volume_lookback - 1:-1].mean())
            vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

            # 롱: VWAP 상향 크로스 + RSI 상승 + 거래량
            long_ok = (
                prev_close < prev_vwap
                and curr_close > curr_vwap
                and curr_rsi >= self.rsi_long_min
                and curr_rsi > prev_rsi
                and vol_ok
            )
            # 숏: VWAP 하향 크로스 + RSI 하락 + 거래량
            short_ok = (
                prev_close > prev_vwap
                and curr_close < curr_vwap
                and curr_rsi <= self.rsi_short_max
                and curr_rsi < prev_rsi
                and vol_ok
            )

            if long_ok:
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
            elif short_ok:
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
