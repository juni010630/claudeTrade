"""스캘핑 전략 2: EMA 리본 눌림목 진입 (5m 봉 기반).

EMA 5 / 8 / 13 / 21 리본:
  - 강세 정렬: EMA5 > EMA8 > EMA13 > EMA21
  - 약세 정렬: EMA5 < EMA8 < EMA13 < EMA21

롱 조건:
  - 15m 기준 EMA 리본 강세 정렬
  - 5m 가격이 EMA8 아래로 눌렸다가 회복 (pullback)
  - RSI 45~70 사이 (과매도도 아니고 과열도 아님)
  - 거래량 확인

숏 조건:
  - 15m 기준 EMA 리본 약세 정렬
  - 5m 가격이 EMA8 위로 튀었다가 재하락
  - RSI 30~55 사이
  - 거래량 확인
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr, ema as calc_ema
from regime.models import RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class ScalpEMARibbonStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.ema_periods: list[int] = cfg.get("ema_periods", [5, 8, 13, 21])
        self.rsi_period: int = cfg.get("rsi_period", 9)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 48.0)
        self.rsi_long_max: float = cfg.get("rsi_long_max", 70.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 52.0)
        self.rsi_short_min: float = cfg.get("rsi_short_min", 30.0)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 1.2)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 0.6)
        self.tp_pct: float = cfg.get("tp_pct", 0.0)
        self.sl_pct: float = cfg.get("sl_pct", 0.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.2)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "5m")
        self.trend_tf: str = cfg.get("trend_tf", "15m")

    @property
    def name(self) -> str:
        return "scalp_ema_ribbon"

    def _ribbon_bullish(self, df: pd.DataFrame) -> bool:
        """EMA 리본이 강세 정렬인지 확인."""
        emas = [calc_ema(df, p).iloc[-1] for p in sorted(self.ema_periods)]
        return all(emas[i] > emas[i + 1] for i in range(len(emas) - 1))

    def _ribbon_bearish(self, df: pd.DataFrame) -> bool:
        """EMA 리본이 약세 정렬인지 확인."""
        emas = [calc_ema(df, p).iloc[-1] for p in sorted(self.ema_periods)]
        return all(emas[i] < emas[i + 1] for i in range(len(emas) - 1))

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        signals: list[Signal] = []

        for sym in self.symbols:
            df5 = snapshot.bars.get(sym, {}).get(self.timeframe)
            df_trend = snapshot.bars.get(sym, {}).get(self.trend_tf)
            min_bars = max(self.ema_periods) + self.rsi_period + 5
            if df5 is None or len(df5) < min_bars:
                continue

            try:
                # 상위 타임프레임 추세 방향
                trend_df = df_trend if (df_trend is not None and len(df_trend) >= max(self.ema_periods) + 2) else df5
                ribbon_bull = self._ribbon_bullish(trend_df)
                ribbon_bear = self._ribbon_bearish(trend_df)

                # 5m 지표
                ema8 = calc_ema(df5, 8)
                rsi_s = calc_rsi(df5, self.rsi_period)
                atr_s = calc_atr(df5, self.atr_period)
            except Exception:
                continue

            curr_close = float(df5["close"].iloc[-1])
            prev_close = float(df5["close"].iloc[-2])
            curr_ema8 = float(ema8.iloc[-1])
            prev_ema8 = float(ema8.iloc[-2])
            curr_rsi = float(rsi_s.iloc[-1])
            curr_atr = float(atr_s.iloc[-1])

            if curr_atr <= 0:
                continue

            curr_vol = float(df5["volume"].iloc[-1])
            avg_vol = float(df5["volume"].iloc[-21:-1].mean())
            vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

            # 롱: 리본 강세 + 이전 봉이 EMA8 아래 터치 후 회복
            long_pullback = prev_close < prev_ema8 and curr_close > curr_ema8
            long_rsi = self.rsi_long_min <= curr_rsi <= self.rsi_long_max
            if ribbon_bull and long_pullback and long_rsi and vol_ok:
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

            short_pullback = prev_close > prev_ema8 and curr_close < curr_ema8
            short_rsi = self.rsi_short_min <= curr_rsi <= self.rsi_short_max
            if ribbon_bear and short_pullback and short_rsi and vol_ok:
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
