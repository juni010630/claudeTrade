"""전략 5: Supertrend 추세 추종 (1h + 4h 이중 확인).

Supertrend 기본 원리:
  - ST 선이 아래 → 가격 위 : 상승 추세 (direction = 1)
  - ST 선이 위  → 가격 아래: 하락 추세 (direction = -1)

진입:
  롱: 1h Supertrend가 -1→1 전환 (bullish flip)
      + 4h Supertrend도 1 (상위 TF 방향 일치)
      + RSI > 50 (모멘텀 확인)
  숏: 1h Supertrend가 1→-1 전환 (bearish flip)
      + 4h Supertrend도 -1
      + RSI < 50

TP: 다음 Supertrend 반전까지 → ATR 배수로 근사
SL: Supertrend 선 (동적) → 진입 시 계산
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.supertrend import supertrend
from indicators.trend import atr as calc_atr
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class SupertrendStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.st_period: int = cfg.get("st_period", 10)
        self.st_multiplier: float = cfg.get("st_multiplier", 3.0)
        self.st_period_4h: int = cfg.get("st_period_4h", 10)
        self.st_multiplier_4h: float = cfg.get("st_multiplier_4h", 3.0)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 50.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 50.0)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 3.5)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "1h")
        self.confirm_tf: str = cfg.get("confirm_tf", "4h")

    @property
    def name(self) -> str:
        return "supertrend"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        # 추세장에서만
        if regime.regime != MarketRegime.TRENDING:
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df_1h = snapshot.bars.get(sym, {}).get(self.timeframe)
            df_4h = snapshot.bars.get(sym, {}).get(self.confirm_tf)
            min_bars = max(self.st_period, self.atr_period, self.rsi_period) * 3 + 5
            if df_1h is None or len(df_1h) < min_bars:
                continue

            try:
                # ── 1h Supertrend ──
                st_line, direction = supertrend(df_1h, self.st_period, self.st_multiplier)
                curr_dir = int(direction.iloc[-1])
                prev_dir = int(direction.iloc[-2])

                # flip 이 없으면 신호 없음
                if curr_dir == prev_dir:
                    continue

                # ── 4h 방향 확인 ──
                if df_4h is not None and len(df_4h) >= min_bars:
                    _, dir_4h = supertrend(df_4h, self.st_period_4h, self.st_multiplier_4h)
                    if int(dir_4h.iloc[-1]) != curr_dir:
                        continue  # 상위 TF 방향 불일치

                # ── RSI 확인 ──
                rsi_s = calc_rsi(df_1h, self.rsi_period)
                curr_rsi = float(rsi_s.iloc[-1])

                atr_s = calc_atr(df_1h, self.atr_period)
                curr_atr = float(atr_s.iloc[-1])
                curr_close = float(df_1h["close"].iloc[-1])
                curr_st = float(st_line.iloc[-1])

                if curr_atr <= 0:
                    continue

                # 롱 진입: -1→1 전환
                if curr_dir == 1 and curr_rsi >= self.rsi_long_min:
                    entry = curr_close
                    sl = curr_st  # Supertrend 선이 초기 SL
                    sl_dist = max(entry - sl, curr_atr * self.atr_sl_mult)
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="long",
                        entry_price=entry,
                        tp_price=entry + curr_atr * self.atr_tp_mult,
                        sl_price=entry - sl_dist,
                        timestamp=snapshot.timestamp,
                    ))

                # 숏 진입: 1→-1 전환
                elif curr_dir == -1 and curr_rsi <= self.rsi_short_max:
                    entry = curr_close
                    sl = curr_st
                    sl_dist = max(sl - entry, curr_atr * self.atr_sl_mult)
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="short",
                        entry_price=entry,
                        tp_price=entry - curr_atr * self.atr_tp_mult,
                        sl_price=entry + sl_dist,
                        timestamp=snapshot.timestamp,
                    ))

            except Exception:
                continue

        return signals
