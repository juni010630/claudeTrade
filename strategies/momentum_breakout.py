"""모멘텀 브레이크아웃 (15m) — Scalp 벡터화 백테스터 검증용 포팅.

Scalp 프로젝트(/Users/bagjunhyeog/Desktop/VS Code/Scalp)에서 walk-forward로
유일하게 양(+)으로 살아남은 전략을 claudeTrade 이벤트루프 엔진에 그대로 이식.
목적: 펀딩비 + 엔진 사이징/청산 같은 현실 마찰을 적용했을 때도 엣지가 살아남는지 검증.

진입 LONG  : close > rolling_high(lookback, 직전봉까지) AND volume_ratio > vol_mult
진입 SHORT : close < rolling_low(lookback, 직전봉까지)  AND volume_ratio > vol_mult
TP = entry ± atr_tp_mult × ATR(14)   (기본 3.0 — 와이드)
SL = entry ∓ atr_sl_mult × ATR(14)   (기본 0.5 — 타이트, 6:1 R:R)
max_hold = 32 봉 (= 8h, 엔진 strategy_max_hold_hours로 적용)

지표는 claudeTrade EWM 컨벤션 사용:
  - ATR(14): indicators.trend.atr (ewm span=14)
  - volume_ratio: indicators.momentum.volume_ratio (vol / SMA(vol,20))
  - rolling high/low: lookback 봉, 현재 봉 EXCLUDING (shift 후 rolling.max/min)

⚠️ 진입 집행 주의: 본 전략은 MAKER 지정가 돌파레벨 진입을 가정한다. claudeTrade 백테
   엔진은 기본적으로 OrderType.MARKET(테이커, 봉 종가)로 진입하나, config의
   execution.backtest_maker_entry: true 로 maker 지정가 경로를 활성화할 수 있다(돌파레벨
   지정가를 다음 봉에서 체결, 미충족 시 스킵 또는 taker 폴백 — engine/backtest.py
   _fill_pending_entries). maker 모드에서도 한도/상관/쿨다운은 체결 시점에 강제된다.
   Scalp 결과(테이커 시 -6bps로 음전환)와 비교 시 maker vs taker 진입비 차이가 결정적이다.
"""
from __future__ import annotations

from data.schemas import MarketSnapshot
from indicators.momentum import volume_ratio as calc_volume_ratio
from indicators.trend import atr as calc_atr
from regime.models import RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.lookback: int = cfg.get("lookback", 15)
        self.vol_mult: float = cfg.get("vol_mult", 2.5)
        self.vol_ma_period: int = cfg.get("vol_ma_period", 20)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 3.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 0.5)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str = cfg.get("signal_tf", "15m")
        self._name: str = cfg.get("strategy_name", "momentum_breakout")

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        # 국면 게이트 없음 — 전체 국면 허용 (regime/filters.py 등록 필요).
        # Scalp 벡터화 백테는 국면 필터가 없었으므로 동일하게 무게이트로 검증.
        signals: list[Signal] = []
        min_bars = max(self.lookback, self.atr_period, self.vol_ma_period) + 2

        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.signal_tf)
            if df is None or len(df) < min_bars:
                continue

            # rolling high/low: lookback 봉, 현재 봉 EXCLUDING (직전 lookback 봉의 극값)
            # shift(1) 후 rolling(lookback) → 현재 봉 직전까지의 최고/최저
            prior_high = df["high"].shift(1).rolling(self.lookback).max().iloc[-1]
            prior_low = df["low"].shift(1).rolling(self.lookback).min().iloc[-1]
            if prior_high != prior_high or prior_low != prior_low:  # NaN
                continue

            vr = calc_volume_ratio(df, self.vol_ma_period).iloc[-1]
            if vr != vr:  # NaN
                continue

            atr_s = calc_atr(df, self.atr_period)
            curr_atr = float(atr_s.iloc[-1])
            if curr_atr <= 0:
                continue

            close = float(df["close"].iloc[-1])

            long_break = close > float(prior_high) and float(vr) > self.vol_mult
            short_break = close < float(prior_low) and float(vr) > self.vol_mult

            if long_break:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="long",
                    entry_price=close,
                    tp_price=close + curr_atr * self.atr_tp_mult,
                    sl_price=close - curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
            elif short_break:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="short",
                    entry_price=close,
                    tp_price=close - curr_atr * self.atr_tp_mult,
                    sl_price=close + curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))

        return signals
