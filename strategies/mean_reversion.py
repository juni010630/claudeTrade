"""전략: 횡보장 평균회귀 (BB 밴드 터치 + RSI 극단 → 중심 회귀 베팅).

RANGING 국면 전용 (regime/filters.py 라우팅). 순수 고정 TP/SL — 동적 청산 없음
→ 1h=5m 패리티 유지 (기존 검증 클래스와 동일).
"""
from __future__ import annotations

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import adx as calc_adx, atr as calc_atr
from indicators.volatility import bollinger_bands
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    진입: signal_tf 종가가 BB(period, std) 하단 아래 + RSI ≤ oversold → 롱
          상단 위 + RSI ≥ overbought → 숏
    TP/SL: ATR(atr_period) × 배수 (고정)
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std: float = cfg.get("bb_std", 2.0)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_oversold: float = cfg.get("rsi_oversold", 30.0)
        self.rsi_overbought: float = cfg.get("rsi_overbought", 70.0)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 1.5)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.5)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str = cfg.get("signal_tf", "1h")
        # 횡보 판정: 엔진 regime(1h ADX)은 너무 민감 → 자체 게이트 선택 가능
        # use_regime_gate=True면 엔진 RANGING만, False면 심볼별 adx_tf ADX < max_adx
        self.use_regime_gate: bool = cfg.get("use_regime_gate", True)
        self.max_adx: float = cfg.get("max_adx", 25.0)
        self.adx_tf: str = cfg.get("adx_tf", "1d")

    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if self.use_regime_gate and regime.regime != MarketRegime.RANGING:
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.signal_tf)
            if df is None or len(df) < max(self.bb_period, self.rsi_period, self.atr_period) + 2:
                continue

            # 자체 횡보 게이트: 심볼 adx_tf ADX(14)가 max_adx 미만일 때만
            # (마지막 가시 봉 기준 — ema_cross 일봉 필터와 동일 관례, look-ahead 없음)
            if not self.use_regime_gate:
                df_adx = snapshot.bars.get(sym, {}).get(self.adx_tf)
                if df_adx is None or len(df_adx) < 30:
                    continue
                if float(calc_adx(df_adx, 14).iloc[-1]) >= self.max_adx:
                    continue

            upper, _, lower = bollinger_bands(df, self.bb_period, self.bb_std)
            rsi_val = float(calc_rsi(df, self.rsi_period).iloc[-1])
            curr_atr = float(calc_atr(df, self.atr_period).iloc[-1])
            if curr_atr <= 0:
                continue
            close = float(df["close"].iloc[-1])

            if close < float(lower.iloc[-1]) and rsi_val <= self.rsi_oversold:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="long",
                    entry_price=close,
                    tp_price=close + curr_atr * self.atr_tp_mult,
                    sl_price=close - curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
            elif close > float(upper.iloc[-1]) and rsi_val >= self.rsi_overbought:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="short",
                    entry_price=close,
                    tp_price=close - curr_atr * self.atr_tp_mult,
                    sl_price=close + curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
        return signals
