"""전략 3: 평균회귀 + 펀딩비 + OI 캐스케이드 필터 (횡보장 전용)."""
from __future__ import annotations

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr
from indicators.volatility import bollinger_bands
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    롱: 종가 < BB 하단 AND RSI < rsi_oversold AND 펀딩비 < funding_long_max
    숏: 종가 > BB 상단 AND RSI > rsi_overbought AND 펀딩비 > funding_short_min
    OI 급등 시 캐스케이드 위험 → 진입 차단
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std: float = cfg.get("bb_std", 2.0)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_oversold: float = cfg.get("rsi_oversold", 30.0)
        self.rsi_overbought: float = cfg.get("rsi_overbought", 70.0)
        self.funding_long_max: float = cfg.get("funding_long_max", 0.0001)
        self.funding_short_min: float = cfg.get("funding_short_min", 0.0001)
        self.oi_cascade_threshold: float = cfg.get("oi_cascade_threshold", 5.0)  # OI 5% 이상 급등
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 1.5)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.timeframe: str = cfg.get("timeframe", "1h")

    @property
    def name(self) -> str:
        return "mean_reversion"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if regime.regime != MarketRegime.RANGING:
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.timeframe)
            if df is None or len(df) < max(self.bb_period, self.rsi_period) + 2:
                continue

            upper, _, lower = bollinger_bands(df, self.bb_period, self.bb_std)
            rsi_s = calc_rsi(df, self.rsi_period)
            atr_s = calc_atr(df, self.atr_period)

            curr_close = float(df["close"].iloc[-1])
            curr_upper = float(upper.iloc[-1])
            curr_lower = float(lower.iloc[-1])
            curr_rsi = float(rsi_s.iloc[-1])
            curr_atr = float(atr_s.iloc[-1])
            funding = snapshot.funding_rates.get(sym, 0.0)

            # OI 캐스케이드 체크
            oi_current = snapshot.open_interest.get(sym, 0.0)
            # open_interest에 이전 값이 없으면 체크 생략
            oi_change_pct = 0.0
            if hasattr(self, "_prev_oi") and sym in self._prev_oi and self._prev_oi[sym] > 0:
                oi_change_pct = abs(oi_current - self._prev_oi[sym]) / self._prev_oi[sym] * 100
            if not hasattr(self, "_prev_oi"):
                self._prev_oi: dict[str, float] = {}
            self._prev_oi[sym] = oi_current

            if oi_change_pct >= self.oi_cascade_threshold:
                continue

            entry = curr_close

            # 롱
            if (
                curr_close < curr_lower
                and curr_rsi < self.rsi_oversold
                and funding < self.funding_long_max
            ):
                signals.append(
                    Signal(
                        symbol=sym,
                        strategy=self.name,
                        direction="long",
                        entry_price=entry,
                        tp_price=entry + curr_atr * self.atr_tp_mult,
                        sl_price=entry - curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    )
                )

            # 숏
            elif (
                curr_close > curr_upper
                and curr_rsi > self.rsi_overbought
                and funding > self.funding_short_min
            ):
                signals.append(
                    Signal(
                        symbol=sym,
                        strategy=self.name,
                        direction="short",
                        entry_price=entry,
                        tp_price=entry - curr_atr * self.atr_tp_mult,
                        sl_price=entry + curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    )
                )

        return signals
