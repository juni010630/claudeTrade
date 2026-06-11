"""전략: 1d 슬로우 EMA 크로스 (광역 알트 추세/숏수확) — greedy search 챔피언.

검증 = NEWEDGE_GREEDY_RESULTS.md (~700변형/7반복 OOS 생존 유일 패밀리).
1d EMA(fast)×EMA(slow) 크로스 플립 진입, 고정 ATR TP/SL, 30d 중위 달러볼륨 필터.
hour==0 1일 1회 신호 — mean_reversion과 동일 관례. iloc[-1] = DataLoader가
미래 마스크한 직전 완성 1d봉 → 0시 진입 = 1d 백테의 당일 시가 진입과 정합 (look-ahead 없음).
순수 고정 TP/SL — 동적 청산 없음.
"""
from __future__ import annotations

from data.schemas import MarketSnapshot
from indicators.trend import atr as calc_atr
from regime.models import RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class EmaSlowDailyStrategy(BaseStrategy):
    """
    진입: 완성 1d봉 기준 EMA(fast)-EMA(slow) 차이 부호가 iloc[-2]→iloc[-1]에서 전환
          (상향 전환=롱, 하향 전환=숏)
    TP/SL: ATR(atr_period) × 배수 (고정), 기준가 = 마지막 완성봉 종가
    유동성: 최근 30개 1d봉 중위 달러볼륨 > liq_min_usd 일 때만
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.fast_period: int = cfg.get("fast_period", 20)
        self.slow_period: int = cfg.get("slow_period", 100)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 6.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 3.0)
        self.signal_tf: str = cfg.get("signal_tf", "1d")
        self.symbols: list[str] = cfg.get("symbols", [])
        self.liq_min_usd: float = cfg.get("liq_min_usd", 0.0)
        self.liq_window: int = cfg.get("liq_window", 30)

    @property
    def name(self) -> str:
        return "macross_d"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if snapshot.timestamp.hour != 0:
            return []

        signals: list[Signal] = []
        need = self.slow_period + 2
        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.signal_tf)
            if df is None or len(df) < need:
                continue

            c = df["close"]
            if self.liq_min_usd > 0:
                dvol = float((c * df["volume"]).tail(self.liq_window).median())
                if not dvol > self.liq_min_usd:
                    continue

            ema_f = c.ewm(span=self.fast_period, adjust=False).mean()
            ema_s = c.ewm(span=self.slow_period, adjust=False).mean()
            d_now = float(ema_f.iloc[-1]) - float(ema_s.iloc[-1])
            d_prev = float(ema_f.iloc[-2]) - float(ema_s.iloc[-2])
            if d_now > 0 and d_prev <= 0:
                direction = "long"
            elif d_now < 0 and d_prev >= 0:
                direction = "short"
            else:
                continue

            curr_atr = float(calc_atr(df, self.atr_period).iloc[-1])
            if curr_atr <= 0:
                continue
            close = float(c.iloc[-1])
            sign = 1.0 if direction == "long" else -1.0
            signals.append(Signal(
                symbol=sym, strategy=self.name, direction=direction,
                entry_price=close,
                tp_price=close + sign * curr_atr * self.atr_tp_mult,
                sl_price=close - sign * curr_atr * self.atr_sl_mult,
                timestamp=snapshot.timestamp,
            ))
        return signals
