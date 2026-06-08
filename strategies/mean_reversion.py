"""전략: 평균회귀 (RSI 극단 ± BB 밴드 터치 → 중심 회귀 베팅).

추세 약한 알트(LTC/UNI/STORJ 등)에서 RSI 극단 양방향 평균회귀가 4년 강건
(2026-06-08 발굴, [[project_altseung_meanrev]]). require_bb=False면 RSI 단독.
signal_tf="1d"면 1일 1회(UTC 0시)만 신호 — 1d 백테와 정합, 같은날 중복진입 방지.
순수 고정 TP/SL — 동적 청산 없음 → 1h=5m 패리티 유지.
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
        self.require_bb: bool = cfg.get("require_bb", True)  # False면 RSI 극단 단독
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
        # 시장 추세 게이트: 슬리브 심볼 바스켓 Kaufman ER(n)이 임계 초과면(=강추세)
        # 신규진입 차단 + 보유 포지션 플랫. 평균회귀는 강추세 구간서 SL 군집 → 그 구간 회피.
        # 비활성(기본)이면 always-on (기존 동작 보존). look-ahead 없음: iloc[-1]=직전 완성봉.
        self.trend_gate_enabled: bool = cfg.get("trend_gate_enabled", False)
        self.trend_gate_er_period: int = cfg.get("trend_gate_er_period", 20)
        self.trend_gate_er_threshold: float = cfg.get("trend_gate_er_threshold", 0.22)
        # 플랫 여부: True면 강추세 시 보유분도 청산, False면 진입만 차단(보유분 TP/SL 유지)
        self.trend_gate_flatten: bool = cfg.get("trend_gate_flatten", False)

    @property
    def name(self) -> str:
        return "mean_reversion"

    def _basket_trending(self, snapshot: MarketSnapshot) -> bool:
        """슬리브 심볼 바스켓의 Kaufman ER(n) 평균 > 임계 → 강추세(게이트 OFF)."""
        ers: list[float] = []
        n = self.trend_gate_er_period
        for sym in self.symbols:
            df = snapshot.bars.get(sym, {}).get(self.signal_tf)
            if df is None or len(df) < n + 2:
                continue
            c = df["close"]
            change = abs(float(c.iloc[-1]) - float(c.iloc[-1 - n]))
            vol = float(c.diff().abs().iloc[-n:].sum())
            if vol > 0:
                ers.append(change / vol)
        if not ers:
            return False
        return (sum(ers) / len(ers)) > self.trend_gate_er_threshold

    def check_early_exit(self, pos, snapshot: MarketSnapshot) -> bool:
        """추세 게이트 ON & 바스켓 강추세 → 보유 포지션 즉시 플랫(추세구간 회피)."""
        if not (self.trend_gate_enabled and self.trend_gate_flatten):
            return False
        if self.signal_tf == "1d" and snapshot.timestamp.hour != 0:
            return False
        return self._basket_trending(snapshot)

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        # signal_tf="1d": 1일 1회만(UTC 0시) — 엔진 1h 순회 중 같은 1d봉 중복진입 방지,
        # 1d 백테("전일 RSI → 당일 진입")와 정합. iloc[-1]은 DataLoader가 미래 마스크한
        # 직전 완성 1d봉이므로 0시 진입은 1d 백테의 당일 open에 해당.
        if self.signal_tf == "1d" and snapshot.timestamp.hour != 0:
            return []

        if self.use_regime_gate and regime.regime != MarketRegime.RANGING:
            return []

        # 시장 추세 게이트: 바스켓 강추세면 신규진입 차단 (보유분은 check_early_exit가 플랫)
        if self.trend_gate_enabled and self._basket_trending(snapshot):
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

            rsi_val = float(calc_rsi(df, self.rsi_period).iloc[-1])
            curr_atr = float(calc_atr(df, self.atr_period).iloc[-1])
            if curr_atr <= 0:
                continue
            close = float(df["close"].iloc[-1])

            long_sig = rsi_val <= self.rsi_oversold
            short_sig = rsi_val >= self.rsi_overbought
            if self.require_bb:  # BB 밴드 터치 동시 요구 (기본)
                upper, _, lower = bollinger_bands(df, self.bb_period, self.bb_std)
                long_sig = long_sig and close < float(lower.iloc[-1])
                short_sig = short_sig and close > float(upper.iloc[-1])

            if long_sig:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="long",
                    entry_price=close,
                    tp_price=close + curr_atr * self.atr_tp_mult,
                    sl_price=close - curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
            elif short_sig:
                signals.append(Signal(
                    symbol=sym, strategy=self.name, direction="short",
                    entry_price=close,
                    tp_price=close - curr_atr * self.atr_tp_mult,
                    sl_price=close + curr_atr * self.atr_sl_mult,
                    timestamp=snapshot.timestamp,
                ))
        return signals
