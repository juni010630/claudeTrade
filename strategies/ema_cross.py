"""전략 2: 다중 타임프레임 EMA 크로스 + MACD 히스토그램."""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.trend import atr as calc_atr, ema as calc_ema, macd as calc_macd
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    """
    필터: filter_tf EMA(daily_ema_period) 위면 롱만, 아래면 숏만 허용
    진입: signal_tf EMA(fast) / EMA(slow) 크로스 + MACD 히스토그램 0선 전환 동시 충족
    TP/SL: ATR(atr_period) × 배수
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.fast_period: int = cfg.get("fast_period", 9)
        self.slow_period: int = cfg.get("slow_period", 21)
        self.daily_ema_period: int = cfg.get("daily_ema_period", 200)
        self.macd_fast: int = cfg.get("macd_fast", 12)
        self.macd_slow: int = cfg.get("macd_slow", 26)
        self.macd_signal: int = cfg.get("macd_signal", 9)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 2.5)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str = cfg.get("signal_tf", "1h")
        self.filter_tf: str = cfg.get("filter_tf", "1d")
        # 거래량 상태 필터: 진입봉 volume/SMA(N) < 임계면 진입 차단 (저유동=노이즈, B1 검증).
        # None(기본)=비활성 → 기존 동작 보존. look-ahead 없음(진입봉까지 데이터).
        self.min_volume_ratio: float | None = cfg.get("min_volume_ratio", None)
        self.volume_ma_period: int = cfg.get("volume_ma_period", 20)
        # 다중 속도 변형용 — 이름을 config로 오버라이드 (기본은 기존과 동일)
        self._name: str = cfg.get("strategy_name", "ema_cross")

    @property
    def name(self) -> str:
        return self._name

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        if regime.regime not in (MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT):
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df_signal = snapshot.bars.get(sym, {}).get(self.signal_tf)
            df_filter = snapshot.bars.get(sym, {}).get(self.filter_tf)

            if df_signal is None or df_filter is None:
                continue
            if len(df_signal) < self.macd_slow + self.macd_signal + 5:
                continue
            if len(df_filter) < self.daily_ema_period + 1:
                continue

            # 거래량 상태 필터 (B1): 진입봉 거래량이 SMA 대비 임계 미만이면 스킵
            if self.min_volume_ratio is not None and "volume" in df_signal:
                vma = df_signal["volume"].rolling(self.volume_ma_period).mean().iloc[-1]
                if vma > 0:
                    vratio = float(df_signal["volume"].iloc[-1]) / float(vma)
                    if vratio < self.min_volume_ratio:
                        continue

            # 1D EMA 방향 필터
            daily_ema = calc_ema(df_filter, self.daily_ema_period)
            price_vs_daily_ema = df_filter["close"].iloc[-1] - float(daily_ema.iloc[-1])

            # 1H EMA 크로스
            fast = calc_ema(df_signal, self.fast_period)
            slow = calc_ema(df_signal, self.slow_period)
            prev_fast, curr_fast = float(fast.iloc[-2]), float(fast.iloc[-1])
            prev_slow, curr_slow = float(slow.iloc[-2]), float(slow.iloc[-1])

            # MACD 히스토그램
            _, _, hist = calc_macd(df_signal, self.macd_fast, self.macd_slow, self.macd_signal)
            prev_hist, curr_hist = float(hist.iloc[-2]), float(hist.iloc[-1])

            atr_s = calc_atr(df_signal, self.atr_period)
            curr_atr = float(atr_s.iloc[-1])
            if curr_atr <= 0:
                continue
            entry = float(df_signal["close"].iloc[-1])

            # 골든크로스 + MACD 히스토그램 양전환 + 일봉 EMA 위
            if (
                prev_fast <= prev_slow
                and curr_fast > curr_slow
                and prev_hist <= 0 < curr_hist
                and price_vs_daily_ema > 0
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

            # 데드크로스 + MACD 히스토그램 음전환 + 일봉 EMA 아래
            elif (
                prev_fast >= prev_slow
                and curr_fast < curr_slow
                and prev_hist >= 0 > curr_hist
                and price_vs_daily_ema < 0
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

    def check_early_exit(self, position, snapshot: MarketSnapshot) -> bool:
        if not self.config.get("early_exit_on_opp", False):
            return False
            
        sym = position.symbol
        df_signal = snapshot.bars.get(sym, {}).get(self.signal_tf)
        if df_signal is None or len(df_signal) < self.macd_slow + self.macd_signal + 5:
            return False

        fast = calc_ema(df_signal, self.fast_period)
        slow = calc_ema(df_signal, self.slow_period)
        prev_fast, curr_fast = float(fast.iloc[-2]), float(fast.iloc[-1])
        prev_slow, curr_slow = float(slow.iloc[-2]), float(slow.iloc[-1])

        _, _, hist = calc_macd(df_signal, self.macd_fast, self.macd_slow, self.macd_signal)
        prev_hist, curr_hist = float(hist.iloc[-2]), float(hist.iloc[-1])

        # 모멘텀 기준만 체크 (1d EMA 필터 무시)
        mom_long = (prev_fast <= prev_slow and curr_fast > curr_slow and prev_hist <= 0 < curr_hist)
        mom_short = (prev_fast >= prev_slow and curr_fast < curr_slow and prev_hist >= 0 > curr_hist)

        if position.direction == "long" and mom_short:
            return True
        if position.direction == "short" and mom_long:
            return True
        return False
