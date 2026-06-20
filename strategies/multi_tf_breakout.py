"""전략 4: 멀티 타임프레임 모멘텀 브레이크아웃 (4h 방향 + 1h 진입).

4h BB 방향 확인:
  - 4h 종가 > 4h BB 상단  → 강세 국면
  - 4h 종가 < 4h BB 하단  → 약세 국면

1h 진입 조건 (방향 일치):
  - 롱: 1h 이전 종가 ≤ 1h BB 상단, 현재 종가 > 상단 (돌파) + 거래량 확인
  - 숏: 1h 이전 종가 ≥ 1h BB 하단, 현재 종가 < 하단 (돌파) + 거래량 확인

추가 필터:
  - 1d EMA200 방향 (롱: 종가 > EMA200, 숏: 종가 < EMA200)
  - 4h RSI 모멘텀 (롱: RSI > 50, 숏: RSI < 50)

TP/SL: ATR 기반
"""
from __future__ import annotations

import pandas as pd

from data.schemas import MarketSnapshot
from indicators.momentum import rsi as calc_rsi
from indicators.trend import atr as calc_atr, ema as calc_ema
from indicators.volatility import bollinger_bands
from regime.models import MarketRegime, RegimeState
from signals.models import Signal
from strategies.base import BaseStrategy


class MultiTFBreakoutStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        super().__init__(cfg)
        self.bb_period: int = cfg.get("bb_period", 20)
        self.bb_std_4h: float = cfg.get("bb_std_4h", 2.0)
        self.bb_std_1h: float = cfg.get("bb_std_1h", 2.0)
        self.rsi_period: int = cfg.get("rsi_period", 14)
        self.rsi_long_min: float = cfg.get("rsi_long_min", 50.0)
        self.rsi_short_max: float = cfg.get("rsi_short_max", 50.0)
        self.volume_multiplier: float = cfg.get("volume_multiplier", 1.5)
        self.volume_lookback: int = cfg.get("volume_lookback", 20)
        self.atr_period: int = cfg.get("atr_period", 14)
        self.atr_tp_mult: float = cfg.get("atr_tp_mult", 3.0)
        self.atr_sl_mult: float = cfg.get("atr_sl_mult", 1.0)
        self.daily_ema_period: int = cfg.get("daily_ema_period", 200)
        self.symbols: list[str] = cfg.get("symbols", ["BTCUSDT"])
        self.signal_tf: str = cfg.get("signal_tf", "1h")
        self.confirm_tf: str = cfg.get("confirm_tf", "4h")
        self.filter_tf: str = cfg.get("filter_tf", "1d")
        # 펀딩 레짐 게이트(기본 off): 펀딩-z > funding_z_max 인 돌파(양방향) 스킵.
        # 돌파는 펀딩 낮은 레짐(고변동/추세)서 작동, 높은 레짐(안일)서 실패 — 진입 필터.
        # funding-z는 scripts/precompute_funding_z.py 산출물. 캐시 없으면 no-op(라이브 안전).
        self.funding_gate: bool = cfg.get("funding_gate", False)
        self.funding_z_max: float = cfg.get("funding_z_max", 0.5)
        # 검증용 대조군: funding 대신 같은 비율을 무작위 스킵(결정론적 해시). "그냥 덜거래" 효과 분리.
        self.funding_gate_random: bool = cfg.get("funding_gate_random", False)
        self.funding_gate_random_pct: int = cfg.get("funding_gate_random_pct", 25)
        self._fz: dict[str, "pd.Series"] = {}
        if self.funding_gate:
            from pathlib import Path
            fzp = Path(__file__).resolve().parents[1] / "data" / "cache" / "funding_z.parquet"
            if fzp.exists():
                az = pd.read_parquet(fzp)
                az["timestamp"] = pd.to_datetime(az["timestamp"], utc=True)
                for sym, g in az[az["symbol"].isin(self.symbols)].groupby("symbol"):
                    self._fz[sym] = g.set_index("timestamp")["z"].sort_index()

    def _fz_asof(self, sym: str, ts) -> float | None:
        s = self._fz.get(sym)
        if s is None or len(s) == 0:
            return None
        v = s.asof(ts)
        return None if pd.isna(v) else float(v)

    @property
    def name(self) -> str:
        return "multi_tf_breakout"

    def generate_signals(
        self, snapshot: MarketSnapshot, regime: RegimeState
    ) -> list[Signal]:
        # 추세장 + 브레이크아웃 국면에서만
        if regime.regime not in (MarketRegime.TRENDING, MarketRegime.PRE_BREAKOUT):
            return []

        signals: list[Signal] = []
        for sym in self.symbols:
            df_1h = snapshot.bars.get(sym, {}).get(self.signal_tf)
            df_4h = snapshot.bars.get(sym, {}).get(self.confirm_tf)
            df_1d = snapshot.bars.get(sym, {}).get(self.filter_tf)

            min_bars = max(self.bb_period, self.atr_period, self.rsi_period) + 5
            if df_1h is None or len(df_1h) < min_bars:
                continue
            if df_4h is None or len(df_4h) < min_bars:
                continue

            try:
                # ── 4h 방향 확인 ──
                upper_4h, mid_4h, lower_4h = bollinger_bands(df_4h, self.bb_period, self.bb_std_4h)
                rsi_4h = calc_rsi(df_4h, self.rsi_period)
                close_4h = float(df_4h["close"].iloc[-1])
                rsi_4h_val = float(rsi_4h.iloc[-1])
                bull_4h = close_4h > float(upper_4h.iloc[-1]) and rsi_4h_val >= self.rsi_long_min
                bear_4h = close_4h < float(lower_4h.iloc[-1]) and rsi_4h_val <= self.rsi_short_max

                if not bull_4h and not bear_4h:
                    continue

                # ── 1d EMA 필터 ──
                if df_1d is not None and len(df_1d) >= self.daily_ema_period:
                    ema_1d = calc_ema(df_1d, self.daily_ema_period)
                    close_1d = float(df_1d["close"].iloc[-1])
                    ema_1d_val = float(ema_1d.iloc[-1])
                    if bull_4h and close_1d < ema_1d_val:
                        continue  # 롱인데 1d 하락 추세
                    if bear_4h and close_1d > ema_1d_val:
                        continue  # 숏인데 1d 상승 추세

                # ── 1h 진입 신호 ──
                upper_1h, _, lower_1h = bollinger_bands(df_1h, self.bb_period, self.bb_std_1h)
                atr_1h = calc_atr(df_1h, self.atr_period)
                curr_close = float(df_1h["close"].iloc[-1])
                prev_close = float(df_1h["close"].iloc[-2])
                curr_upper = float(upper_1h.iloc[-1])
                curr_lower = float(lower_1h.iloc[-1])
                prev_upper = float(upper_1h.iloc[-2])
                prev_lower = float(lower_1h.iloc[-2])
                curr_atr = float(atr_1h.iloc[-1])

                if curr_atr <= 0:
                    continue

                curr_vol = float(df_1h["volume"].iloc[-1])
                avg_vol = float(df_1h["volume"].iloc[-self.volume_lookback - 1:-1].mean())
                vol_ok = avg_vol > 0 and curr_vol / avg_vol >= self.volume_multiplier

                # 펀딩 레짐 게이트: 펀딩-z 높은(안일) 레짐의 돌파 스킵 (양방향, 룩어헤드 없음=asof)
                if self.funding_gate:
                    if self.funding_gate_random:   # 대조군: 같은 비율 무작위 스킵(결정론)
                        import hashlib
                        h = int(hashlib.md5(f"{sym}|{snapshot.timestamp.isoformat()}".encode()).hexdigest(), 16) % 100
                        if h < self.funding_gate_random_pct:
                            continue
                    else:
                        fz = self._fz_asof(sym, snapshot.timestamp)
                        if fz is not None and fz > self.funding_z_max:
                            continue

                if bull_4h and prev_close <= prev_upper and curr_close > curr_upper and vol_ok:
                    entry = curr_close
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="long",
                        entry_price=entry,
                        tp_price=entry + curr_atr * self.atr_tp_mult,
                        sl_price=entry - curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    ))
                elif bear_4h and prev_close >= prev_lower and curr_close < curr_lower and vol_ok:
                    entry = curr_close
                    signals.append(Signal(
                        symbol=sym, strategy=self.name, direction="short",
                        entry_price=entry,
                        tp_price=entry - curr_atr * self.atr_tp_mult,
                        sl_price=entry + curr_atr * self.atr_sl_mult,
                        timestamp=snapshot.timestamp,
                    ))

            except Exception as e:
                import logging
                logging.getLogger(__name__).debug("multi_tf %s 시그널 생성 실패: %s", sym, e)
                continue

        return signals

    def check_early_exit(self, position, snapshot: MarketSnapshot) -> bool:
        if not self.config.get("early_exit_on_opp", False):
            return False
            
        sym = position.symbol
        df_1h = snapshot.bars.get(sym, {}).get(self.signal_tf)
        df_4h = snapshot.bars.get(sym, {}).get(self.confirm_tf)
        
        if df_1h is None or len(df_1h) < self.bb_period + 5:
            return False
        if df_4h is None or len(df_4h) < self.bb_period + 5:
            return False

        try:
            upper_4h, mid_4h, lower_4h = bollinger_bands(df_4h, self.bb_period, self.bb_std_4h)
            rsi_4h = calc_rsi(df_4h, self.rsi_period)
            close_4h = float(df_4h["close"].iloc[-1])
            rsi_4h_val = float(rsi_4h.iloc[-1])
            
            bull_4h = close_4h > float(upper_4h.iloc[-1]) and rsi_4h_val >= self.rsi_long_min
            bear_4h = close_4h < float(lower_4h.iloc[-1]) and rsi_4h_val <= self.rsi_short_max

            upper_1h, _, lower_1h = bollinger_bands(df_1h, self.bb_period, self.bb_std_1h)
            curr_close = float(df_1h["close"].iloc[-1])
            prev_close = float(df_1h["close"].iloc[-2])
            curr_upper = float(upper_1h.iloc[-1])
            curr_lower = float(lower_1h.iloc[-1])
            prev_upper = float(upper_1h.iloc[-2])
            prev_lower = float(lower_1h.iloc[-2])

            # 조기 청산 시 거래량 필터 제외, 가격+모멘텀 반전만 확인
            if position.direction == "long" and bear_4h and prev_close >= prev_lower and curr_close < curr_lower:
                return True
            if position.direction == "short" and bull_4h and prev_close <= prev_upper and curr_close > curr_upper:
                return True
        except Exception:
            pass
            
        return False
