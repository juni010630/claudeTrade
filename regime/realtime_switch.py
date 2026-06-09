"""실시간 장 국면 스위치 — 지금 이 순간이 추세장인지 횡보장인지 판별.

백테스트로 임계값을 튜닝하지 않는다. 각 지표는 업계 표준(교과서·피보나치) 임계값을
그대로 쓰며, 서로 보완적인 5개 지표의 투표로 현재 국면을 읽는다. 한 지표가 흔들려도
다수결로 견고하게 판단한다.

지표 (모두 확정된 과거 봉만 사용 — look-ahead 없음):
  - ADX (추세 강도)             : >25 추세 / <20 횡보 / 중간 중립           (Wilder 고전)
  - Choppiness Index (CI)        : <38.2 추세 / >61.8 횡보                   (피보나치, 횡보 전용)
  - Kaufman Efficiency Ratio(ER): >0.35 추세 / <0.20 횡보                   (방향 효율 0~1)
  - 선형회귀 R²  (보조)          : >0.60 추세 직선성 / <0.20 무방향
  - BB폭 백분위  (보조)          : 하위 20% 수축=횡보 / 상위 20% 확장=추세

스위치는 강한 신호(|score|>=1.5)에서만 상태를 바꾸고 애매한 구간에선 직전 상태를
유지하는 히스테리시스를 둬 잦은 깜빡임(whipsaw)을 막는다 — 진짜 '스위치'처럼 동작.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from indicators.volatility import bb_width as calc_bb_width


# ── 표준 임계값 (튜닝하지 않음) ──────────────────────────────────────
ADX_TREND, ADX_RANGE = 25.0, 20.0
CI_TREND, CI_RANGE = 38.2, 61.8          # 피보나치 38.2 / 61.8
ER_TREND, ER_RANGE = 0.35, 0.20
R2_TREND, R2_RANGE = 0.60, 0.20
BBW_SQUEEZE_PCT, BBW_EXPAND_PCT = 0.20, 0.80

# 스위치 전환/유지 경계 (히스테리시스)
SWITCH_STRONG = 1.5   # |score| 이 값 이상이면 상태 전환


# ── 국면 전용 지표 (regime 모듈 내부 — 공유 indicators 파일 불변) ──────
def wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """표준 Wilder ADX (alpha=1/period RMA). 정설 임계값(>25 추세/<20 횡보)이 그대로 유효.

    엔진의 indicators.trend.adx는 ewm(span=period)라 값이 ~1.87배 부풀려져 25 임계에
    거의 항상 걸린다. 국면 판별엔 표준 정의가 필요하므로 여기서 별도 계산한다
    (엔진 지표는 패리티 위해 불변).
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_high, prev_low, prev_close = high.shift(1), low.shift(1), close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    alpha = 1.0 / period  # Wilder 평활 (= span 2*period-1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=alpha, adjust=False).mean()


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index. 100에 가까우면 횡보(톱니), 0에 가까우면 추세.

    CI = 100 * log10( sum(TR, n) / (maxHigh(n) - minLow(n)) ) / log10(n)
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    rng = (high.rolling(period).max() - low.rolling(period).min()).replace(0, np.nan)
    return 100.0 * np.log10(atr_sum / rng) / np.log10(period)


def efficiency_ratio(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    """Kaufman 효율성 비율 = |순방향 이동| / 총 경로길이. 1에 가까우면 추세, 0이면 톱니."""
    direction = df[col].diff(period).abs()
    volatility = df[col].diff().abs().rolling(period).sum().replace(0, np.nan)
    return direction / volatility


def trend_r2(df: pd.DataFrame, period: int = 20, col: str = "close") -> tuple[float, int]:
    """최근 period봉 종가의 선형회귀 결정계수(R²)와 기울기 부호.

    반환: (r2 0~1, slope_sign -1/0/+1). 데이터 부족 시 (nan, 0).
    """
    y = df[col].to_numpy(dtype=float)[-period:]
    if len(y) < period or np.allclose(y, y[0]):
        return float("nan"), 0
    x = np.arange(len(y), dtype=float)
    r = np.corrcoef(x, y)[0, 1]
    slope_sign = int(np.sign(r))  # 기울기 부호 = 상관 부호
    return float(r * r), slope_sign


# ── 결과 모델 ────────────────────────────────────────────────────────
@dataclass
class IndicatorVote:
    name: str
    value: float
    vote: int          # +1 추세 / -1 횡보 / 0 중립
    note: str = ""

    @property
    def label(self) -> str:
        return {1: "추세", -1: "횡보", 0: "중립"}[self.vote]


@dataclass
class RegimeVerdict:
    regime: str                      # "추세장" / "횡보장" / "전환/중립"
    score: float                     # +면 추세, -면 횡보 (보조지표 0.5 가중)
    confidence: float                # 0~1, 핵심 3지표 중 결론과 일치하는 비율
    direction: str                   # "상승" / "하락" / "-"
    votes: list[IndicatorVote] = field(default_factory=list)
    timestamp: pd.Timestamp | None = None

    def summary(self) -> str:
        arrow = {"상승": "↑", "하락": "↓", "-": " "}[self.direction]
        return (f"{self.regime}{arrow} (score {self.score:+.1f}, "
                f"신뢰도 {self.confidence*100:.0f}%)")


def _vote(value: float, trend_thr: float, range_thr: float, higher_is_trend: bool) -> int:
    """value를 임계값과 비교해 투표. higher_is_trend=True면 값↑=추세."""
    if np.isnan(value):
        return 0
    if higher_is_trend:
        if value >= trend_thr:
            return 1
        if value <= range_thr:
            return -1
    else:  # 값↓=추세 (Choppiness)
        if value <= trend_thr:
            return 1
        if value >= range_thr:
            return -1
    return 0


def classify_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    ci_period: int = 14,
    er_period: int = 20,
    r2_period: int = 20,
    bbw_period: int = 20,
    bbw_lookback: int = 50,
) -> RegimeVerdict:
    """OHLCV DataFrame(확정 봉)을 받아 현재 국면을 판별한다 (상태 없음)."""
    n_min = max(adx_period * 2, ci_period, er_period, r2_period, bbw_period) + 5
    if len(df) < n_min:
        return RegimeVerdict("전환/중립", 0.0, 0.0, "-",
                             timestamp=_last_ts(df))

    adx_v = float(wilder_adx(df, adx_period).iloc[-1])
    ci_v = float(choppiness_index(df, ci_period).iloc[-1])
    er_v = float(efficiency_ratio(df, er_period).iloc[-1])
    r2_v, slope = trend_r2(df, r2_period)

    bw = calc_bb_width(df, bbw_period)
    bw_now = float(bw.iloc[-1])
    recent = bw.iloc[-bbw_lookback:]
    bbw_pct = float((recent < bw_now).mean())  # 0~1 백분위

    # ── 핵심 3지표 (가중 1.0) ──
    v_adx = IndicatorVote("ADX", adx_v, _vote(adx_v, ADX_TREND, ADX_RANGE, True),
                          f">{ADX_TREND:.0f}추세 <{ADX_RANGE:.0f}횡보")
    v_ci = IndicatorVote("Choppiness", ci_v, _vote(ci_v, CI_TREND, CI_RANGE, False),
                         f"<{CI_TREND}추세 >{CI_RANGE}횡보")
    v_er = IndicatorVote("EfficiencyRatio", er_v, _vote(er_v, ER_TREND, ER_RANGE, True),
                         f">{ER_TREND}추세 <{ER_RANGE}횡보")
    core = [v_adx, v_ci, v_er]

    # ── 보조 2지표 (가중 0.5) ──
    v_r2 = IndicatorVote("LinReg R²", r2_v, _vote(r2_v, R2_TREND, R2_RANGE, True),
                         f">{R2_TREND}추세 <{R2_RANGE}무방향")
    bbw_vote = -1 if bbw_pct < BBW_SQUEEZE_PCT else (1 if bbw_pct > BBW_EXPAND_PCT else 0)
    v_bbw = IndicatorVote("BB폭 백분위", bbw_pct, bbw_vote,
                          f"<{BBW_SQUEEZE_PCT:.0%}수축=횡보 >{BBW_EXPAND_PCT:.0%}확장=추세")
    aux = [v_r2, v_bbw]

    score = sum(v.vote for v in core) + 0.5 * sum(v.vote for v in aux)

    if score >= SWITCH_STRONG:
        regime = "추세장"
    elif score <= -SWITCH_STRONG:
        regime = "횡보장"
    else:
        regime = "전환/중립"

    # 신뢰도 = 핵심 3지표 중 최종 결론 부호와 일치하는 비율
    final_sign = 1 if score > 0 else (-1 if score < 0 else 0)
    if final_sign == 0:
        confidence = 0.0
    else:
        agree = sum(1 for v in core if v.vote == final_sign)
        confidence = agree / len(core)

    if regime == "추세장":
        direction = "상승" if slope > 0 else ("하락" if slope < 0 else "-")
    else:
        direction = "-"

    return RegimeVerdict(
        regime=regime, score=score, confidence=confidence, direction=direction,
        votes=core + aux, timestamp=_last_ts(df),
    )


def _last_ts(df: pd.DataFrame) -> pd.Timestamp | None:
    if "timestamp" in df.columns and len(df):
        return df["timestamp"].iloc[-1]
    if len(df):
        return df.index[-1]
    return None


# ── 히스테리시스 스위치 (잦은 깜빡임 방지) ──────────────────────────
class RealtimeRegimeSwitch:
    """실시간 국면 스위치. 강한 신호에서만 상태를 바꾸고 애매하면 직전 상태 유지.

    사용: switch.update(df) → 현재 스위치 상태("추세장"/"횡보장") 반환.
    raw 판독은 verdict로 함께 노출.
    """

    def __init__(self, **classify_kwargs) -> None:
        self._kw = classify_kwargs
        self.state: str = "전환/중립"     # 첫 강신호 전까지 중립
        self.last_verdict: RegimeVerdict | None = None

    def update(self, df: pd.DataFrame) -> str:
        v = classify_regime(df, **self._kw)
        self.last_verdict = v
        # 강신호일 때만 상태 전환, 그 외엔 직전 상태 유지(히스테리시스)
        if v.score >= SWITCH_STRONG:
            self.state = "추세장"
        elif v.score <= -SWITCH_STRONG:
            self.state = "횡보장"
        # else: 상태 유지
        return self.state
