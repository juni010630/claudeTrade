"""TRI (Trend-Range Index) — 횡보/추세 구분 연속 지수 (0~100, 정보용).

기존 5지표 투표(realtime_switch)와 달리: ① 연속 통계량 ② 고정 임계 대신
트레일링 1년 분위수 인과 정규화 ③ 등가중 합성 (가중치 학습 없음 — 과적합 방지).

성분 (전부 롤링·인과 — t 시점까지 데이터만 사용):
  vr5/vr10   분산비율 (Lo-MacKinlay): q일 수익 분산 / (q × 1일 분산). >1 추세
  hurst      허스트 지수 (스케일별 변위 표준편차 기울기). >0.5 추세
  acsum      수익률 자기상관 합 (lag 1~5). + = 추세
  dircon     방향일관성: 최근 20일 중 순변위 방향과 같은 날 비율
  tstat      드리프트/노이즈: |20일 평균수익| / 표준편차 × √20
  adx_w      Wilder ADX (realtime_switch 재사용)
  chop       Choppiness (방향: 낮을수록 추세 → 부호 반전)
  er         Kaufman 효율비
  r2         선형회귀 R² (롤링)
  bbwpct     BB폭 트레일링 백분위

검증·성분 선별 = scripts/trend_index_validate.py → TREND_INDEX_RESULTS.md.
스위칭 배선 금지 (정보용 — 2026-06-10 동적 전환 전 격자 기각).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regime.realtime_switch import choppiness_index, efficiency_ratio, wilder_adx
from indicators.volatility import bb_width as calc_bb_width

# 합성 채택 성분 — trend_index_validate.py IS(2020-07~2024) 단독 AUC>=0.60 선별 결과
# (er .744 / tstat .743 / r2 .737 / bbwpct .714 / dircon .663 / chop .643 — TREND_INDEX_RESULTS.md)
DEFAULT_SELECTED = ["dircon", "tstat", "chop", "er", "r2", "bbwpct"]
SMOOTH_SPAN = 3          # 합성 후 EMA 평활

# 고정 앵커 시그모이드 정규화: (anchor, scale) — 정설/통계 기준점, 데이터 튜닝 없음.
# (트레일링 분위수 정규화는 연도 간 수준 정보를 지움 — 매년 평균이 50으로 강제되는 결함)
ANCHORS: dict[str, tuple[float, float]] = {
    "vr5": (1.0, 0.25),      # 분산비율: 랜덤워크=1
    "vr10": (1.0, 0.25),
    "hurst": (0.5, 0.10),    # 랜덤워크 H=0.5
    "acsum": (0.0, 0.50),    # 무자기상관=0
    "dircon": (0.62, 0.12),  # 랜덤워크 기대 일치율(~0.6) 상회 여부
    "tstat": (1.0, 0.60),    # |t|=1 기준
    "adx_w": (22.5, 5.0),    # Wilder 정설 20/25 중점
    "chop": (-50.0, 11.8),   # CI 피보나치 38.2/61.8 중점 (부호반전 입력)
    "er": (0.275, 0.075),    # Kaufman 0.20/0.35 중점
    "r2": (0.40, 0.20),      # 0.2/0.6 중점
    "bbwpct": (0.5, 0.25),   # 자체 백분위 (0~1)
}
TF_4H_WEIGHT = 0.5       # 1d 1.0 : 4h 0.5 블렌드
THR_TREND, THR_RANGE = 65.0, 35.0   # 라벨 임계 (히스테리시스: 사이값은 직전 상태 유지)


# ── 성분 통계 (입력: OHLCV df, 출력: 추세=+ 방향 정렬된 Series) ──────────
def variance_ratio(close: pd.Series, q: int, window: int = 120) -> pd.Series:
    lr = np.log(close).diff()
    var1 = lr.rolling(window).var()
    varq = lr.rolling(q).sum().rolling(window).var()
    return varq / (q * var1.replace(0, np.nan))


def hurst(close: pd.Series, window: int = 90, lags=(2, 4, 8, 16)) -> pd.Series:
    logp = np.log(close.to_numpy(dtype=float))
    out = np.full(len(logp), np.nan)
    ll = np.log(lags)
    for i in range(window, len(logp)):
        seg = logp[i - window:i + 1]
        stds = np.array([np.std(seg[l:] - seg[:-l]) for l in lags])
        if stds.min() > 0:
            out[i] = np.polyfit(ll, np.log(stds), 1)[0]
    return pd.Series(out, index=close.index)


def autocorr_sum(close: pd.Series, window: int = 60, max_lag: int = 5) -> pd.Series:
    r = close.pct_change()
    s = None
    for k in range(1, max_lag + 1):
        c = r.rolling(window).corr(r.shift(k))
        s = c if s is None else s + c
    return s


def directional_consistency(close: pd.Series, n: int = 20) -> pd.Series:
    up_frac = (close.diff() > 0).rolling(n).mean()
    net_up = close.diff(n) > 0
    return up_frac.where(net_up, 1.0 - up_frac)


def drift_tstat(close: pd.Series, n: int = 20) -> pd.Series:
    lr = np.log(close).diff()
    return (lr.rolling(n).mean() / lr.rolling(n).std().replace(0, np.nan)).abs() * np.sqrt(n)


def rolling_r2(close: pd.Series, n: int = 20) -> pd.Series:
    t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
    return close.rolling(n).corr(t) ** 2


def _bbw_pct(df: pd.DataFrame, period: int = 20, lookback: int = 50) -> pd.Series:
    bw = calc_bb_width(df, period)
    return bw.rolling(lookback).rank(pct=True)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV df → 성분별 (추세=+) 원시 시리즈. index는 df.index 그대로."""
    close = df["close"]
    return pd.DataFrame({
        "vr5": variance_ratio(close, 5),
        "vr10": variance_ratio(close, 10),
        "hurst": hurst(close),
        "acsum": autocorr_sum(close),
        "dircon": directional_consistency(close),
        "tstat": drift_tstat(close),
        "adx_w": wilder_adx(df),
        "chop": -choppiness_index(df),     # 낮을수록 추세 → 반전
        "er": efficiency_ratio(df),
        "r2": rolling_r2(close),
        "bbwpct": _bbw_pct(df),
    }, index=df.index)


def normalize(comp: pd.DataFrame) -> pd.DataFrame:
    """고정 앵커 시그모이드 정규화 (0~1) — 인과·무상태, 연도 간 수준 비교 가능."""
    out = {}
    for c in comp.columns:
        anchor, scale = ANCHORS[c]
        out[c] = 1.0 / (1.0 + np.exp(-(comp[c] - anchor) / scale))
    return pd.DataFrame(out, index=comp.index)


def composite(norm: pd.DataFrame, selected: list[str] | None = None) -> pd.Series:
    sel = selected or DEFAULT_SELECTED
    return norm[sel].mean(axis=1)


def trend_index(df_1d: pd.DataFrame, df_4h: pd.DataFrame | None = None,
                selected: list[str] | None = None) -> pd.DataFrame:
    """TRI 시계열. 반환: index=1d, columns=[tri, tri_1d, tri_4h] (0~100).

    df는 timestamp 인덱스(또는 timestamp 컬럼) OHLCV. 완성봉만 넣을 것 (look-ahead 책임은 호출자).
    """
    def _prep(df):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        return df.sort_index()

    d1 = _prep(df_1d)
    c1 = composite(normalize(compute_components(d1)), selected)

    if df_4h is not None:
        d4 = _prep(df_4h)
        c4 = composite(normalize(compute_components(d4)), selected)
        c4_d = c4.resample("1D").last().reindex(c1.index).ffill(limit=2)
        blended = (c1 + TF_4H_WEIGHT * c4_d) / (1.0 + TF_4H_WEIGHT)
        blended = blended.fillna(c1)
    else:
        c4_d = pd.Series(np.nan, index=c1.index)
        blended = c1

    tri = blended.ewm(span=SMOOTH_SPAN, adjust=False).mean() * 100.0
    return pd.DataFrame({"tri": tri, "tri_1d": c1 * 100.0, "tri_4h": c4_d * 100.0})


def label_series(tri: pd.Series, thr_trend: float = THR_TREND,
                 thr_range: float = THR_RANGE) -> pd.Series:
    """히스테리시스 라벨: 임계 돌파 시만 전환, 사이값은 직전 상태 유지."""
    out, state = [], "전환/중립"
    for v in tri:
        if pd.notna(v):
            if v >= thr_trend:
                state = "추세장"
            elif v <= thr_range:
                state = "횡보장"
        out.append(state)
    return pd.Series(out, index=tri.index)
