"""adaptive_tpsl.py — 국면/변동성 조건부 TP/SL 규칙을 진입 덤프에 적용.

핵심 설계: 규칙은 베이스라인 배수에 대한 **스케일(배율)** 을 버킷별로 정의한다.
  effective_tp_mult = baseline_tp_mult[strategy] * tp_scale[bucket]
  effective_sl_mult = baseline_sl_mult[strategy] * sl_scale[bucket]

이렇게 하면:
  - 전략별 베이스라인 차이(ema 3.5 / multi_tf 4.0)가 그대로 보존됨
  - 항등 규칙(모든 스케일 = 1.0)은 기존 고정 TP/SL과 **정확히 일치** → 하네스 무결성 회귀 테스트 자명
  - 그리드 탐색이 베이스라인(1.0)을 중심으로 tighter(<1)/wider(>1)를 대칭 탐색

버킷: 진입은 entry 필터상 TRENDING/PRE_BREAKOUT 에서만 발생하므로
실질 조건축은 추세강도(adx)와 변동성(bb_width_pct). 과적합 억제를 위해 2×2 로 거칠게 나눈다.

TP/SL 가격 계산 수식은 strategies/*.py 및 scripts/fast_sweep.py:47-50 과 동일.
"""
from __future__ import annotations

import itertools

import pandas as pd

# config/final_v13_eth.yaml 의 전략별 베이스라인 배수
BASELINE_TP = {"ema_cross": 3.5, "multi_tf_breakout": 4.0}
BASELINE_SL = {"ema_cross": 1.8, "multi_tf_breakout": 1.8}

# 2×2 버킷 키
BUCKETS = ("hiadx_lobw", "hiadx_hibw", "loadx_lobw", "loadx_hibw")

# 항등 규칙: 모든 버킷 스케일 1.0 → 베이스라인과 동일
IDENTITY_RULE = {b: (1.0, 1.0) for b in BUCKETS}


def assign_bucket(
    df: pd.DataFrame, adx_split: float = 25.0, bbw_split: float = 0.5
) -> pd.Series:
    """각 진입 행을 (adx 고/저) × (bb_width_pct 고/저) 2×2 버킷에 배정."""
    for col in ("adx", "bb_width_pct"):
        if col not in df.columns:
            raise KeyError(
                f"entries 덤프에 '{col}' 컬럼이 없습니다. "
                "run_fill_dump 를 갱신된 엔진으로 다시 실행해 컨텍스트를 포함시키세요."
            )
    adx_hi = df["adx"] >= adx_split
    bbw_hi = df["bb_width_pct"] >= bbw_split
    bucket = pd.Series(index=df.index, dtype=object)
    bucket[adx_hi & ~bbw_hi] = "hiadx_lobw"
    bucket[adx_hi & bbw_hi] = "hiadx_hibw"
    bucket[~adx_hi & ~bbw_hi] = "loadx_lobw"
    bucket[~adx_hi & bbw_hi] = "loadx_hibw"
    return bucket


def apply_adaptive_tpsl(
    df: pd.DataFrame,
    rule: dict[str, tuple[float, float]],
    baseline_tp: dict[str, float] = BASELINE_TP,
    baseline_sl: dict[str, float] = BASELINE_SL,
    adx_split: float = 25.0,
    bbw_split: float = 0.5,
) -> pd.DataFrame:
    """규칙(버킷→(tp_scale, sl_scale))을 적용해 행별 tp_price/sl_price 재계산.

    rule 의 모든 스케일이 1.0 이면 입력과 동일한 tp/sl 을 반환(항등).
    반환 DataFrame 은 입력 복사본이며 tp_price/sl_price/sl_dist 가 갱신된다.
    """
    out = df.copy()

    base_sl = out["strategy"].map(baseline_sl)
    base_tp = out["strategy"].map(baseline_tp)
    if base_sl.isna().any() or base_tp.isna().any():
        unknown = sorted(set(out.loc[base_sl.isna(), "strategy"]))
        raise KeyError(f"베이스라인 배수가 없는 전략: {unknown}")

    # ATR 역산: sl_dist = baseline_sl_mult * ATR
    atr = out["sl_dist"] / base_sl

    bucket = assign_bucket(out, adx_split, bbw_split)
    tp_scale = bucket.map(lambda b: rule[b][0])
    sl_scale = bucket.map(lambda b: rule[b][1])

    eff_tp = base_tp * tp_scale
    eff_sl = base_sl * sl_scale

    long = out["direction"] == "long"
    tp_dist = eff_tp * atr
    sl_dist = eff_sl * atr

    out.loc[long, "tp_price"] = out.loc[long, "entry_price"] + tp_dist[long]
    out.loc[~long, "tp_price"] = out.loc[~long, "entry_price"] - tp_dist[~long]
    out.loc[long, "sl_price"] = out.loc[long, "entry_price"] - sl_dist[long]
    out.loc[~long, "sl_price"] = out.loc[~long, "entry_price"] + sl_dist[~long]
    # 다운스트림 일관성을 위해 sl_dist 도 갱신 (항상 양수)
    out["sl_dist"] = sl_dist.abs()
    return out


def make_scale_grid(
    tp_scales: tuple[float, ...] = (0.7, 1.0, 1.3),
    sl_scales: tuple[float, ...] = (0.7, 1.0, 1.3),
    buckets: tuple[str, ...] = BUCKETS,
):
    """버킷별 (tp_scale, sl_scale) 조합의 전수 그리드를 lazy 하게 생성.

    주의: 버킷마다 독립이면 조합수가 (|tp|*|sl|)^|buckets| 로 폭발한다.
    기본값(3*3)^4 = 6561. 워크포워드 폴드마다 이만큼 리플레이는 과함 →
    harness 에서 버킷 수를 줄이거나(adx만, 2버킷) 좌표하강(coordinate descent)으로 탐색 권장.
    이 함수는 작은 버킷 집합용 전수 그리드 유틸.
    """
    per_bucket = list(itertools.product(tp_scales, sl_scales))
    for combo in itertools.product(per_bucket, repeat=len(buckets)):
        yield {b: combo[i] for i, b in enumerate(buckets)}
