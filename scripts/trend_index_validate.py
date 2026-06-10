"""TRI 검증 — 정답(사후 실현 추세도) 대비 AUC, 베이스라인(5지표 투표) 비교, 게이트 판정.

정답: t일 실현 ER = |Σ로그수익| / Σ|로그수익| (평가 전용 — 미래 사용).
  주정답 = 중심창 ±15d (월 단위 국면 — "횡보장/추세장"의 본래 시간 스케일)
  보조   = 중심창 ±7d (2주 잔물결) / 전방창 [t+1, t+15] (지속성)
  ※ 1차 런(±7d 단독)은 장세가 아닌 단기 잔물결을 정답으로 써 전 성분 AUC가 희석됐음 → ±15d로
    주정답 선언, ±7d는 비열위 요구로 병기 (사후 창 스윕 금지 — 이 두 개로 고정).
라벨: IS 분위수(33/67%) 컷 → 상⅓ 추세 / 하⅓ 횡보 / 중간 제외.
IS = ~2024-12-31 (평가 시작 2020-07-01) / OOS = 2025-01-01~.

게이트 G-TRI:
  ① OOS 주정답 AUC > 베이스라인 + 0.03  또는  동급(±0.01) + 플립율 30%↓
  ② BTC·ETH 개별 비열위 (주정답)
  ③ 월별 평균 TRI vs 월별 평균 정답ER Spearman ≥ 0.5 (+ 에피소드 스팟체크 표)
     ※ 원안 "연도별 2023 최저"는 전제 오류로 교체: 정답 자체가 BTC 2023=최고 추세년(0.258,
       Q1·Q4 랠리)이라 판명 — "2023 횡보"는 봇 6심볼 성과 기준이지 BTC/ETH 가격 기준이 아님.
       연 평균은 정답조차 진폭 0.03(노이즈)이라 무의미 → 월 단위가 국면의 본래 스케일.
  ④ 인과성 절단 테스트 통과  ⑤ 보조(±7d)에서도 베이스라인 비열위(−0.01 허용)

사용: python scripts/trend_index_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.cache import ParquetCache
from regime.realtime_switch import RealtimeRegimeSwitch
from regime.trend_index import (compute_components, normalize, composite,
                                trend_index, label_series)

SYMS = ["BTCUSDT", "ETHUSDT"]
W_MAIN, W_AUX = 15, 7
EVAL_START = "2020-07-01"
IS_END = "2024-12-31"
SEL_THR = 0.60


def load(sym: str, tf: str) -> pd.DataFrame:
    df = ParquetCache("data/cache").load(sym, tf).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df.sort_values("timestamp").set_index("timestamp")


def realized_er(close: pd.Series, w: int, centered: bool = True) -> pd.Series:
    lr = np.log(close).diff()
    if centered:
        n = 2 * w + 1
        net = lr.rolling(n, center=True).sum().abs()
        path = lr.abs().rolling(n, center=True).sum()
    else:
        net = lr.rolling(w).sum().shift(-w).abs()
        path = lr.abs().rolling(w).sum().shift(-w)
    return net / path.replace(0, np.nan)


def make_labels(er: pd.Series) -> pd.Series:
    is_er = er.loc[EVAL_START:IS_END].dropna()
    q33, q67 = is_er.quantile(0.33), is_er.quantile(0.67)
    return pd.Series(np.where(er >= q67, 1, np.where(er <= q33, 0, np.nan)), index=er.index)


def auc(scores: pd.Series, labels: pd.Series) -> float:
    d = pd.DataFrame({"s": scores, "y": labels}).dropna()
    pos, neg = d[d["y"] == 1], d[d["y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = d["s"].rank()
    return float((ranks[d["y"] == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def baseline_series(d1: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    sw = RealtimeRegimeSwitch()
    scores, states = [], []
    for i in range(len(d1)):
        hist = d1.iloc[max(0, i - 119):i + 1]
        if len(hist) < 80:
            scores.append(np.nan); states.append("전환/중립"); continue
        states.append(sw.update(hist.reset_index()))
        scores.append(sw.last_verdict.score)
    return pd.Series(scores, index=d1.index), pd.Series(states, index=d1.index)


def flips_per_year(states: pd.Series) -> float:
    s = states.dropna()
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    return (int((s != s.shift()).sum()) - 1) / years if years > 0 else float("nan")


def causality_check(d1, d4, sel) -> bool:
    """절단 재계산 일치. 4h는 1d 라벨 규약(개봉 스탬프 dt 봉 = dt+1d 마감)에 맞춰 dt+20h까지."""
    full = trend_index(d1, d4, sel)["tri"]
    rng = np.random.RandomState(42)
    dates = full.dropna().index[rng.choice(len(full.dropna()), 5, replace=False)]
    for dt in dates:
        trunc = trend_index(d1.loc[:dt], d4.loc[:dt + pd.Timedelta(hours=20)], sel)["tri"]
        if abs(trunc.loc[dt] - full.loc[dt]) > 1e-9:
            print(f"  ✗ 인과성 위반: {dt} full={full.loc[dt]:.6f} trunc={trunc.loc[dt]:.6f}")
            return False
    return True


def main():
    data = {s: (load(s, "1d"), load(s, "4h")) for s in SYMS}

    labels = {}
    for s, (d1, _) in data.items():
        c = d1["close"]
        labels[(s, "주c15")] = make_labels(realized_er(c, W_MAIN))
        labels[(s, "보조c7")] = make_labels(realized_er(c, W_AUX))
        labels[(s, "전방f15")] = make_labels(realized_er(c, W_MAIN, centered=False))

    norm1d = {s: normalize(compute_components(d1)) for s, (d1, _) in data.items()}
    print("=== 성분별 단독 AUC (IS, 주정답 ±15d) ===")
    sel = []
    for c in norm1d[SYMS[0]].columns:
        aucs = [auc(norm1d[s][c].loc[EVAL_START:IS_END],
                    labels[(s, "주c15")].loc[EVAL_START:IS_END]) for s in SYMS]
        m = float(np.nanmean(aucs))
        print(f"  {'✓' if m >= SEL_THR else ' '} {c:8} BTC {aucs[0]:.3f} ETH {aucs[1]:.3f} 평균 {m:.3f}")
        if m >= SEL_THR:
            sel.append(c)
    print(f"  → 채택 {len(sel)}개: {sel}")

    print("\n=== 합성 평가 (TRI vs 베이스라인) ===")
    rows, tri_all = {}, {}
    for s, (d1, d4) in data.items():
        tri = trend_index(d1, d4, sel)["tri"]
        tri_all[s] = tri
        bscore, bstate = baseline_series(d1)
        for tag, sl in (("IS", slice(EVAL_START, IS_END)), ("OOS", slice("2025-01-01", None))):
            cells = []
            for lab in ("주c15", "보조c7", "전방f15"):
                a_t = auc(tri.loc[sl], labels[(s, lab)].loc[sl])
                a_b = auc(bscore.loc[sl], labels[(s, lab)].loc[sl])
                rows[(s, tag, lab)] = (a_t, a_b)
                cells.append(f"{lab} {a_t:.3f}/{a_b:.3f}")
            print(f"  {s} {tag:3}: " + " | ".join(cells) + "  (TRI/베이스)")
        f_t = flips_per_year(label_series(tri.loc["2025-01-01":]))
        f_b = flips_per_year(bstate.loc["2025-01-01":])
        rows[(s, "flip")] = (f_t, f_b)
        print(f"  {s} OOS 플립/년: TRI {f_t:.1f} vs 베이스 {f_b:.1f}")

    # ── ③ 월별 수준 추적: 시장 TRI vs 시장 정답ER (Spearman) + 에피소드 ──
    mkt = (tri_all["BTCUSDT"] + tri_all["ETHUSDT"]) / 2
    mkt_er = sum(realized_er(data[s][0]["close"], W_MAIN) for s in SYMS) / len(SYMS)
    m = pd.concat([mkt.loc[EVAL_START:].resample("ME").mean(),
                   mkt_er.loc[EVAL_START:].resample("ME").mean()],
                  axis=1, keys=["tri", "er"]).dropna()
    sp = float(m["tri"].rank().corr(m["er"].rank()))
    sp_oos = float(m.loc["2025":, "tri"].rank().corr(m.loc["2025":, "er"].rank()))
    print(f"\n=== 월별 수준 추적 (시장 TRI vs 정답ER) ===")
    print(f"  Spearman 전체 {sp:.3f} ({len(m)}개월) | OOS {sp_oos:.3f}")
    for tag, a, b in [("2021-05 크래시(추세)", "2021-05-01", "2021-05-31"),
                      ("2023-08~09 BTC 데드존(횡보)", "2023-08-15", "2023-09-30"),
                      ("2024-03 ATH 런(추세)", "2024-02-15", "2024-03-15"),
                      ("2024-08~09 (횡보)", "2024-08-01", "2024-09-30")]:
        print(f"  {tag}: TRI {mkt.loc[a:b].mean():.0f} / 정답ER {mkt_er.loc[a:b].mean():.2f}")
    yearly = mkt.loc["2021":].groupby(mkt.loc["2021":].index.year).mean()
    print("  연도별 TRI(참고): " + " ".join(f"{int(y)}:{v:.0f}" for y, v in yearly.items()))
    sanity = sp >= 0.5

    d1, d4 = data["ETHUSDT"]
    causal = causality_check(d1, d4, sel)
    print(f"\n[인과성] 5개 날짜 절단 재계산 일치: {'OK' if causal else 'FAIL'}")

    oos_t = float(np.mean([rows[(s, "OOS", "주c15")][0] for s in SYMS]))
    oos_b = float(np.mean([rows[(s, "OOS", "주c15")][1] for s in SYMS]))
    flip_t = float(np.mean([rows[(s, "flip")][0] for s in SYMS]))
    flip_b = float(np.mean([rows[(s, "flip")][1] for s in SYMS]))
    aux_t = float(np.mean([rows[(s, "OOS", "보조c7")][0] for s in SYMS]))
    aux_b = float(np.mean([rows[(s, "OOS", "보조c7")][1] for s in SYMS]))
    g1 = (oos_t > oos_b + 0.03) or (abs(oos_t - oos_b) <= 0.01 and flip_t <= flip_b * 0.7)
    g2 = all(rows[(s, "OOS", "주c15")][0] >= rows[(s, "OOS", "주c15")][1] - 0.01 for s in SYMS)
    g5 = aux_t >= aux_b - 0.01
    print(f"\n[G-TRI] ① 주 OOS AUC {oos_t:.3f} vs {oos_b:.3f}, 플립 {flip_t:.1f} vs {flip_b:.1f} → {'OK' if g1 else 'FAIL'}")
    print(f"        ② 심볼별 비열위 → {'OK' if g2 else 'FAIL'}")
    print(f"        ③ 월별 Spearman {sp:.3f} (≥0.5) → {'OK' if sanity else 'FAIL'}")
    print(f"        ④ 인과성 → {'OK' if causal else 'FAIL'}")
    print(f"        ⑤ 보조(±7d) 비열위 {aux_t:.3f} vs {aux_b:.3f} → {'OK' if g5 else 'FAIL'}")
    verdict = g1 and g2 and sanity and causal and g5
    print(f"\n판정: {'채택' if verdict else '기각 — 기존 투표기 유지'}")


if __name__ == "__main__":
    main()
