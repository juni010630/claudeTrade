"""3-way 블렌드: 추세(엔진) × 슬리브(엔진) × 신규 macross(격리 시뮬) — 월간 리밸.

입력: data/results/comp_trend_daily.parquet, comp_sleeve_daily.parquet, newedge_ma20x100_eq.parquet
가중 그리드 전수 + 1:1:1 하이라이트. 연도별/OOS/MDD 병기.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

RES = Path(__file__).parent.parent / "data" / "results"


def load_ret(fname):
    eq = pd.read_parquet(RES / fname)["eq"]
    eq.index = pd.DatetimeIndex(eq.index)
    if eq.index.tz is None:
        eq.index = eq.index.tz_localize("UTC")
    return eq.pct_change().dropna()


def blend(rets, ws):
    j = pd.concat(rets, axis=1).fillna(0.0)
    caps = list(ws)
    out = []
    cur_month = None
    for d, row in j.iterrows():
        if cur_month is not None and d.month != cur_month:
            tot = sum(caps)
            caps = [w * tot for w in ws]
        cur_month = d.month
        caps = [c * (1 + r) for c, r in zip(caps, row.values)]
        out.append((d, sum(caps)))
    return pd.Series(dict(out)).sort_index()


def stats(eq):
    ret = eq.pct_change().dropna()
    sh = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    yr = {}
    for y in (2022, 2023, 2024, 2025, 2026):
        s = eq[(eq.index >= f"{y}-01-01") & (eq.index <= f"{y}-12-31")]
        if len(s) > 30:
            yr[y] = round((s.iloc[-1] / s.iloc[0] - 1) * 100)
    oos = eq[eq.index >= "2025-01-01"]
    orr = oos.pct_change().dropna()
    osh = orr.mean() / orr.std() * np.sqrt(365) if orr.std() > 0 else 0
    omdd = ((oos - oos.cummax()) / oos.cummax()).min()
    return (round(eq.iloc[-1] / eq.iloc[0], 2), round(sh, 2), round(mdd * 100, 1), yr,
            round(osh, 2), round(omdd * 100, 1))


def main():
    r_t = load_ret("comp_trend_daily.parquet")
    r_s = load_ret("comp_sleeve_daily.parquet")
    r_n = load_ret("newedge_ma20x100_eq.parquet")
    rets = [r_t, r_s, r_n]
    j = pd.concat(rets, axis=1).fillna(0.0)
    j.columns = ["trend", "sleeve", "newedge"]
    print("일별 상관:")
    print(j.corr().round(3).to_string())
    print()
    print(f"{'T:S:N':<12} {'x배':>8} {'Sh':>5} {'MDD%':>7} {'OOS_Sh':>7} {'OOS_MDD':>8}  연도별")
    grid = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (50, 50, 0),            # 현행 v17 구조 근사
        (33.3, 33.3, 33.3),     # 1:1:1
        (40, 30, 30), (30, 40, 30), (30, 30, 40),
        (50, 25, 25), (25, 50, 25), (25, 25, 50),
        (40, 40, 20), (40, 20, 40), (20, 40, 40),
        (60, 20, 20), (20, 60, 20), (20, 20, 60),
    ]
    for w in grid:
        tot = sum(w)
        ws = [x / tot for x in w]
        eq = blend(rets, ws)
        fx, sh, mdd, yr, osh, omdd = stats(eq)
        tag = ":".join(str(int(round(x / tot * 100))) for x in w)
        print(f"{tag:<12} {fx:>8} {sh:>5} {mdd:>7} {osh:>7} {omdd:>8}  {yr}")


if __name__ == "__main__":
    main()
