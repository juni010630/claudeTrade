"""v17 × 신규 슬리브(macross-1d) 합산 검증 — 월간 리밸런스 블렌드.

v17 일별 수익 = v17_trades_full.csv 청산일 귀속 PnL로 실현 equity 재구성 ($100 시작).
블렌드 = 각 슬리브 독립 복리 + 월말 목표가중 리셋 (매일리밸 가정 금지 — 블렌드 사이징 교훈).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

RES = Path(__file__).parent.parent / "data" / "results"


def v17_ret():
    t = pd.read_csv(RES / "v17_trades_full.csv", parse_dates=["exit_time"])
    pnl = t.groupby(t.exit_time.dt.floor("D")).pnl.sum()
    idx = pd.date_range(pnl.index.min(), pnl.index.max(), freq="D", tz="UTC")
    pnl = pnl.reindex(idx).fillna(0.0)
    eq = 100 + pnl.cumsum()
    return eq.pct_change().dropna()


def ne_ret(fname):
    eq = pd.read_parquet(RES / fname)["eq"]
    eq.index = pd.DatetimeIndex(eq.index)
    if eq.index.tz is None:
        eq.index = eq.index.tz_localize("UTC")
    return eq.pct_change().dropna()


def blend(r1, r2, w1, w2):
    j = pd.concat([r1, r2], axis=1).fillna(0.0)
    j.columns = ["a", "b"]
    c1, c2 = w1, w2
    out = []
    cur_month = None
    for d, row in j.iterrows():
        if cur_month is not None and d.month != cur_month:
            tot = c1 + c2
            c1, c2 = w1 * tot, w2 * tot
        cur_month = d.month
        c1 *= 1 + row.a
        c2 *= 1 + row.b
        out.append((d, c1 + c2))
    eq = pd.Series(dict(out)).sort_index()
    return eq


def stats(eq, label):
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
    print(f"{label:<22} x{round(eq.iloc[-1]/eq.iloc[0],2):>8} Sh={round(sh,2):<5} MDD={round(mdd*100,1):>6}% "
          f"yr={yr} | OOS Sh={round(osh,2)} MDD={round(omdd*100,1)}%")


def main():
    ne_file = sys.argv[1] if len(sys.argv) > 1 else "newedge_ma20x100_eq.parquet"
    r_v17 = v17_ret()
    r_ne = ne_ret(ne_file)
    j = pd.concat([r_v17, r_ne], axis=1).dropna()
    j.columns = ["v17", "newedge"]
    print(f"== {ne_file} | 일별상관 all={round(j.v17.corr(j.newedge),3)} ==")
    for w1, w2 in ((1.0, 0.0), (0.85, 0.15), (0.70, 0.30), (0.50, 0.50), (0.0, 1.0)):
        eq = blend(r_v17, r_ne, w1, w2)
        stats(eq, f"v17:{int(w1*100)} ne:{int(w2*100)}")


if __name__ == "__main__":
    main()
