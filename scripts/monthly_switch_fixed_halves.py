"""고정 임계 스위치 H1/H2 분해 — 0.22가 in-sample 엿보기였나, 고정 프라이어도 robust한가.
저장된 funding3 v16 + 슬리브 시계열 재사용(백테X). 고정 임계 0.18/0.20/0.22 × 스킴 C/B."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def er(c, n=20):
    ch = (c - c.shift(n)).abs(); vol = c.diff().abs().rolling(n).sum()
    return ch/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def monthly_blend(v, s, wv_of_month):
    months = v.index.to_period("M"); eq = pd.Series(index=v.index, dtype=float); total = 1.0
    for m in months.unique():
        mask = months == m; wv = wv_of_month(m)
        seg = wv*(1+v[mask]).cumprod() + (1-wv)*(1+s[mask]).cumprod()
        eq[mask] = total*seg.values; total = eq[mask].iloc[-1]
    return eq


def st(eq):
    daily = eq.pct_change().fillna(0)
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    return eq.iloc[-1]/eq.iloc[0]*100-100, sh, mdd


def seg_stats(eq, a, b):
    seg = eq[(eq.index >= a) & (eq.index < b)]
    return st(seg/seg.iloc[0])


def main():
    df = pd.read_parquet("data/results/funding3_sleeve_daily.parquet")
    v, s = df["v"], df["s"]
    j = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()
    be = pd.concat([er(load_close(x), 20) for x in SLEEVE], axis=1).mean(axis=1)
    if be.index.tz is not None: be.index = be.index.tz_localize(None)
    be = be.reindex(j.v.index, method="ffill")
    months = j.v.index.to_period("M")
    sig = {}
    for m in months.unique():
        prev_end = j.v.index[months == m][0] - pd.Timedelta(days=1)
        pr = be[be.index <= prev_end]; sig[m] = pr.iloc[-1] if len(pr) else be.iloc[0]

    spans = [("full", "2022-01-01", "2026-04-23"), ("H1", "2022-01-01", "2024-01-01"),
             ("H2", "2024-01-01", "2026-04-23")]

    def show(tag, eq):
        cells = []
        for name, a, b in spans:
            r, sh, mdd = seg_stats(eq, a, b)
            cells.append(f"{name} {sh:.2f}/{mdd:.0f}%")
        print(f"  {tag:24} {' | '.join(cells)}")

    print("=== 고정 임계 스위치 H1/H2 (Sharpe/MDD) ===")
    show("정적 50:50", monthly_blend(j.v, j.s, lambda m: 0.5))
    for thr in [0.18, 0.20, 0.22]:
        print(f"  -- thr {thr} --")
        show(f"C 70:30/30:70", monthly_blend(j.v, j.s,
             lambda m, t=thr: 0.7 if sig[m] > t else 0.3))
        show(f"B 80:20/50:50", monthly_blend(j.v, j.s,
             lambda m, t=thr: 0.8 if sig[m] > t else 0.5))
    print("\n판정: 스킴이 H1·H2 *양쪽* 정적 대비 개선해야 robust. 한쪽만이면 과적합/국면의존.")


if __name__ == "__main__":
    main()
