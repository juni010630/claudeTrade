"""단일계좌 병합 vs 2계좌 블렌드 비교.
병합(merged_v16_sleeve, 자동연속리밸) vs 2계좌(v16+슬리브 일간/월간리밸) vs v16단독.
병합이 2계좌 일간블렌드에 근접하면 = 단일계좌로 수동리밸 없이 분산 실현 성공."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    ntr = len(eng.ledger.records)
    return cfg, d, ntr


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, (eq.iloc[-1]-1)*100, sh, mdd, yr


def blend_monthly(v, s):
    months = v.index.to_period("M")
    eq = pd.Series(index=v.index, dtype=float); total = 1.0
    for m in months.unique():
        mask = months == m
        seg = 0.5*(1+v[mask]).cumprod() + 0.5*(1+s[mask]).cumprod()
        eq[mask] = total*seg.values; total = eq[mask].iloc[-1]
    return eq


def show(tag, eq, ntr=None):
    daily = eq.pct_change().fillna(0)
    f, tot, sh, mdd, yr = stats(daily)
    ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
    t = f" 거래{ntr}" if ntr is not None else ""
    print(f"  {tag:26} ${f:>9,.0f} 총{tot:>+7.0f}% Sh{sh:.2f} MDD{mdd:>4.0f}%{t}  {ys}")


def main():
    jobs = [("config/merged_v16_sleeve.yaml", "1h"),
            ("config/final_v16_slwide.yaml", "1h"),
            ("config/sleeve_meanrev.yaml", "1d")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        res = {c: (d, n) for c, d, n in ex.map(run, jobs)}
    dm, nm = res["config/merged_v16_sleeve.yaml"]
    dv, nv = res["config/final_v16_slwide.yaml"]
    ds, ns = res["config/sleeve_meanrev.yaml"]

    j = pd.concat([dv, ds], axis=1, keys=["v", "s"]).dropna()

    print("=== 단일계좌 병합 vs 2계좌 블렌드 ===")
    show("v16 단독(funding3)", (1+dv).cumprod(), nv)
    show("슬리브 단독", (1+ds).cumprod(), ns)
    print("  --- 2계좌 50:50 (참고) ---")
    show("  일간리밸", (1+0.5*j.v+0.5*j.s).cumprod())
    show("  월간리밸", blend_monthly(j.v, j.s))
    print("  --- 단일계좌 병합 (자동 연속 리밸) ---")
    show("merged_v16_sleeve", (1+dm).cumprod(), nm)
    print("\n판정: 병합이 2계좌 일간/월간블렌드에 근접+거래수 합리적이면 = 단일계좌 성공(수동리밸 불요).")


if __name__ == "__main__":
    main()
