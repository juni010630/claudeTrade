"""merged 분해 + 고수익 프론티어.
1) 분해: 단일계좌서 v16-only(cap0.5)/sleeve-only(cap0.5)/both → both MDD가 v16-only와 같으면 'v16 녹아내림'.
   + both 최저점에 v16-only·sleeve-only 각각 드로다운.
2) 고수익: v16 비중↑ (60:40/70:30/80:20) MDD/수익 출력."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(spec):
    name, vf, sf, only = spec
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/merged_v16_sleeve.yaml")))
    p["strategy_capital_fraction"] = {"ema_cross": vf, "multi_tf_breakout": vf, "mean_reversion": sf}
    if only == "v16":
        p["strategies"]["mean_reversion"]["enabled"] = False
    elif only == "sleeve":
        p["strategies"]["ema_cross"]["enabled"] = False
        p["strategies"]["multi_tf_breakout"]["enabled"] = False
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return name, d


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    return eq.iloc[-1]*100, (eq.iloc[-1]-1)*100, sh, mdd, eq


def dd(eq):
    return eq/eq.cummax()-1


def main():
    specs = [
        ("both 50:50", 0.5, 0.5, None),
        ("v16-only(0.5)", 0.5, 0.0, "v16"),
        ("sleeve-only(0.5)", 0.0, 0.5, "sleeve"),
        ("60:40", 0.6, 0.4, None),
        ("70:30", 0.7, 0.3, None),
        ("80:20", 0.8, 0.2, None),
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = dict(ex.map(run, specs))

    print("=== 1) 분해: single account v16-only vs sleeve-only vs both ===")
    eqs = {}
    for name in ["both 50:50", "v16-only(0.5)", "sleeve-only(0.5)"]:
        f, tot, sh, mdd, eq = stats(res[name]); eqs[name] = eq
        print(f"  {name:18} ${f:>8,.0f} 총{tot:>+7.0f}% Sh{sh:.2f} MDD{mdd:>5.1f}%")
    # both 최저점 분해
    both_eq = eqs["both 50:50"]
    trough = dd(both_eq).idxmin()
    print(f"\n  both 최저점({trough.date()}) 그날:")
    print(f"    v16-only 드로다운  : {dd(eqs['v16-only(0.5)']).reindex([trough]).iloc[0]*100:>5.1f}%")
    print(f"    sleeve-only 드로다운: {dd(eqs['sleeve-only(0.5)']).reindex([trough]).iloc[0]*100:>5.1f}%")
    print(f"  → both MDD가 v16-only와 비슷하면 'v16 녹아내림'이 주범, sleeve가 낮추면 진짜 분산")

    print("\n=== 2) 고수익 프론티어 (v16 비중↑) ===")
    print(f"{'v16:슬리브':>10} {'최종$':>9} {'총수익%':>9} {'Sh':>5} {'MDD%':>6}")
    for name in ["both 50:50", "60:40", "70:30", "80:20"]:
        f, tot, sh, mdd, _ = stats(res[name])
        print(f"  {name:8} ${f:>8,.0f} {tot:>+9.0f} {sh:>5.2f} {mdd:>6.1f}")


if __name__ == "__main__":
    main()
