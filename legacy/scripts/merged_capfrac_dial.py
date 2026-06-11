"""merged 시스템 capital_fraction 다이얼 — v16:슬리브 비중별 MDD/수익 프론티어.
순수 비중조절(robust, 과적합 아님). MDD 목표에 맞춰 선택용. 연도별 포함."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(split):
    vf, sf = split  # v16 fraction, sleeve fraction
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/merged_v16_sleeve.yaml")))
    p["strategy_capital_fraction"] = {"ema_cross": vf, "multi_tf_breakout": vf, "mean_reversion": sf}
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return split, d


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, (eq.iloc[-1]-1)*100, sh, mdd, yr


def main():
    splits = [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.6, 0.4)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(run, splits))
    print("=== merged capital_fraction 다이얼 (v16:슬리브, daily MDD) ===")
    print(f"{'v16:슬리브':>10} {'최종$':>9} {'총수익%':>9} {'Sh':>5} {'MDD%':>6}  연도별")
    for sp in sorted(splits, key=lambda x: -x[0]):
        f, tot, sh, mdd, yr = stats(res[sp])
        ys = " ".join(f"{int(k)}:{round(x):+}" for k, x in yr.items())
        print(f"{int(sp[0]*100)}:{int(sp[1]*100):<7} ${f:>8,.0f} {tot:>+9.0f} {sh:>5.2f} {mdd:>6.0f}  {ys}")


if __name__ == "__main__":
    main()
