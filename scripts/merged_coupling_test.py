"""병합 저조 원인 진단 — 공유 리스크한도(corr/daily DD stop) 끄면 회복되나.
회복되면 = 공유 리스크 결합이 주범(per-strategy 리스크한도 필요). 안되면 = 1h 마찰."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(variant):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/merged_v16_sleeve.yaml")))
    if variant == "corr_off":
        p["risk"]["correlation_block_threshold"] = 1.01
    elif variant == "ddstop_off":
        p["risk"]["daily_drawdown_stop"] = -1.0
        p["risk"]["daily_drawdown_pause"] = -1.0
    elif variant == "both_off":
        p["risk"]["correlation_block_threshold"] = 1.01
        p["risk"]["daily_drawdown_stop"] = -1.0
        p["risk"]["daily_drawdown_pause"] = -1.0
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return variant, d


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def main():
    variants = ["base", "corr_off", "ddstop_off", "both_off"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(run, variants))
    print("=== 병합 공유리스크 결합 진단 (2023 회복=결합 주범) ===")
    print("  [목표: 2계좌 일간 Sh1.65/MDD-49%/2023+39%]")
    for v in variants:
        f, sh, mdd, yr = stats(res[v])
        ys = " ".join(f"{int(k)}:{round(x):+}" for k, x in yr.items())
        print(f"  {v:12} ${f:>8,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


if __name__ == "__main__":
    main()
