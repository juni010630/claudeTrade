"""게이트 모드 비교 — off / 진입만차단 / 진입+플랫, 임계 0.18/0.22/0.28.
엔진 실제 체결. 단독 + v16블렌드(50:50). 연도별."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

BASE = "config/sleeve_meanrev.yaml"


def run(args):
    mode, thr = args  # mode: "off"|"entry"|"flat"
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    mr = p["strategies"]["mean_reversion"]
    if mode != "off":
        mr["trend_gate_enabled"] = True
        mr["trend_gate_er_period"] = 20
        mr["trend_gate_er_threshold"] = thr
        mr["trend_gate_flatten"] = (mode == "flat")
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return (mode, thr, d)


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def main():
    jobs = [("off", 0.0)]
    for thr in [0.18, 0.22, 0.28]:
        jobs += [("entry", thr), ("flat", thr)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=7) as ex:
        res = list(ex.map(run, jobs))
    v = pd.read_parquet("data/results/dyn_series.parquet")["v16"]

    def line(tag, d):
        f, sh, mdd, yr = stats(d)
        j = pd.concat([v, d], axis=1, keys=["v", "s"]).dropna()
        bf, bsh, bmdd, _ = stats(0.5*j.v + 0.5*j.s)
        ys = " ".join(f"{int(k)}:{round(x):+}" for k, x in yr.items())
        print(f"  {tag:24} 단독 ${f:>7,.0f} Sh{sh:>5.2f} MDD{mdd:>4.0f}% | "
              f"50:50 ${bf:>9,.0f} Sh{bsh:.2f} MDD{bmdd:>4.0f}% | {ys}")

    print("=== 게이트 모드 비교 (단독 + v16 50:50 블렌드) ===")
    for mode, thr, d in res:
        tag = "OFF(always-on)" if mode == "off" else f"{mode} thr{thr}"
        line(tag, d)


if __name__ == "__main__":
    main()
