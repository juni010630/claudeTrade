"""merged 단일계좌 v16:슬리브 비중별 분기 수익률 — 5:5/6:4/7:3/8:2. 출력+CSV."""
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
    vf, sf = split
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


def qret(r):
    return ((1+r).prod()-1)*100


def main():
    splits = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    names = {(0.5,0.5):"5:5", (0.6,0.4):"6:4", (0.7,0.3):"7:3", (0.8,0.2):"8:2"}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(ex.map(run, splits))

    out = {}
    for sp in splits:
        q = res[sp].resample("QE").apply(qret).round(1)
        out[names[sp]] = q
    q = pd.DataFrame(out)
    q.index = [f"{t.year}Q{t.quarter}" for t in q.index]

    print("=== merged 단일계좌 v16:슬리브 비중별 분기 수익률(%) ===")
    print(q.to_string())

    print("\n=== 전체 요약 ===")
    print(f"{'비중':>5} {'최종$':>9} {'총수익%':>9} {'Sharpe':>7} {'MDD%':>6}")
    for sp in splits:
        d = res[sp]; eq = (1+d).cumprod()
        mdd = ((eq/eq.cummax()-1)).min()*100
        sh = d.mean()/d.std()*np.sqrt(365)
        print(f"{names[sp]:>5} ${eq.iloc[-1]*100:>8,.0f} {(eq.iloc[-1]-1)*100:>+9.0f} {sh:>7.2f} {mdd:>6.0f}")

    Path("data/results").mkdir(exist_ok=True)
    q.to_csv("data/results/merged_quarterly.csv")
    print("\n저장: data/results/merged_quarterly.csv")


if __name__ == "__main__":
    main()
