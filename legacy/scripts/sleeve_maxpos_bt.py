"""확장 슬리브 + 동시포지션 제어 — max_positions 그리드로 꼬리군집 억제 검증.
기존5 vs +4, max_positions 2/3/4/5 병렬. 수익 핏 아닌 리스크 컨트롤 레버 검증."""
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
EX5 = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
P4 = EX5 + ["MTLUSDT", "BELUSDT", "SNXUSDT", "ONTUSDT"]


def run_one(args):
    name, syms, maxpos = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    p["symbols"] = syms
    p["strategies"]["mean_reversion"]["symbols"] = syms
    p["risk"]["max_positions"] = maxpos
    p["risk"]["max_same_direction"] = maxpos
    loader = DataLoader(symbols=syms, timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0.0
    return (name, maxpos, eq.iloc[-1], sh, mdd)


def main():
    jobs = []
    for maxpos in [2, 3, 4, 5]:
        jobs.append((f"기존5", EX5, maxpos))
        jobs.append((f"+4", P4, maxpos))
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run_one, jobs))

    print("=== max_positions 그리드 (1d 엔진, full $100→) ===\n")
    print(f"{'조합':8} {'maxpos':>6} {'full$':>9} {'Sharpe':>7} {'MDD%':>6}")
    for name, mp, f, sh, mdd in sorted(res, key=lambda x: (x[0], x[1])):
        print(f"{name:8} {mp:>6} {f:>9,.0f} {sh:>7.2f} {mdd:>6.0f}")


if __name__ == "__main__":
    main()
