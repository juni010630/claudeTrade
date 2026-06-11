"""기존5 슬리브 max_positions 2/3/5 연도별 검증 — maxpos 축소가 단일연 아티팩트 아닌지."""
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
PERIODS = {"full": ("2022-01-01", "2026-04-23"), "2022": ("2022-01-01", "2023-01-01"),
           "2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
           "2025": ("2025-01-01", "2026-04-23")}


def run_one(args):
    maxpos, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    p["risk"]["max_positions"] = maxpos
    p["risk"]["max_same_direction"] = maxpos
    loader = DataLoader(symbols=EX5, timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"), until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0.0
    return (maxpos, per, eq.iloc[-1], sh, mdd)


def main():
    jobs = [(mp, per) for mp in [2, 3, 5] for per in PERIODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run_one, jobs))
    R = {}
    for mp, per, f, sh, mdd in res:
        R.setdefault(mp, {})[per] = (f, sh, mdd)
    print("=== 기존5 maxpos 연도별 ($100→, Sharpe, MDD%) ===\n")
    for mp in [2, 3, 5]:
        print(f"--- maxpos={mp} ---")
        for per in PERIODS:
            f, sh, mdd = R[mp][per]
            print(f"  {per:5} ${f:>7,.0f}  Sh {sh:>5.2f}  MDD {mdd:>5.0f}%")


if __name__ == "__main__":
    main()
