"""daily DD stop 제거가 robust 개선인지 — v16단독·merged, ddstop on/off, full/H1/H2.
구조적 변경(파라미터 핏 아님). 양쪽 반쪽서 일관 개선이면 채택 후보."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(args):
    cfg, ddstop = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(cfg)))
    ptf = p.get("primary_timeframe", "1h")
    if not ddstop:
        p["risk"]["daily_drawdown_stop"] = -1.0
        p["risk"]["daily_drawdown_pause"] = -1.0
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return (cfg, ddstop), d


def seg(eq, a, b):
    s = eq[(eq.index >= a) & (eq.index < b)]
    s = s/s.iloc[0]
    daily = s.pct_change().fillna(0)
    mdd = ((s/s.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    return s.iloc[-1]*100-100, sh, mdd


def main():
    jobs = [("config/final_v16_slwide.yaml", True), ("config/final_v16_slwide.yaml", False),
            ("config/merged_v16_sleeve.yaml", True), ("config/merged_v16_sleeve.yaml", False)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = {k: d for k, d in ex.map(run, jobs)}
    spans = [("full", "2022-01-01", "2026-04-23"), ("H1", "2022-01-01", "2024-01-01"),
             ("H2", "2024-01-01", "2026-04-23")]
    labels = {("config/final_v16_slwide.yaml", True): "v16 ddstop ON",
              ("config/final_v16_slwide.yaml", False): "v16 ddstop OFF",
              ("config/merged_v16_sleeve.yaml", True): "merged ddstop ON",
              ("config/merged_v16_sleeve.yaml", False): "merged ddstop OFF"}
    print("=== daily DD stop 제거 효과 (수익%/Sharpe/MDD) ===")
    for k in jobs:
        eq = (1+res[k]).cumprod()
        cells = []
        for name, a, b in spans:
            r, sh, mdd = seg(eq, a, b)
            cells.append(f"{name} {r:>+6.0f}%/{sh:.2f}/{mdd:.0f}%")
        print(f"  {labels[k]:18} {' | '.join(cells)}")
    print("\n판정: OFF가 ON 대비 full·H1·H2 MDD↓ 또는 수익↑ 일관이면 구조적 robust 개선.")


if __name__ == "__main__":
    main()
