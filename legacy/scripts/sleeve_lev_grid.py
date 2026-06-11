"""평균회귀 슬리브 — size_fraction 정밀 그리드 (cross margin), 연도별 + plateau."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

# (name, leverage, size_fraction)  — 전부 cross margin
VARIANTS = [
    ("lev3 sf0.06", 3, 0.06),
    ("lev3 sf0.08", 3, 0.08),
    ("lev3 sf0.10", 3, 0.10),
    ("lev3 sf0.12", 3, 0.12),
    ("lev3 sf0.15", 3, 0.15),
    ("lev2 sf0.15", 2, 0.15),
    ("lev5 sf0.06", 5, 0.06),
    ("lev4 sf0.08", 4, 0.08),
]


def run(v):
    name, lev, sf = v
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open("config/sleeve_meanrev.yaml"))
    for t in ["SSS", "SS", "S", "A", "B", "C"]:
        if t in p.get("leverage_tiers", {}):
            p["leverage_tiers"][t]["leverage"] = lev
            p["leverage_tiers"][t]["size_fraction"] = sf
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100, isolated_margin=False)
    r = eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                               until=pd.Timestamp("2026-04-23", tz="UTC")))
    df = eng.ledger.to_dataframe()
    fs = (df.exit_reason == "forced_stop").mean()*100 if len(df) else 0
    df["year"] = pd.to_datetime(df.entry_time).dt.year.clip(upper=2025)
    yr = df.groupby("year").pnl.sum()
    ys = {y: round(yr.get(y, 0)) for y in [2022, 2023, 2024, 2025]}
    py = sum(1 for y in [2022, 2023, 2024, 2025] if yr.get(y, 0) > 0)
    return {"name": name, "eq": round(r.final_equity), "sharpe": round(r.sharpe, 2),
            "mdd": round(r.max_drawdown, 1), "fs": round(fs), "py": py, "ys": ys}


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, VARIANTS))
    print(f"{'변형':14} | {'최종$':>7} {'Sharpe':>6} {'MDD%':>6} {'forced':>6} {'양수년':>5}  연도별($100→)")
    for r in res:
        y = r["ys"]
        print(f"{r['name']:14} | {r['eq']:7} {r['sharpe']:6.2f} {r['mdd']:6.1f} {r['fs']:5}% "
              f"{r['py']:4}/4  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
