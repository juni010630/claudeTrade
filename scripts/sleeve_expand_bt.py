"""슬리브 심볼 확장 엔진 백테 — 기존5 vs 확장조합, 전체+연도별 병렬.
판정: 합산 Sharpe/MDD가 연도별로 개선되는가 (경로카오스/과최적 가드)."""
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

SETS = {
    "기존5":      ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"],
    "+MTL":       ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT", "MTLUSDT"],
    "+MTL_SNX":   ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT", "MTLUSDT", "SNXUSDT"],
    "+4(M/B/S/O)": ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT",
                    "MTLUSDT", "BELUSDT", "SNXUSDT", "ONTUSDT"],
    "+6(전후보)":  ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT",
                    "MTLUSDT", "BELUSDT", "SNXUSDT", "ONTUSDT", "RSRUSDT", "ROSEUSDT"],
}

PERIODS = {
    "full": ("2022-01-01", "2026-04-23"),
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-04-23"),
}


def run_one(args):
    name, syms, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    p["symbols"] = syms
    p["strategies"]["mean_reversion"]["symbols"] = syms
    loader = DataLoader(symbols=syms, timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"),
                           until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0.0
    final = eq.iloc[-1]
    ntr = len(eng.ledger.records) if hasattr(eng, "ledger") else 0
    return (name, per, final, sh, mdd, ntr)


def main():
    jobs = [(n, s, per) for n, s in SETS.items() for per in PERIODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run_one, jobs))

    R = {}
    for name, per, final, sh, mdd, ntr in res:
        R.setdefault(name, {})[per] = (final, sh, mdd, ntr)

    print("=== 슬리브 확장 비교 (1d 엔진, $100 시작) ===\n")
    print(f"{'조합':14} {'full$':>9} {'Sharpe':>7} {'MDD%':>6} {'거래':>4} | "
          f"{'22$':>6} {'23$':>6} {'24$':>6} {'25$':>6}")
    for name in SETS:
        f, sh, mdd, n = R[name]["full"]
        y = {p: R[name][p][0] for p in ["2022", "2023", "2024", "2025"]}
        print(f"{name:14} {f:>9,.0f} {sh:>7.2f} {mdd:>6.0f} {n:>4} | "
              f"{y['2022']:>6,.0f} {y['2023']:>6,.0f} {y['2024']:>6,.0f} {y['2025']:>6,.0f}")


if __name__ == "__main__":
    main()
