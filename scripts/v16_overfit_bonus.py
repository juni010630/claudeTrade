"""과적합 진단 — v16 보너스 시간대(size_bonus 1.25x) 강건성.
원본 vs 무보너스 vs ±시프트 vs 동일개수 랜덤셋. 시간대차단 테스트와 동일 논리.
무너지면=정확시간 핏(과적합). 전체+연도별 병렬."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

BASE = "config/final_v16_slwide.yaml"
PERIODS = {"full": ("2022-01-01", "2026-04-23"), "2022": ("2022-01-01", "2023-01-01"),
           "2023": ("2023-01-01", "2024-01-01"), "2024": ("2024-01-01", "2025-01-01"),
           "2025": ("2025-01-01", "2026-04-23")}


def shift_b(bonus, k):
    return {s: sorted({(h + k) % 24 for h in hrs}) for s, hrs in bonus.items()}


def random_b(bonus, seed):
    rng = np.random.default_rng(seed)
    return {s: sorted(rng.choice(24, size=len(hrs), replace=False).tolist())
            for s, hrs in bonus.items()}


def run(args):
    label, bonus, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    if bonus is None:
        p.pop("strategy_size_bonus", None)
    else:
        p["strategy_size_bonus"] = bonus
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"), until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0
    return (label, per, eq.iloc[-1], sh)


def main():
    base = yaml.safe_load(open(BASE))
    orig = base["strategy_size_bonus"]
    variants = {"원본": orig, "무보너스": None}
    for k in [1, 2, 3, 6, 12]:
        variants[f"시프트+{k}h"] = shift_b(orig, k)
    for sd in [1, 2, 3]:
        variants[f"랜덤{sd}"] = random_b(orig, sd)

    jobs = [(lbl, b, per) for lbl, b in variants.items() for per in PERIODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, jobs))
    R = {}
    for lbl, per, f, sh in res:
        R.setdefault(lbl, {})[per] = (f, sh)

    print("=== v16 보너스 시간대 강건성 (전체+연도별, $100→ / Sharpe) ===\n")
    print(f"{'변형':10} {'full$':>12} {'fullSh':>6} | {'22':>7} {'23':>7} {'24':>7} {'25':>8}")
    for lbl in variants:
        f, sh = R[lbl]["full"]
        ys = {p: R[lbl][p][0] for p in ["2022", "2023", "2024", "2025"]}
        print(f"{lbl:10} {f:>12,.0f} {sh:>6.2f} | {ys['2022']:>7,.0f} {ys['2023']:>7,.0f} "
              f"{ys['2024']:>7,.0f} {ys['2025']:>8,.0f}")


if __name__ == "__main__":
    main()
