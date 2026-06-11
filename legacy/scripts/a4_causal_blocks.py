"""A4 인과 기반 시간대 차단 선별 — 사전선언 가설만, walk-forward 일관성 검증.
무차단 v16 베이스에 인과 차단셋 적용. full/H1/H2 Sharpe·MDD. 양쪽 일관개선만 채택.
데이터마이닝 아님: 각 셋은 메커니즘 선언 후 테스트(사후 시간선택 금지)."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

PERIODS = {"full": ("2022-01-01", "2026-04-23"),
           "H1": ("2022-01-01", "2024-01-01"), "H2": ("2024-01-01", "2026-04-23")}

# 사전선언 인과 가설 (양 전략 공통 적용)
CANDS = {
    "무차단": None,
    "funding3 {0,8,16}": [0, 8, 16],
    "funding_post {0,1,8,9,16,17}": [0, 1, 8, 9, 16, 17],
    "us_open {13,14}": [13, 14],
    "thin_overnight {22,23,0,1}": [22, 23, 0, 1],
    "funding+us {0,8,13,14,16}": [0, 8, 13, 14, 16],
}


def run(args):
    label, hours, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))  # 이미 무차단
    if hours is not None:
        p["strategy_block_hours"] = {"ema_cross": list(hours), "multi_tf_breakout": list(hours)}
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"), until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0
    mdd = ((eq/eq.cummax()-1)).min()*100
    return (label, per, eq.iloc[-1], sh, mdd)


def main():
    jobs = [(lbl, h, per) for lbl, h in CANDS.items() for per in PERIODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, jobs))
    R = {}
    for lbl, per, f, sh, mdd in res:
        R.setdefault(lbl, {})[per] = (f, sh, mdd)

    print("=== A4 인과 차단셋 (무차단 v16 베이스) ===\n")
    print(f"{'가설':30} {'full Sh':>7} {'fMDD':>6} | {'H1 Sh':>6} {'H1MDD':>6} | {'H2 Sh':>6} {'H2MDD':>6}")
    base = R["무차단"]
    for lbl in CANDS:
        ff, fsh, fmdd = R[lbl]["full"]; h1 = R[lbl]["H1"]; h2 = R[lbl]["H2"]
        mark = ""
        if lbl != "무차단":
            better_both = (h1[1] >= base["H1"][1] and h2[1] >= base["H2"][1])
            mark = " ✅양쪽Sharpe↑" if better_both else ""
        print(f"{lbl:30} {fsh:>7.2f} {fmdd:>6.0f} | {h1[1]:>6.2f} {h1[2]:>6.0f} | {h2[1]:>6.2f} {h2[2]:>6.0f}{mark}")
    print(f"\n무차단 기준: full Sh{base['full'][1]:.2f}/MDD{base['full'][2]:.0f} | "
          f"H1 {base['H1'][1]:.2f}/{base['H1'][2]:.0f} | H2 {base['H2'][1]:.2f}/{base['H2'][2]:.0f}")
    print("채택 기준: H1·H2 양쪽 Sharpe ≥ 무차단 (한쪽만이면 국면의존=기각)")


if __name__ == "__main__":
    main()
