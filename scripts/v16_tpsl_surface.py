"""과적합 진단 — TP/SL plateau vs spike.
전략별 2D 격자(TP×SL) 표면 + 선택값 행/열의 연도별. 다른전략 고정.
판정: 선택점이 완만한 고원이면 견고, 외딴 봉우리면 과적합. 경로카오스→모양으로 판단."""
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

EMA_TP = [2.5, 3.0, 3.5, 4.0, 4.5]
EMA_SL = [1.4, 1.6, 1.8, 2.0, 2.2]
MTF_TP = [3.0, 3.5, 4.0, 4.5, 5.0]
MTF_SL = [1.7, 1.9, 2.1, 2.3, 2.5]


def run(args):
    strat, tp, sl, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    p["strategies"][strat]["atr_tp_mult"] = tp
    p["strategies"][strat]["atr_sl_mult"] = sl
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"), until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0
    return (strat, tp, sl, per, eq.iloc[-1], sh)


def surface(res, strat, TPs, SLs, sel_tp, sel_sl):
    R = {(tp, sl): v for (st, tp, sl, per, f, sh) in res if st == strat and per == "full" for v in [(f, sh)]}
    print(f"\n=== {strat} full Sharpe 표면 (행=SL, 열=TP; ★=선택 {sel_tp}/{sel_sl}) ===")
    print("SL\\TP  " + "".join(f"{tp:>7.1f}" for tp in TPs))
    for sl in SLs:
        row = f"{sl:>5.1f}  "
        for tp in TPs:
            sh = R[(tp, sl)][1]
            mark = "★" if (tp == sel_tp and sl == sel_sl) else " "
            row += f"{sh:>6.2f}{mark}"
        print(row)
    print(f"--- {strat} full 최종자산$ 표면 ---")
    print("SL\\TP  " + "".join(f"{tp:>8.1f}" for tp in TPs))
    for sl in SLs:
        row = f"{sl:>5.1f}  "
        for tp in TPs:
            f = R[(tp, sl)][0]
            row += f"{f:>8,.0f}"
        print(row)


def yearly_at(res, strat, sel_tp, sel_sl, TPs):
    """선택 SL 고정, TP 행의 연도별 Sharpe (TP 견고성)."""
    print(f"\n--- {strat} SL={sel_sl} 고정, TP별 연도 Sharpe ---")
    print(f"{'TP':>5} " + "".join(f"{p:>8}" for p in ["full", "2022", "2023", "2024", "2025"]))
    for tp in TPs:
        cells = {per: sh for (st, t, sl, per, f, sh) in res if st == strat and t == tp and sl == sel_sl}
        mark = "★" if tp == sel_tp else " "
        print(f"{tp:>4.1f}{mark}" + "".join(f"{cells.get(p, 0):>8.2f}" for p in ["full", "2022", "2023", "2024", "2025"]))


def main():
    jobs = []
    # full 표면: 전 셀 / 연도별: 선택 SL 행(TP 견고성)만
    for tp in EMA_TP:
        for sl in EMA_SL:
            jobs.append(("ema_cross", tp, sl, "full"))
            if sl == 1.8:
                for per in ["2022", "2023", "2024", "2025"]:
                    jobs.append(("ema_cross", tp, sl, per))
    for tp in MTF_TP:
        for sl in MTF_SL:
            jobs.append(("multi_tf_breakout", tp, sl, "full"))
            if sl == 2.1:
                for per in ["2022", "2023", "2024", "2025"]:
                    jobs.append(("multi_tf_breakout", tp, sl, per))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, jobs))

    surface(res, "ema_cross", EMA_TP, EMA_SL, 3.5, 1.8)
    yearly_at(res, "ema_cross", 3.5, 1.8, EMA_TP)
    surface(res, "multi_tf_breakout", MTF_TP, MTF_SL, 4.0, 2.1)
    yearly_at(res, "multi_tf_breakout", 4.0, 2.1, MTF_TP)


if __name__ == "__main__":
    main()
