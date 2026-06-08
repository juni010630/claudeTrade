"""B1 거래량 필터 walk-forward — min_volume_ratio 임계 스윕 + OOS 검증.
무차단 v16(ema_cross에 거래량필터). full + 양방향 walk-forward(반쪽 IS최적→반쪽 OOS).
측정: Sharpe + MDD (구조적 MDD 줄이는지도). 과적합 가드: OOS가 무필터 이기는지."""
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


def run(args):
    thr, per = args
    s, e = PERIODS[per]
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    p.pop("strategy_block_hours", None)  # 정직한 베이스
    if thr is not None:
        p["strategies"]["ema_cross"]["min_volume_ratio"] = thr
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(s, tz="UTC"), until=pd.Timestamp(e, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0
    mdd = ((eq/eq.cummax()-1)).min()*100
    return (thr, per, eq.iloc[-1], sh, mdd)


def main():
    thrs = [None, 0.7, 0.9, 1.0, 1.2, 1.5]
    jobs = [(t, per) for t in thrs for per in PERIODS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, jobs))
    R = {}
    for thr, per, f, sh, mdd in res:
        R.setdefault(thr, {})[per] = (f, sh, mdd)

    print("=== B1 거래량필터(min_volume_ratio) — 무차단 v16, ema_cross ===\n")
    print(f"{'임계':>6} | {'full $':>10} {'Sh':>5} {'MDD%':>6} | {'H1 Sh':>6} {'H1MDD':>6} | {'H2 Sh':>6} {'H2MDD':>6}")
    for t in thrs:
        lbl = "무필터" if t is None else f"{t:.1f}"
        ff, fsh, fmdd = R[t]["full"]
        h1 = R[t]["H1"]; h2 = R[t]["H2"]
        print(f"{lbl:>6} | {ff:>10,.0f} {fsh:>5.2f} {fmdd:>6.0f} | {h1[1]:>6.2f} {h1[2]:>6.0f} | {h2[1]:>6.2f} {h2[2]:>6.0f}")

    print("\n판정: 각 반쪽(H1/H2)에서 무필터 대비 Sharpe↑·MDD↓가 양쪽 다 나오면 robust(과적합 아님).")
    base = R[None]
    print(f"  무필터 기준: full Sh{base['full'][1]:.2f}/MDD{base['full'][2]:.0f}% | "
          f"H1 Sh{base['H1'][1]:.2f}/MDD{base['H1'][2]:.0f}% | H2 Sh{base['H2'][1]:.2f}/MDD{base['H2'][2]:.0f}%")


if __name__ == "__main__":
    main()
