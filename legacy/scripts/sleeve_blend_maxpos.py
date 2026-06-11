"""v16 + 슬리브(maxpos 2/3/5) 합산 검증 — maxpos 축소가 실제 배포(블렌드)에서 개선되나."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

EX5 = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def run_sleeve(maxpos):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/sleeve_meanrev.yaml")))
    p["risk"]["max_positions"] = maxpos
    p["risk"]["max_same_direction"] = maxpos
    loader = DataLoader(symbols=EX5, timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return ("sl", maxpos, eng.equity_curve.to_series())


def run_v16(_):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open("config/final_v16_slwide.yaml"))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return ("v16", 0, eng.equity_curve.to_series())


def stats(r):
    eq = (1+r).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = r.mean()/r.std()*np.sqrt(365)
    return eq.iloc[-1]*100, sh, mdd


def dispatch(args):
    kind, mp = args
    return run_v16(0) if kind == "v16" else run_sleeve(mp)


def main():
    jobs = [("v16", 0)] + [("sl", mp) for mp in [2, 3, 5]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(dispatch, jobs))
    curves = {}
    for kind, mp, eq in res:
        curves[(kind, mp)] = eq.resample("1D").last().pct_change()
    dv = curves[("v16", 0)]

    print("=== v16 + 슬리브 블렌드, 슬리브 maxpos별 (일별) ===\n")
    for ratio in [(75, 25), (50, 50)]:
        a, b = ratio
        print(f"--- {a}:{b} (v16:슬리브) ---")
        for mp in [2, 3, 5]:
            j = pd.concat([dv, curves[("sl", mp)]], axis=1, keys=["v", "s"]).dropna()
            r = (a/100)*j.v + (b/100)*j.s
            f, sh, mdd = stats(r)
            tag = " ← 현재" if mp == 5 else ""
            print(f"  슬리브maxpos={mp}:  ${f:>9,.0f}  Sharpe {sh:.2f}  MDD {mdd:.0f}%{tag}")
    # 순수 v16 참고
    f, sh, mdd = stats(dv.dropna())
    print(f"\n참고 v16 단독: ${f:,.0f}  Sharpe {sh:.2f}  MDD {mdd:.0f}%")


if __name__ == "__main__":
    main()
