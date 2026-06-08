"""슬리브 비중 확장 (50~90%) — v16 베이스 / p7 베이스 둘 다."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def bt(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return eng.equity_curve.to_series()


def sharpe(r): return r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0
def mdd(r):
    c = (1+r).cumprod(); return ((c-c.cummax())/c.cummax()).min()*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        e16, ep7, esl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                         ("config/p7_whi.yaml", "1h"),
                                         ("config/sleeve_meanrev.yaml", "1d")]))
    d16 = e16.resample("1D").last().pct_change()
    dp7 = ep7.resample("1D").last().pct_change()
    dsl = esl.resample("1D").last().pct_change()

    for name, base in [("v16", d16), ("p7_whi", dp7)]:
        j = pd.concat([base, dsl], axis=1, keys=["b", "sl"]).dropna()
        print(f"\n=== {name} : 슬리브 배분 (슬리브 비중 ↑) ===")
        print(f"{'베이스:슬리브':>12} | {'총수익%':>12} {'Sharpe':>6} {'MDD%':>6}")
        for bw in [50, 40, 30, 20, 10, 0]:
            x = bw/100
            r = x*j.b + (1-x)*j.sl
            eq = (1+r).cumprod().iloc[-1]
            tag = "(슬리브 단독)" if bw == 0 else ""
            print(f"{bw:>4}:{100-bw:<3}     | {(eq-1)*100:>12,.0f} {sharpe(r):>6.2f} {mdd(r):>6.0f} {tag}")


if __name__ == "__main__":
    main()
