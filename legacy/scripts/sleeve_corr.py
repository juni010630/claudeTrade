"""v16(추세) + meanrev 슬리브 상관/합산 분석 — 분산 효과 정량."""
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


def sharpe(r):
    return r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0


def mdd(r):
    c = (1+r).cumprod()
    return ((c-c.cummax())/c.cummax()).min()*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        e16, esl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                    ("config/sleeve_meanrev.yaml", "1d")]))
    d16 = e16.resample("1D").last().pct_change().dropna()
    dsl = esl.resample("1D").last().pct_change().dropna()
    j = pd.concat([d16, dsl], axis=1, keys=["v16", "sleeve"]).dropna()
    corr = j.v16.corr(j.sleeve)
    comb = 0.5*j.v16 + 0.5*j.sleeve
    print(f"일별 상관(v16 vs 슬리브): {corr:+.3f}")
    print(f"v16    단독: Sharpe {sharpe(j.v16):.2f}  MDD {mdd(j.v16):.0f}%")
    print(f"슬리브 단독: Sharpe {sharpe(j.sleeve):.2f}  MDD {mdd(j.sleeve):.0f}%")
    print(f"50:50 합산: Sharpe {sharpe(comb):.2f}  MDD {mdd(comb):.0f}%  ← 분산효과")
    # 연도별 상관
    j["year"] = j.index.year.clip(upper=2025)
    print("\n연도별 상관:")
    for y in [2022, 2023, 2024, 2025]:
        sub = j[j.year == y]
        if len(sub) > 5:
            print(f"  {y}: {sub.v16.corr(sub.sleeve):+.3f}")


if __name__ == "__main__":
    main()
