"""A1b 사이징 — funding3 베이스(현 config)에서 rpt 스윕. MDD 감소 여부.
단독 + 슬리브 50:50. 목표: 대출시드 생존 MDD(-40%대 이하)를 어느 rpt에서 얻나."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

BASE_RPT = 0.099
BASE_SSS = 0.2


def run_v16(factor):
    """현 config(funding3) 그대로, rpt·SSS rpt 배율만."""
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    p["risk"]["risk_per_trade"] = BASE_RPT * factor
    p["leverage_tiers"]["SSS"]["risk_per_trade"] = BASE_SSS * factor
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return (factor, d)


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    cagr = (eq.iloc[-1] ** (365/len(daily)) - 1) * 100
    return eq.iloc[-1]*100, sh, mdd, cagr


def main():
    factors = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run_v16, factors))
    res.sort(key=lambda x: -x[0])
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== A1b: funding3 v16 rpt 사이징 ===")
    print(f"{'rpt':>6} | {'단독 $':>10} {'Sh':>5} {'MDD%':>6} {'CAGR%':>6} | "
          f"{'+슬리브50:50 $':>13} {'Sh':>5} {'MDD%':>6} {'CAGR%':>6}")
    for factor, d in res:
        f, sh, mdd, cagr = stats(d)
        j = pd.concat([d, sl], axis=1, keys=["v", "s"]).dropna()
        bf, bsh, bmdd, bcagr = stats(0.5*j.v + 0.5*j.s)
        print(f"{BASE_RPT*factor:>6.3f} | {f:>10,.0f} {sh:>5.2f} {mdd:>6.0f} {cagr:>6.0f} | "
              f"{bf:>13,.0f} {bsh:>5.2f} {bmdd:>6.0f} {bcagr:>6.0f}")


if __name__ == "__main__":
    main()
