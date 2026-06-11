"""A1c 사이징 상방 — rpt 키웠을 때 MDD 정체 여부 + 수익. funding3 베이스.
가설: sf×lev 캡이 rpt를 대신 묶으면 rpt↑해도 MDD 정체 → 공짜 수익. 어디까지 -50%대 유지되나."""
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
    tot = (eq.iloc[-1] - 1) * 100
    return eq.iloc[-1]*100, tot, sh, mdd, cagr


def main():
    factors = [1.0, 1.3, 1.5, 2.0, 2.5, 3.0]
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(run_v16, factors))
    res.sort(key=lambda x: x[0])
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== A1c: funding3 rpt 상방 스윕 (수익률 + MDD 정체 확인) ===\n")
    print("[v16 단독]")
    print(f"{'rpt':>6} | {'최종$':>13} {'총수익%':>10} {'CAGR%':>7} {'Sh':>5} {'MDD%':>6}")
    for factor, d in res:
        f, tot, sh, mdd, cagr = stats(d)
        print(f"{BASE_RPT*factor:>6.3f} | {f:>13,.0f} {tot:>+10.0f} {cagr:>7.0f} {sh:>5.2f} {mdd:>6.0f}")

    print("\n[슬리브 50:50 블렌드]")
    print(f"{'rpt':>6} | {'최종$':>13} {'총수익%':>10} {'CAGR%':>7} {'Sh':>5} {'MDD%':>6}")
    for factor, d in res:
        j = pd.concat([d, sl], axis=1, keys=["v", "s"]).dropna()
        f, tot, sh, mdd, cagr = stats(0.5*j.v + 0.5*j.s)
        print(f"{BASE_RPT*factor:>6.3f} | {f:>13,.0f} {tot:>+10.0f} {cagr:>7.0f} {sh:>5.2f} {mdd:>6.0f}")


if __name__ == "__main__":
    main()
