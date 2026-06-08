"""A1 사이징 — risk_per_trade 스윕으로 MDD/수익 트레이드오프.
정직한 베이스(무차단 v16)에 rpt 배율 적용. 대출시드 생존 사이징 결정용.
+ 슬리브 50:50 블렌드 MDD도 같이(슬리브가 추가로 잡아주는 양)."""
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
    """무차단 v16, rpt·SSS rpt 배율 적용."""
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    p.pop("strategy_block_hours", None)  # 정직한 베이스
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

    print("=== A1: 무차단 v16 rpt 사이징 (정직한 베이스) ===")
    print(f"{'rpt':>6} {'배율':>5} | {'v16단독 $':>11} {'Sh':>5} {'MDD%':>6} {'CAGR%':>6} | {'+슬리브50:50 MDD%':>16} {'Sh':>5}")
    for factor, d in res:
        f, sh, mdd, cagr = stats(d)
        j = pd.concat([d, sl], axis=1, keys=["v", "s"]).dropna()
        bf, bsh, bmdd, bcagr = stats(0.5*j.v + 0.5*j.s)
        print(f"{BASE_RPT*factor:>6.3f} {factor:>5.2f} | {f:>11,.0f} {sh:>5.2f} {mdd:>6.0f} {cagr:>6.0f} | "
              f"{bmdd:>16.0f} {bsh:>5.2f}")
    print("\n참고: rpt는 %MDD를 거의 선형 지배. 슬리브 블렌드가 추가로 MDD 완충.")


if __name__ == "__main__":
    main()
