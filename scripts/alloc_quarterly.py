"""v16:슬리브 배분(90:10~50:50) 분기별 수익률 — 출력 + CSV 저장."""
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


def qret(r):
    return ((1+r).prod()-1)*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        e16, esl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                    ("config/sleeve_meanrev.yaml", "1d")]))
    d16 = e16.resample("1D").last().pct_change()
    dsl = esl.resample("1D").last().pct_change()
    j = pd.concat([d16, dsl], axis=1, keys=["v16", "sl"]).dropna()

    allocs = [(90, 10), (80, 20), (70, 30), (60, 40), (50, 50)]
    out = {}
    for a, b in allocs:
        r = (a/100)*j.v16 + (b/100)*j.sl
        out[f"{a}:{b}"] = r.resample("QE").apply(qret).round(1)
    q = pd.DataFrame(out)
    q.index = [f"{t.year}Q{t.quarter}" for t in q.index]

    print("=== v16:슬리브 배분별 분기 수익률(%) ===")
    print(q.to_string())
    # 전체 요약
    print("\n=== 전체 (참고) ===")
    for a, b in allocs:
        r = (a/100)*j.v16 + (b/100)*j.sl
        eq = (1+r).cumprod()
        mdd = ((eq-eq.cummax())/eq.cummax()).min()*100
        sh = r.mean()/r.std()*np.sqrt(365)
        print(f"{a}:{b}  최종 ${eq.iloc[-1]*100:,.0f}  Sharpe {sh:.2f}  일별MDD {mdd:.0f}%")

    Path("data/results").mkdir(exist_ok=True)
    q.to_csv("data/results/alloc_quarterly.csv")
    print("\n저장: data/results/alloc_quarterly.csv")


if __name__ == "__main__":
    main()
