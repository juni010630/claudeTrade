"""v16 MDD 분해 — 봉기준 vs 일별, 분기별 낙폭, 슬리브 합산."""
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


def mdd_of(eq):
    return ((eq - eq.cummax())/eq.cummax()).min()*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        e16, esl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                    ("config/sleeve_meanrev.yaml", "1d")]))

    # v16: 봉기준 MDD vs 일별 MDD
    e16d = e16.resample("1D").last().ffill()
    print("=== v16 MDD: 측정단위 차이 ===")
    print(f"봉(1h)기준 MDD: {mdd_of(e16):.1f}%   ← 실제 계좌가 겪는 일중 최저점 포함")
    print(f"일별 종가 MDD: {mdd_of(e16d):.1f}%   ← 일중 회복분 안 보임(과소)")

    # 분기별 낙폭 (각 분기 내 최대 낙폭, 봉기준)
    print("\n=== v16 분기별 최대낙폭(봉기준) + 분기수익 ===")
    print(f"{'분기':8} {'분기수익%':>9} {'분기내MDD%':>10}")
    for (y, q), seg in e16.groupby([e16.index.year, e16.index.quarter]):
        if len(seg) < 2: continue
        qret = (seg.iloc[-1]/seg.iloc[0]-1)*100
        qmdd = mdd_of(seg)
        print(f"{y}Q{q:<5} {qret:>9,.0f} {qmdd:>10.1f}")

    # 슬리브 합산 (일별 기준 — 1d 슬리브라 일별이 공정)
    d16 = e16.resample("1D").last().pct_change()
    dsl = esl.resample("1D").last().pct_change()
    j = pd.concat([d16, dsl], axis=1, keys=["v16", "sl"]).dropna()
    print("\n=== 일별 기준: v16 단독 vs 슬리브합산 (공정 비교, 같은 단위) ===")
    for w, lab in [(1.0, "v16 단독"), (0.6, "60:40"), (0.5, "50:50")]:
        r = w*j.v16 + (1-w)*j.sl
        eq = (1+r).cumprod()
        sh = r.mean()/r.std()*np.sqrt(365)
        print(f"{lab:10}: Sharpe {sh:.2f}  일별MDD {mdd_of(eq):.1f}%")


if __name__ == "__main__":
    main()
