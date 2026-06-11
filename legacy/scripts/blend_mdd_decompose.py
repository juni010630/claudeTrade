"""블렌드 MDD 분해 — "-50%가 v16 절반 증발인가 슬리브 분산인가".
비교: v16단독 / 슬리브단독 / 50%v16+50%현금(분산0) / 50:50블렌드.
+ 블렌드 바닥 시점에 v16·슬리브 각각의 드로다운 분해."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run_v16():
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open("config/final_v16_slwide.yaml"))  # funding3, rpt 0.099
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return d


def dd_series(eq):
    return eq/eq.cummax() - 1


def mdd(eq):
    return dd_series(eq).min()*100


def main():
    v = run_v16()
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]
    j = pd.concat([v, sl], axis=1, keys=["v", "s"]).dropna()

    ev = (1+j.v).cumprod()
    es = (1+j.s).cumprod()
    e_cash = (1 + 0.5*j.v).cumprod()           # 50% v16 + 50% 현금 (매일리밸)
    e_blend = (1 + 0.5*j.v + 0.5*j.s).cumprod()  # 50:50 (매일리밸)

    print("=== MDD 분해 (매일 리밸런싱) ===")
    print(f"  v16 단독                : {mdd(ev):>5.0f}%")
    print(f"  슬리브 단독              : {mdd(es):>5.0f}%")
    print(f"  50% v16 + 50% 현금(분산0) : {mdd(e_cash):>5.0f}%   ← 그냥 절반만 베팅")
    print(f"  50% v16 + 50% 슬리브     : {mdd(e_blend):>5.0f}%   ← 슬리브가 현금보다 낮추면 진짜 분산")
    print(f"\n  슬리브의 순수 분산 효과 = (v16+현금) - (v16+슬리브) = "
          f"{mdd(e_cash) - mdd(e_blend):+.0f}pp")

    # 블렌드 바닥 시점 분해
    ddb = dd_series(e_blend)
    trough = ddb.idxmin()
    # 그 시점 각 자산의 (자기 peak 대비) 드로다운
    print(f"\n=== 블렌드 최저점({trough.date()}) 분해 ===")
    print(f"  그날 v16 드로다운(자기peak대비)   : {dd_series(ev).loc[trough]*100:>5.0f}%")
    print(f"  그날 슬리브 드로다운(자기peak대비) : {dd_series(es).loc[trough]*100:>5.0f}%")
    print(f"  → v16이 깊고 슬리브가 얕으면 = '주로 v16 증발'")


if __name__ == "__main__":
    main()
