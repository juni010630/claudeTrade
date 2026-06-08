"""A2 보너스 제거 — v16 원본 vs 무보너스, 단독 & 슬리브 50:50. MDD 감소 여부 핵심.
보너스=레버리지 가설: Sharpe 거의 불변 + 자산↓ + MDD↓면 MDD민감 계좌엔 제거가 정답."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run_v16(nobonus):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    if nobonus:
        p.pop("strategy_size_bonus", None)
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return d


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def show(tag, daily):
    f, sh, mdd, yr = stats(daily)
    ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
    print(f"  {tag:28} ${f:>12,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


def main():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        d_on, d_off = list(ex.map(run_v16, [False, True]))
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== v16 단독 ===")
    show("보너스 有 (현재)", d_on)
    show("보너스 제거", d_off)

    print("\n=== 슬리브 50:50 블렌드 ===")
    jon = pd.concat([d_on, sl], axis=1, keys=["v", "s"]).dropna()
    joff = pd.concat([d_off, sl], axis=1, keys=["v", "s"]).dropna()
    show("보너스有 + 슬리브 50:50", 0.5*jon.v + 0.5*jon.s)
    show("보너스제거 + 슬리브 50:50", 0.5*joff.v + 0.5*joff.s)


if __name__ == "__main__":
    main()
