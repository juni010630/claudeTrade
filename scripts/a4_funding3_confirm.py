"""funding3 {0,8,16} 확정 검증 — 무차단 vs funding3, 연도별 + 슬리브 50:50 블렌드."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run(hours):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))  # 무차단
    if hours is not None:
        p["strategy_block_hours"] = {"ema_cross": list(hours), "multi_tf_breakout": list(hours)}
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
    print(f"  {tag:26} ${f:>11,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


def main():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        d_nb, d_f3 = list(ex.map(run, [None, (0, 8, 16)]))
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== v16 단독 ===")
    show("무차단", d_nb)
    show("funding3 {0,8,16}", d_f3)
    print("\n=== 슬리브 50:50 블렌드 ===")
    jn = pd.concat([d_nb, sl], axis=1, keys=["v", "s"]).dropna()
    jf = pd.concat([d_f3, sl], axis=1, keys=["v", "s"]).dropna()
    show("무차단 + 슬리브", 0.5*jn.v + 0.5*jn.s)
    show("funding3 + 슬리브", 0.5*jf.v + 0.5*jf.s)


if __name__ == "__main__":
    main()
