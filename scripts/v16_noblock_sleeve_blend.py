"""v16(시간대 차단 전부 제거) + 슬리브 50:50 블렌드.
정직한(과적합 부스트 제거) v16을 슬리브로 분산. 원본v16 블렌드와 비교. 연도별."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def run_v16(noblock):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    if noblock:
        p.pop("strategy_block_hours", None)
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
    print(f"  {tag:32} ${f:>11,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


def main():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        d_nb, d_orig = list(ex.map(run_v16, [True, False]))
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== v16 단독 ===")
    show("v16 원본(차단 有)", d_orig)
    show("v16 무차단(차단 全제거)", d_nb)

    print("\n=== 슬리브 50:50 블렌드 ===")
    jo = pd.concat([d_orig, sl], axis=1, keys=["v", "s"]).dropna()
    jn = pd.concat([d_nb, sl], axis=1, keys=["v", "s"]).dropna()
    show("원본v16 + 슬리브 50:50", 0.5*jo.v + 0.5*jo.s)
    show("무차단v16 + 슬리브 50:50", 0.5*jn.v + 0.5*jn.s)
    print("\n  (참고) 슬리브 단독")
    show("슬리브", sl)


if __name__ == "__main__":
    main()
