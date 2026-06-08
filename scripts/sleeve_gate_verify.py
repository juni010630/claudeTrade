"""게이트 엔진 검증 — trend_gate on/off 슬리브 엔진 백테(실제 체결/플랫비용).
이상치(일별마스크: 단독 Sh~3.0/MDD-15, 50:50 Sh~3.36/MDD-31) 재현 여부.
연도별 + v16 블렌드(저장 시계열 재사용)."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

BASE = "config/sleeve_meanrev.yaml"
THR = 0.22  # q0.60


def run_sleeve(args):
    gate, thr = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    if gate:
        mr = p["strategies"]["mean_reversion"]
        mr["trend_gate_enabled"] = True
        mr["trend_gate_er_period"] = 20
        mr["trend_gate_er_threshold"] = thr
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1d",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return eng.equity_curve.to_series()


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def to_daily(eq):
    d = eq.resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return d


def show(tag, daily):
    f, sh, mdd, yr = stats(daily)
    ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
    print(f"  {tag:30} ${f:>10,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        e_off, e_on = list(ex.map(run_sleeve, [(False, THR), (True, THR)]))
    d_off, d_on = to_daily(e_off), to_daily(e_on)

    print("=== 슬리브 단독 (엔진 실제 체결) ===")
    show("게이트 OFF (always-on)", d_off)
    show(f"게이트 ON (ER>{THR} 플랫)", d_on)
    print("  [이상치 예측: ON 단독 Sh~3.0 MDD~-15%]")

    # v16 블렌드
    v = pd.read_parquet("data/results/dyn_series.parquet")["v16"]
    print("\n=== v16 블렌드 ===")
    for wv, ws in [(0.5, 0.5), (0.75, 0.25)]:
        j = pd.concat([v, d_on], axis=1, keys=["v", "s"]).dropna()
        show(f"{int(wv*100)}:{int(ws*100)} 게이트ON", wv*j.v + ws*j.s)
    jo = pd.concat([v, d_off], axis=1, keys=["v", "s"]).dropna()
    show("50:50 게이트OFF(비교)", 0.5*jo.v + 0.5*jo.s)
    print("  [이상치 예측: 50:50 게이트 Sh~3.36 MDD~-31%]")


if __name__ == "__main__":
    main()
