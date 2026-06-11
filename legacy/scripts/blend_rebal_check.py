"""블렌드 리밸런싱 빈도 검증 — rpt 상방 이득이 매일리밸런싱(변동성하베스팅) 산물인지.
v16(rpt 0.099 vs 0.247) × 슬리브, 매일/월간/무리밸런싱 블렌드 비교.
무리밸런싱서 rpt 상방 이득 사라지면 = 계산 가정 산물(현실 아님)."""
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


def mstat(eq):
    mdd = ((eq/eq.cummax()-1)).min()*100
    r = eq.pct_change().fillna(0)
    sh = r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0
    return eq.iloc[-1]*100, (eq.iloc[-1]-1)*100, sh, mdd


def blend_daily(v, s):  # 매일 리밸런싱
    r = 0.5*v + 0.5*s
    return (1+r).cumprod()


def blend_buyhold(v, s):  # 무리밸런싱 (50:50 후 방치)
    return 0.5*(1+v).cumprod() + 0.5*(1+s).cumprod()


def blend_monthly(v, s):  # 월초 리밸런싱
    ev, es = (1+v).cumprod(), (1+s).cumprod()
    idx = v.index
    eq = pd.Series(1.0, index=idx)
    cur = 1.0
    last_m = None
    wv = ws = 0.5
    base_v = base_s = 1.0
    pv = ps = 1.0
    for t in idx:
        # 일별 성장 적용
        gv = (1+v.loc[t]); gs = (1+s.loc[t])
        pv *= gv; ps *= gs
        val = cur * (wv*pv + ws*ps) if False else None
    # 단순화: 월간은 resample 방식으로
    rv = v.copy(); rs = s.copy()
    months = idx.to_period("M")
    eqv = pd.Series(index=idx, dtype=float)
    total = 1.0
    for m in months.unique():
        mask = months == m
        seg_v = (1+rv[mask]).cumprod()
        seg_s = (1+rs[mask]).cumprod()
        seg = 0.5*seg_v + 0.5*seg_s  # 월초 50:50 리밸런싱 후 월내 방치
        eqv[mask] = total * seg.values
        total = eqv[mask].iloc[-1]
    return eqv


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        res = dict(ex.map(run_v16, [1.0, 2.5]))
    v_lo, v_hi = res[1.0], res[2.5]  # rpt 0.099, 0.247
    sl = pd.read_parquet("data/results/dyn_series.parquet")["sl"]

    print("=== 블렌드 리밸런싱 빈도별 — rpt 0.099 vs 0.247 ===\n")
    for label, vser in [("rpt 0.099", v_lo), ("rpt 0.247", v_hi)]:
        j = pd.concat([vser, sl], axis=1, keys=["v", "s"]).dropna()
        print(f"[{label}]")
        for rname, fn in [("매일리밸", blend_daily), ("월간리밸", blend_monthly), ("무리밸(방치)", blend_buyhold)]:
            eq = fn(j.v, j.s)
            f, tot, sh, mdd = mstat(eq)
            print(f"  {rname:12} 최종${f:>8,.0f} 총수익{tot:>+8.0f}% Sh{sh:.2f} MDD{mdd:>4.0f}%")
        print()
    print("판정: 매일→무리밸로 갈수록 rpt 0.099 vs 0.247 격차가 줄면 = 상방이득은 변동성하베스팅 산물.")


if __name__ == "__main__":
    main()
