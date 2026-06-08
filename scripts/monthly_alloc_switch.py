"""2계좌 월간 배분 스위치 — 슬리브 거래 불변, 월간 리밸 시 regime로 v16:슬리브 비중 조절.
추세장(basket ER 높음)→v16쪽, 횡보장(ER 낮음)→슬리브쪽. 슬리브 내부 게이트 아님(딥셀 없음).
정적 50:50 월간 대비. regime신호=전월말 basket ER(20), look-ahead 없음."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def run(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return cfg, d


def er(c, n=20):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def stats(eq):
    daily = eq.pct_change().fillna(0)
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, (eq.iloc[-1]-1)*100, sh, mdd, yr


def monthly_blend(v, s, wv_func):
    """월간 리밸, 월별 target weight = wv_func(month_signal)."""
    months = v.index.to_period("M")
    eq = pd.Series(index=v.index, dtype=float); total = 1.0
    for m in months.unique():
        mask = months == m
        wv = wv_func(m)
        seg = wv*(1+v[mask]).cumprod() + (1-wv)*(1+s[mask]).cumprod()
        eq[mask] = total*seg.values; total = eq[mask].iloc[-1]
    return eq


def show(tag, eq):
    f, tot, sh, mdd, yr = stats(eq)
    ys = " ".join(f"{int(k)}:{round(x):+}" for k, x in yr.items())
    print(f"  {tag:30} ${f:>8,.0f} 총{tot:>+7.0f}% Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        res = dict(ex.map(run, [("config/final_v16_slwide.yaml", "1h"),
                                ("config/sleeve_meanrev.yaml", "1d")]))
    v = res["config/final_v16_slwide.yaml"]
    s = res["config/sleeve_meanrev.yaml"]
    j = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()

    # 월별 regime 신호: 전월말 basket ER(20)
    be = pd.concat([er(load_close(x), 20) for x in SLEEVE], axis=1).mean(axis=1)
    if be.index.tz is not None:
        be.index = be.index.tz_localize(None)
    be = be.reindex(j.v.index, method="ffill")
    month_sig = {}  # period -> 전월말 ER
    months = j.v.index.to_period("M")
    for m in months.unique():
        prev_end = j.v.index[months == m][0] - pd.Timedelta(days=1)
        prior = be[be.index <= prev_end]
        month_sig[m] = prior.iloc[-1] if len(prior) else be.iloc[0]

    print("=== 2계좌 월간 배분 스위치 (funding3 v16 + 슬리브) ===")
    show("정적 50:50 월간", monthly_blend(j.v, j.s, lambda m: 0.5))

    for thr in [0.20, 0.22, 0.25]:
        print(f"\n  --- ER 임계 {thr} (추세장 비중↑v16 / 횡보장 비중↑슬리브) ---")
        # A: 추세 100:0 / 횡보 50:50
        show(f"A 추세100:0/횡보50:50", monthly_blend(j.v, j.s,
             lambda m, t=thr: 1.0 if month_sig[m] > t else 0.5))
        # B: 추세 80:20 / 횡보 50:50
        show(f"B 추세80:20/횡보50:50", monthly_blend(j.v, j.s,
             lambda m, t=thr: 0.8 if month_sig[m] > t else 0.5))
        # C: 추세 70:30 / 횡보 30:70 (양방향 틸트)
        show(f"C 추세70:30/횡보30:70", monthly_blend(j.v, j.s,
             lambda m, t=thr: 0.7 if month_sig[m] > t else 0.3))


if __name__ == "__main__":
    main()
