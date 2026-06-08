"""동적 배분 스위치 — OFF(강추세)=v16 단독(100:0), ON(횡보)=50:50.
regime=basket ER(20).shift1. 임계값 스윕 + 연도별 + baseline 대비.
일별 시계열 저장(data/results/dyn_series.parquet)→이후 임계조정 즉시."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
SER_PATH = "data/results/dyn_series.parquet"


def er(c, n=20):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def run_engine(args):
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


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def get_series():
    if Path(SER_PATH).exists():
        df = pd.read_parquet(SER_PATH)
        return df["v16"], df["sl"]
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        ev16, esl = list(ex.map(run_engine, [("config/final_v16_slwide.yaml", "1h"),
                                             ("config/sleeve_meanrev.yaml", "1d")]))
    dv16 = ev16.resample("1D").last().pct_change().fillna(0); dv16.index = dv16.index.tz_localize(None)
    dsl = esl.resample("1D").last().pct_change().fillna(0); dsl.index = dsl.index.tz_localize(None)
    df = pd.concat([dv16, dsl], axis=1, keys=["v16", "sl"]).dropna()
    Path("data/results").mkdir(exist_ok=True)
    df.to_parquet(SER_PATH)
    return df["v16"], df["sl"]


def main():
    dv16, dsl = get_series()
    j = pd.concat([dv16, dsl], axis=1, keys=["v", "s"]).dropna()
    basket_er = pd.concat([er(load_close(s), 20) for s in SLEEVE], axis=1).mean(axis=1).shift(1)
    if basket_er.index.tz is not None:
        basket_er.index = basket_er.index.tz_localize(None)
    er_j = basket_er.reindex(j.index, method="ffill")

    def show(tag, r):
        f, sh, mdd, yr = stats(r)
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
        print(f"  {tag:28} ${f:>10,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")

    print("=== baseline ===")
    show("v16 단독(100:0)", j.v)
    show("항상 75:25", 0.75*j.v + 0.25*j.s)
    show("항상 50:50", 0.50*j.v + 0.50*j.s)

    print("\n=== 동적: OFF=v16단독, ON=50:50 (ER<=thr면 ON) ===")
    for q in [0.50, 0.60, 0.67, 0.75, 0.80, 0.90]:
        thr = er_j.quantile(q)
        on = er_j <= thr
        r = np.where(on, 0.50*j.v + 0.50*j.s, j.v)
        r = pd.Series(r, index=j.index)
        on_pct = on.mean()*100
        show(f"q{q:.2f}(ER<{thr:.2f}) ON {on_pct:.0f}%일", r)

    print("\n=== 참고: 동적 OFF=v16, ON=70:30 ===")
    for q in [0.60, 0.67, 0.75]:
        thr = er_j.quantile(q)
        on = er_j <= thr
        r = pd.Series(np.where(on, 0.70*j.v + 0.30*j.s, j.v), index=j.index)
        show(f"q{q:.2f} ON", r)


if __name__ == "__main__":
    main()
