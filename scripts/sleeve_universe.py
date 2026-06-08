"""슬리브 심볼 풀 확장 — 5 vs 8심볼 분산 효과 (합산 equity MDD/Sharpe)."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SLIP = {"LTCUSDT": 13, "UNIUSDT": 14, "STORJUSDT": 36, "ARPAUSDT": 40, "BANDUSDT": 56,
        "MTLUSDT": 60, "SNXUSDT": 46, "ONTUSDT": 60}
POOLS = {
    "5심볼(현행)": ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"],
    "8심볼(+MTL/SNX/ONT)": ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT",
                            "MTLUSDT", "SNXUSDT", "ONTUSDT"],
}


def rsi(c, n=14):
    dd = c.diff(); up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-100/(1+up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def sym_trades(s, cost):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"]); d = d.set_index("timestamp").sort_index()
    d = d[d.index >= "2022-01-01"]
    d["rsi"] = rsi(d["close"], 14); d["atr"] = atr(d, 14)
    arr = d.reset_index(); out = []; pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        if pos is None:
            pr, pa = arr.iloc[i-1]["rsi"], arr.iloc[i-1]["atr"]; e = row["open"]
            if pr < 30 and pa > 0: pos = ("long", e, e+3*pa, e-2*pa)
            elif pr > 70 and pa > 0: pos = ("short", e, e-3*pa, e+2*pa)
        else:
            dr, e, tp, sl = pos; hi, lo = row["high"], row["low"]; r = None
            if dr == "long":
                if lo <= sl: r = (sl-e)/e
                elif hi >= tp: r = (tp-e)/e
            else:
                if hi >= sl: r = (e-sl)/e
                elif lo <= tp: r = (e-tp)/e
            if r is not None:
                out.append((row["timestamp"], r-cost)); pos = None
    return out


def run(args):
    name, syms = args
    N = len(syms)
    allt = []
    for s in syms:
        for ts, r in sym_trades(s, SLIP[s]/10000):
            allt.append((ts, r/N))  # 자본 1/N
    t = pd.DataFrame(allt, columns=["t", "r"]).set_index("t").sort_index()
    daily = t.r.groupby(t.index.normalize()).sum()  # 청산일별 실현R 합
    eq = (1+daily).cumprod()
    mdd = ((eq-eq.cummax())/eq.cummax()).min()*100
    sharpe = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year.map(lambda y: min(y, 2025))).sum()*100
    return {"name": name, "n": len(t), "tot": round((eq.iloc[-1]-1)*100),
            "mdd": round(mdd), "sharpe": round(sharpe, 2),
            "yr": {y: round(yr.get(y, 0)) for y in [2022, 2023, 2024, 2025]}}


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(run, POOLS.items()))
    print(f"{'풀':22} | {'거래':>4} {'총R%':>6} {'Sharpe':>6} {'MDD%':>6}  연도별")
    for r in res:
        y = r["yr"]
        print(f"{r['name']:22} | {r['n']:4} {r['tot']:+6} {r['sharpe']:6.2f} {r['mdd']:6} "
              f"  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
