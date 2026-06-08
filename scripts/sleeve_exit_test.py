"""슬리브 청산 로직 비교 — ATR TP/SL vs 시간청산 vs RSI 중앙복귀 (1d, 5심볼 합산)."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SYMS = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
SLIP = {"LTCUSDT": 13, "UNIUSDT": 14, "STORJUSDT": 36, "ARPAUSDT": 40, "BANDUSDT": 56}


def rsi(c, n=14):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-100/(1+up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def backtest(d, mode, sl_mult=2.0, tp_mult=3.0, hold_days=10, cost=0.001):
    d = d.copy(); d["rsi"] = rsi(d["close"], 14); d["atr"] = atr(d, 14)
    arr = d.reset_index(); trades = []; pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        if pos is None:
            pr, pa = arr.iloc[i-1]["rsi"], arr.iloc[i-1]["atr"]
            e = row["open"]
            if pr < 30 and pa > 0:
                pos = ["long", e, e-sl_mult*pa, e+tp_mult*pa, row["timestamp"], 0]
            elif pr > 70 and pa > 0:
                pos = ["short", e, e+sl_mult*pa, e-tp_mult*pa, row["timestamp"], 0]
        else:
            dr, e, sl, tp, et, days = pos; hi, lo, c, rv = row["high"], row["low"], row["close"], row["rsi"]
            pos[5] += 1; days = pos[5]; r = None
            # SL 항상 우선
            if dr == "long" and lo <= sl: r = (sl-e)/e
            elif dr == "short" and hi >= sl: r = (e-sl)/e
            else:
                if mode == "atr":
                    if dr == "long" and hi >= tp: r = (tp-e)/e
                    elif dr == "short" and lo <= tp: r = (e-tp)/e
                elif mode == "time":
                    if days >= hold_days: r = (c-e)/e if dr == "long" else (e-c)/e
                elif mode == "rsi":  # RSI 중앙(50) 복귀 시 청산
                    if (dr == "long" and rv >= 50) or (dr == "short" and rv <= 50):
                        r = (c-e)/e if dr == "long" else (e-c)/e
            if r is not None:
                trades.append((et, r-cost)); pos = None
    if not trades: return None
    t = pd.DataFrame(trades, columns=["t", "r"]); t["year"] = pd.to_datetime(t.t).dt.year.clip(upper=2025)
    return t


def run(args):
    mode, label = args
    allt = []
    for s in SYMS:
        d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"]); d = d.set_index("timestamp").sort_index()
        d = d[d.index >= "2022-01-01"]
        t = backtest(d, mode, cost=SLIP[s]/10000)
        if t is not None:
            t["sym"] = s; allt.append(t)
    T = pd.concat(allt); T["sr"] = T.r*0.2  # 자본 1/5
    gw = T.sr[T.sr > 0].sum(); gl = -T.sr[T.sr <= 0].sum()
    yr = T.groupby("year").sr.sum()*100
    return {"label": label, "n": len(T), "wr": round((T.r > 0).mean()*100),
            "pf": round(gw/gl, 2), "tot": round(T.sr.sum()*100),
            "yr": {y: round(yr.get(y, 0)) for y in [2022, 2023, 2024, 2025]}}


def main():
    variants = [("atr", "ATR TP3/SL2 (현행)"), ("time", "시간청산 10일+SL2"),
                ("rsi", "RSI50복귀+SL2")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(run, variants))
    print(f"{'청산방식':22} | {'거래':>4} {'승률':>4} {'PF':>5} {'총R%':>6}  연도별")
    for r in res:
        y = r["yr"]
        print(f"{r['label']:22} | {r['n']:4} {r['wr']:3}% {r['pf']:5.2f} {r['tot']:+6} "
              f"  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
