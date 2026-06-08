"""슬리브 추세 필터 — 역추세 칼받기 방어. EMA 방향 순응 평균회귀 (1d, 5심볼)."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

SYMS = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
SLIP = {"LTCUSDT": 13, "UNIUSDT": 14, "STORJUSDT": 36, "ARPAUSDT": 40, "BANDUSDT": 56}


def rsi(c, n=14):
    dd = c.diff(); up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-100/(1+up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def backtest(d, ema_period, cost=0.001):
    d = d.copy(); d["rsi"] = rsi(d["close"], 14); d["atr"] = atr(d, 14)
    d["ema"] = d["close"].ewm(span=ema_period, adjust=False).mean() if ema_period else None
    arr = d.reset_index(); trades = []; pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        if pos is None:
            pr, pa = arr.iloc[i-1]["rsi"], arr.iloc[i-1]["atr"]; e = row["open"]
            up = (arr.iloc[i-1]["close"] > arr.iloc[i-1]["ema"]) if ema_period else None
            long_ok = pr < 30 and (ema_period is None or up)      # 상승추세 눌림목만 롱
            short_ok = pr > 70 and (ema_period is None or not up)  # 하락추세 반등만 숏
            if long_ok and pa > 0:
                pos = ("long", e, e+3*pa, e-2*pa, row["timestamp"])
            elif short_ok and pa > 0:
                pos = ("short", e, e-3*pa, e+2*pa, row["timestamp"])
        else:
            dr, e, tp, sl, et = pos; hi, lo = row["high"], row["low"]; r = None
            if dr == "long":
                if lo <= sl: r = (sl-e)/e
                elif hi >= tp: r = (tp-e)/e
            else:
                if hi >= sl: r = (e-sl)/e
                elif lo <= tp: r = (e-tp)/e
            if r is not None:
                trades.append((et, r-cost)); pos = None
    if not trades: return None
    t = pd.DataFrame(trades, columns=["t", "r"]); t["year"] = pd.to_datetime(t.t).dt.year.clip(upper=2025)
    return t


def run(args):
    ema, label = args
    allt = []
    for s in SYMS:
        d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
        d["timestamp"] = pd.to_datetime(d["timestamp"]); d = d.set_index("timestamp").sort_index()
        d = d[d.index >= "2022-01-01"]
        t = backtest(d, ema, cost=SLIP[s]/10000)
        if t is not None: allt.append(t)
    T = pd.concat(allt); T["sr"] = T.r*0.2
    gw = T.sr[T.sr > 0].sum(); gl = -T.sr[T.sr <= 0].sum()
    yr = T.groupby("year").sr.sum()*100
    return {"label": label, "n": len(T), "wr": round((T.r > 0).mean()*100),
            "pf": round(gw/gl, 2), "tot": round(T.sr.sum()*100),
            "yr": {y: round(yr.get(y, 0)) for y in [2022, 2023, 2024, 2025]}}


def main():
    variants = [(None, "무필터(현행)"), (50, "EMA50 추세순응"),
                (100, "EMA100 추세순응"), (200, "EMA200 추세순응")]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(run, variants))
    print(f"{'추세필터':18} | {'거래':>4} {'승률':>4} {'PF':>5} {'총R%':>6}  연도별(22/23/24/25)")
    for r in res:
        y = r["yr"]
        print(f"{r['label']:18} | {r['n']:4} {r['wr']:3}% {r['pf']:5.2f} {r['tot']:+6} "
              f"  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
