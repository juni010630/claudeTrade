"""RSI 극단 평균회귀를 전 심볼에 스크리닝 — 연도별 강건한 알트 발굴.

1d RSI(14) 극단 진입, ATR 기반 고정 TP/SL, 비용 차감, 연도별 분해.
목적: LTC처럼 RSI 평균회귀가 4년 강건한 알트를 찾기.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

COST = 0.0010  # 왕복 비용 10bp (수수료+슬리피지 보수적)


def rsi(c, n=14):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def backtest(d, tp_mult=3.0, sl_mult=2.0, os_th=30, ob_th=70):
    d = d.copy()
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    arr = d.reset_index()
    trades = []
    pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        if pos is None:
            pr, pa = arr.iloc[i-1]["rsi"], arr.iloc[i-1]["atr"]
            e = row["open"]
            if pr < os_th and pa > 0:
                pos = ("long", e, e+tp_mult*pa, e-sl_mult*pa, row["timestamp"])
            elif pr > ob_th and pa > 0:
                pos = ("short", e, e-tp_mult*pa, e+sl_mult*pa, row["timestamp"])
        else:
            dr, e, tp, sl, et = pos
            hi, lo = row["high"], row["low"]
            r = None
            if dr == "long":
                if lo <= sl: r = (sl-e)/e
                elif hi >= tp: r = (tp-e)/e
            else:
                if hi >= sl: r = (e-sl)/e
                elif lo <= tp: r = (e-tp)/e
            if r is not None:
                trades.append((et, r - COST)); pos = None
    if not trades:
        return None
    tdf = pd.DataFrame(trades, columns=["t", "r"])
    tdf["year"] = pd.to_datetime(tdf.t).dt.year.clip(upper=2025)
    gw = tdf.r[tdf.r > 0].sum(); gl = -tdf.r[tdf.r <= 0].sum()
    pf = gw/gl if gl > 0 else 99.0
    yr = tdf.groupby("year").r.sum()
    pos_years = sum(1 for y in [2022, 2023, 2024, 2025] if yr.get(y, 0) > 0)
    return {"pf": round(pf, 2), "wr": round((tdf.r > 0).mean()*100, 0),
            "totR%": round(tdf.r.sum()*100, 0), "n": len(tdf),
            "pos_years": pos_years,
            "yr": {y: round(yr.get(y, 0)*100) for y in [2022, 2023, 2024, 2025]}}


def main():
    syms = []
    for f in os.listdir("data/cache"):
        if f.startswith("ohlcv_") and f.endswith("_1d.parquet"):
            syms.append(f[6:-11])
    results = []
    for s in syms:
        try:
            d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            d = d.set_index("timestamp").sort_index()
            d = d[d.index >= "2022-01-01"]
            if len(d) < 400:  # 4년 데이터 없으면 스킵
                continue
            r = backtest(d)
            if r and r["n"] >= 15:
                results.append({"sym": s, **r})
        except Exception:
            continue
    df = pd.DataFrame(results)
    # 강건성: 4년 양수 개수 우선, 그다음 PF
    df = df.sort_values(["pos_years", "pf"], ascending=False)
    print(f"=== RSI 평균회귀(3.0/2.0, 30/70, 비용10bp) 전심볼 스크리닝 ({len(df)}개) ===")
    print(f"{'심볼':12} {'PF':>5} {'승률':>4} {'총R%':>6} {'거래':>4} {'양수년':>5}  연도별(22/23/24/25)")
    for _, x in df.head(25).iterrows():
        y = x["yr"]
        print(f"{x['sym']:12} {x['pf']:5.2f} {x['wr']:3.0f}% {x['totR%']:+6.0f} {x['n']:4} "
              f"{x['pos_years']:4}/4  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
