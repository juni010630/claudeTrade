"""슬리브 후보 상관 진단 — 각 심볼 평균회귀 일별 손익 시계열의 상관행렬.
기존 슬리브와 상관 낮은(=분산되는) 후보를 식별. altscan과 동일 로직(공통 파라미터)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

COST = 0.0010

EXIST = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
CAND = ["MTLUSDT", "BELUSDT", "SNXUSDT", "ONTUSDT", "RSRUSDT", "ROSEUSDT"]


def rsi(c, n=14):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def daily_pnl(s, tp_mult=3.0, sl_mult=2.0, os_th=30, ob_th=70):
    """심볼의 평균회귀 매매를 일별 손익(청산일에 R 기록) 시계열로 반환."""
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.set_index("timestamp").sort_index()
    d = d[d.index >= "2022-01-01"]
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    arr = d.reset_index()
    pnl = pd.Series(0.0, index=d.index)
    pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        if pos is None:
            pr, pa = arr.iloc[i-1]["rsi"], arr.iloc[i-1]["atr"]
            e = row["open"]
            if pr < os_th and pa > 0:
                pos = ("long", e, e+tp_mult*pa, e-sl_mult*pa)
            elif pr > ob_th and pa > 0:
                pos = ("short", e, e-tp_mult*pa, e+sl_mult*pa)
        else:
            dr, e, tp, sl = pos
            hi, lo = row["high"], row["low"]
            r = None
            if dr == "long":
                if lo <= sl: r = (sl-e)/e
                elif hi >= tp: r = (tp-e)/e
            else:
                if hi >= sl: r = (e-sl)/e
                elif lo <= tp: r = (e-tp)/e
            if r is not None:
                pnl.loc[row["timestamp"]] += r - COST
                pos = None
    return pnl


def main():
    syms = EXIST + CAND
    pnl = pd.DataFrame({s: daily_pnl(s) for s in syms}).fillna(0.0)
    corr = pnl.corr()

    print("=== 기존 슬리브 합산 대비 후보 상관 (낮을수록 분산 ↑) ===")
    exist_sum = pnl[EXIST].sum(axis=1)
    print(f"{'후보':10} {'기존합산상관':>10} {'기존5개 평균상관':>14}")
    for c in CAND:
        c_vs_sum = pnl[c].corr(exist_sum)
        c_vs_each = corr.loc[c, EXIST].mean()
        print(f"{c:10} {c_vs_sum:>10.3f} {c_vs_each:>14.3f}")

    print("\n=== 전체 상관행렬 ===")
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
