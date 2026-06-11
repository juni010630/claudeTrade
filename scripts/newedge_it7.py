"""반복 7 — 미탐색 축: 12h TF / 챔피언 미세구조 / 주간 TF. 전부 IS(2022~24) 전용.

사용: python3 scripts/newedge_it7.py {12h|micro|weekly}
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import COST, atr, ema, load_universe, rsi
from scripts.newedge_grid import sim_fast
from scripts.newedge_it2_macross import macross_sig


def stats(tag, tr):
    if not tr:
        print(f"{tag:<40} (no trades)")
        return
    t = pd.DataFrame(tr, columns=["t", "r"])
    t["y"] = pd.to_datetime(t.t).dt.year
    gw = t.r[t.r > 0].sum()
    gl = -t.r[t.r <= 0].sum()
    pf = gw / gl if gl > 0 else 99
    yr = t.groupby("y").r.sum()
    posY = sum(yr.get(y, 0) > 0 for y in (2022, 2023, 2024))
    line = " ".join(f"{y}:{round(t[t.y==y].r.mean()*10000) if len(t[t.y==y]) else 0:>5}" for y in (2022, 2023, 2024))
    mark = " ***" if (len(t) >= 200 and t.r.mean() >= 0.004 and pf >= 1.25 and posY == 3) else ""
    print(f"{tag:<40} n={len(t):>5} pf={round(pf,2):<5} avg={round(t.r.mean()*10000):>5}bp posY={posY} | {line}{mark}")


def resample_tf(df, rule):
    d = df.set_index("timestamp").resample(rule).agg(
        dict(open="first", high="max", low="min", close="last", volume="sum")).dropna().reset_index()
    return d


def run_12h():
    dfs1h = load_universe("1h")
    dfs = {sym: resample_tf(df, "12h") for sym, df in dfs1h.items()}
    del dfs1h
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"universe 12h: {len(dfs)}")
    for f_, s_ in ((8, 21), (10, 50), (20, 100)):
        for tp, sl in ((4.0, 2.0), (6.0, 3.0)):
            tr = []
            for df in dfs.values():
                tr += sim_fast(df, macross_sig(df, f_, s_), tp, sl, 120)
            stats(f"12h ema{f_}x{s_} tp{tp}/{sl}", tr)
    for df in dfs.values():
        c = df["close"]
        df["__kup"] = (c > ema(c, 20) + 2 * df["__atr"])
        df["__kdn"] = (c < ema(c, 20) - 2 * df["__atr"])
    tr = []
    for df in dfs.values():
        sig = np.where(df["__kup"].shift(1).fillna(False), 1,
                       np.where(df["__kdn"].shift(1).fillna(False), -1, 0))
        tr += sim_fast(df, sig, 6.0, 3.0, 120)
    stats("12h keltner20x2 follow", tr)
    for N in (20, 55):
        tr = []
        for df in dfs.values():
            c = df["close"]
            up = c > c.rolling(N).max().shift(1)
            dn = c < c.rolling(N).min().shift(1)
            sig = np.where(up.shift(1).fillna(False), 1, np.where(dn.shift(1).fillna(False), -1, 0))
            tr += sim_fast(df, sig, 6.0, 3.0, 120)
        stats(f"12h donchian{N} follow", tr)


def run_micro():
    dfs = load_universe("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"universe 1d: {len(dfs)}")

    print("== 출구 비대칭 (20x100, hold60) ==")
    for tp, sl in ((6.0, 3.0), (6.0, 2.0), (8.0, 2.0), (8.0, 3.0), (10.0, 3.0), (4.0, 3.0)):
        tr = []
        for df in dfs.values():
            tr += sim_fast(df, macross_sig(df, 20, 100), tp, sl, 60)
        stats(f"exit tp{tp}/sl{sl}", tr)

    print("== hold 스윕 (20x100, tp6/sl3) ==")
    for hd in (30, 60, 90, 120):
        tr = []
        for df in dfs.values():
            tr += sim_fast(df, macross_sig(df, 20, 100), 6.0, 3.0, hd)
        stats(f"hold {hd}", tr)

    print("== 컨플루언스 필터 (20x100 tp6/3 hold60) ==")
    for df in dfs.values():
        c = df["close"]
        macd = ema(c, 12) - ema(c, 26)
        df["__hist"] = macd - macd.ewm(span=9, adjust=False).mean()
        df["__volok"] = df["volume"] > 1.5 * df["volume"].rolling(20).mean()
        df["__rsi14"] = rsi(c, 14)
    def filt(df, sig, kind):
        s = pd.Series(sig, index=df.index)
        if kind == "macd":
            ok = (np.sign(df["__hist"].shift(1)) == s) | (s == 0)
        elif kind == "vol":
            ok = df["__volok"].shift(1).fillna(False) | (s == 0)
        elif kind == "rsi_mid":
            r = df["__rsi14"].shift(1)
            ok = ((r > 30) & (r < 70)) | (s == 0)
        return np.where(ok, s, 0)
    for kind in ("macd", "vol", "rsi_mid"):
        tr = []
        for df in dfs.values():
            sig = filt(df, macross_sig(df, 20, 100), kind)
            tr += sim_fast(df, sig, 6.0, 3.0, 60)
        stats(f"filter +{kind}", tr)

    print("== 유니버스 분해 (20x100 tp6/3) ==")
    dvol = {sym: float((df.close * df.volume).mean()) for sym, df in dfs.items()}
    top12 = set(sorted(dvol, key=dvol.get, reverse=True)[:12])
    for tag, syms in (("majors_top12", top12), ("alts_rest", set(dfs) - top12)):
        tr = []
        for sym in syms:
            df = dfs[sym]
            tr += sim_fast(df, macross_sig(df, 20, 100), 6.0, 3.0, 60)
        stats(f"univ {tag}", tr)


def run_weekly():
    dfs1d = load_universe("1d")
    dfs = {sym: resample_tf(df, "7D") for sym, df in dfs1d.items()}
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"universe 7d: {len(dfs)}")
    for f_, s_ in ((5, 20), (10, 50)):
        tr = []
        for df in dfs.values():
            tr += sim_fast(df, macross_sig(df, f_, s_), 6.0, 3.0, 12)
        stats(f"7d ema{f_}x{s_}", tr)
    for N in (8, 12):
        tr = []
        for df in dfs.values():
            c = df["close"]
            up = c > c.rolling(N).max().shift(1)
            dn = c < c.rolling(N).min().shift(1)
            sig = np.where(up.shift(1).fillna(False), 1, np.where(dn.shift(1).fillna(False), -1, 0))
            tr += sim_fast(df, sig, 6.0, 3.0, 12)
        stats(f"7d donchian{N}", tr)


if __name__ == "__main__":
    {"12h": run_12h, "micro": run_micro, "weekly": run_weekly}[sys.argv[1]]()
