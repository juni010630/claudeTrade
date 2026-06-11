"""반복 5 — 미탐색 축: ①macross 채널청산 변형 ②2팩터 조합(게이트×트리거) ③2d/3d 슬로우 TF.

전부 IS(2022~24) 전용 — OOS 봉인. 게이트 동일: n>=200, avg>=40bp, PF>=1.25, 3년 흑자.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import COST, atr, ema, load_universe
from scripts.newedge_grid import sim_fast
from scripts.newedge_it2_macross import macross_sig


def stats(tag, tr):
    if not tr:
        print(f"{tag:<46} (no trades)")
        return
    t = pd.DataFrame(tr, columns=["t", "r"])
    t["y"] = pd.to_datetime(t.t).dt.year
    gw = t.r[t.r > 0].sum()
    gl = -t.r[t.r <= 0].sum()
    yr = t.groupby("y").r.sum()
    posY = sum(yr.get(y, 0) > 0 for y in (2022, 2023, 2024))
    line = " ".join(f"{y}:{round(t[t.y==y].r.mean()*10000) if len(t[t.y==y]) else 0:>5}" for y in (2022, 2023, 2024))
    mark = " ***" if (len(t) >= 200 and t.r.mean() >= 0.004 and (gw / gl if gl > 0 else 99) >= 1.25 and posY == 3) else ""
    print(f"{tag:<46} n={len(t):>5} pf={round(gw/gl if gl>0 else 99,2):<5} avg={round(t.r.mean()*10000):>5}bp posY={posY} | {line}{mark}")


def resample(df, days):
    d = df.set_index("timestamp").resample(f"{days}D").agg(
        dict(open="first", high="max", low="min", close="last", volume="sum")).dropna().reset_index()
    return d


def macross_channel_exit(df, f, s, exit_n=10, sl_mult=3.0):
    """macross 진입 + 반대편 N일 채널 이탈 청산(다음 시가) + SL 보호."""
    c = df["close"]
    sig = macross_sig(df, f, s)
    o, h, l, cv = df["open"].values, df["high"].values, df["low"].values, c.values
    a = df["__atr"].values
    ts = df["timestamp"].values
    ehi = c.rolling(exit_n).max().shift(1).values
    elo = c.rolling(exit_n).min().shift(1).values
    n = len(df)
    trades = []
    pos_end = 0
    for i in np.nonzero(sig)[0]:
        if i <= pos_end or i < 1:
            continue
        s_ = sig[i]
        ai = a[i - 1]
        if not np.isfinite(ai) or ai <= 0:
            continue
        e = o[i]
        slp = e - s_ * sl_mult * ai
        r = None
        j = i
        while j < n - 1:
            if s_ > 0 and l[j] <= slp:
                r = (slp - e) / e
                break
            if s_ < 0 and h[j] >= slp:
                r = (e - slp) / e
                break
            if j > i and ((s_ > 0 and cv[j] < elo[j]) or (s_ < 0 and cv[j] > ehi[j])):
                r = (o[j + 1] - e) / e * s_
                break
            j += 1
        if r is None:
            r = (cv[-1] - e) / e * s_
            j = n - 1
        trades.append((ts[i], r - COST))
        pos_end = j
    return trades


def gate_series(df, kind):
    c = df["close"]
    if kind == "up100":   # ema100 위 + 상승 기울기
        e100 = ema(c, 100)
        return (c > e100) & (e100 > e100.shift(5)), (c < e100) & (e100 < e100.shift(5))
    if kind == "squeeze":
        natr = df["__atr"] / c
        rk = natr.rolling(100).rank(pct=True)
        g = rk < 0.35
        return g, g
    if kind == "calm":    # 저변동 국면
        natr = df["__atr"] / c
        g = natr < natr.rolling(200).median()
        return g, g
    raise ValueError(kind)


def trigger_series(df, kind):
    c = df["close"]
    if kind == "don10":
        up = c > c.rolling(10).max().shift(1)
        dn = c < c.rolling(10).min().shift(1)
    elif kind == "roc20":
        rl = c.pct_change(20)
        sg = rl.rolling(100).std()
        up, dn = rl > 1.5 * sg, rl < -1.5 * sg
    elif kind == "macross_f":
        fa, sl_ = ema(c, 5), ema(c, 20)
        up = (fa > sl_) & (fa.shift(1) <= sl_.shift(1))
        dn = (fa < sl_) & (fa.shift(1) >= sl_.shift(1))
    else:
        raise ValueError(kind)
    return up, dn


def main():
    dfs = load_universe("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"universe={len(dfs)} (IS only)\n== ① macross 채널청산 변형 ==")
    for f_, s_ in ((10, 100), (20, 100)):
        for en in (10, 20):
            tr = []
            for df in dfs.values():
                tr += macross_channel_exit(df, f_, s_, en)
            stats(f"chx ema{f_}x{s_} exit{en}", tr)

    print("\n== ② 2팩터 조합 (게이트 × 트리거, tp6/sl3 hold60) ==")
    for g in ("up100", "squeeze", "calm"):
        for tk in ("don10", "roc20", "macross_f"):
            tr = []
            for df in dfs.values():
                gl_, gs_ = gate_series(df, g)
                tu, td = trigger_series(df, tk)
                up = (gl_ & tu)
                dn = (gs_ & td)
                sig = np.where(up.shift(1).fillna(False), 1, np.where(dn.shift(1).fillna(False), -1, 0))
                tr += sim_fast(df, sig, 6.0, 3.0, 60)
            stats(f"combo {g}+{tk}", tr)

    print("\n== ③ 2d/3d 슬로우 TF (리샘플) ==")
    for days in (2, 3):
        rdfs = {sym: resample(df, days) for sym, df in dfs.items()}
        for df in rdfs.values():
            df["__atr"] = atr(df, 14)
        for f_, s_ in ((10, 50), (20, 100)):
            tr = []
            for df in rdfs.values():
                sig = macross_sig(df, f_, s_)
                tr += sim_fast(df, sig, 6.0, 3.0, max(20, 60 // days))
            stats(f"{days}d ema{f_}x{s_} tp6/3", tr)
        for N in (10, 20):
            tr = []
            for df in rdfs.values():
                c = df["close"]
                up = c > c.rolling(N).max().shift(1)
                dn = c < c.rolling(N).min().shift(1)
                sig = np.where(up.shift(1).fillna(False), 1, np.where(dn.shift(1).fillna(False), -1, 0))
                tr += sim_fast(df, sig, 6.0, 3.0, max(20, 60 // days))
            stats(f"{days}d donchian{N} tp6/3", tr)


if __name__ == "__main__":
    main()
