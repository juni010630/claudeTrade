"""새 엣지 조합 전수 greedy search — IS 그리드 스크리닝 (반복 확장형).

TF × 지표템플릿 × 파라미터 × 방향(follow/fade) × 출구(TP/SL) 조합을 전수 평가.
IS = 2022-01-01 ~ 2024-12-31 (데이터 절단 로드, OOS 봉인).
신호 = t-1 완성봉(cond.shift(1)), 진입 = t 시가, SL 우선(비관), 비용 왕복 10bp.

사용: python3 scripts/newedge_grid.py <tf>   (tf ∈ 1d/4h/1h)
출력: data/results/newedge_grid_<tf>.csv (append 아님 — TF별 전체 재생성)
지표 컬럼은 심볼당 1회 사전계산 → 변형은 임계만 적용 (반복 줄이기).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import COST, atr, ema, load_universe, rsi

RES = Path(__file__).parent.parent / "data" / "results"

HOLD = {"1d": 30, "4h": 90, "1h": 168}
EXITS = [(3.0, 1.5), (4.0, 2.0), (6.0, 3.0)]


def precompute(df):
    c = df["close"]
    df["__atr"] = atr(df, 14)
    for n in (2, 7, 14):
        df[f"rsi{n}"] = rsi(c, n)
    for n in (5, 8, 10, 20, 21, 50, 100):
        df[f"ema{n}"] = ema(c, n)
    for f, s, sg, tag in ((12, 26, 9, "a"), (12, 21, 7, "b")):
        macd = ema(c, f) - ema(c, s)
        df[f"hist_{tag}"] = macd - macd.ewm(span=sg, adjust=False).mean()
    for N in (10, 20, 55):
        df[f"dhi{N}"] = c.rolling(N).max().shift(1)  # t 시점: t-N..t-1 최고종가
        df[f"dlo{N}"] = c.rolling(N).min().shift(1)
    df["bbmid"] = c.rolling(20).mean()
    df["bbstd"] = c.rolling(20).std()
    for L in (5, 20, 60):
        rl = c.pct_change(L)
        df[f"roc{L}"] = rl
        df[f"rocsig{L}"] = rl.rolling(100).std()
    df["r1"] = c.pct_change()
    df["r1sig"] = df["r1"].rolling(100).std()
    df["volma"] = df["volume"].rolling(100).mean()
    return df


def t_oscillator(df, n, lo, hi, mode):
    r = df[f"rsi{n}"]
    if mode == "fade":  # 과매도 롱 / 과매수 숏
        return (r < lo), (r > hi)
    return (r > hi), (r < lo)  # follow: 강세 모멘텀 롱


def t_macross(df, f, s, mode):
    fa, sl_ = df[f"ema{f}"], df[f"ema{s}"]
    up = (fa > sl_) & (fa.shift(1) <= sl_.shift(1))
    dn = (fa < sl_) & (fa.shift(1) >= sl_.shift(1))
    return (up, dn) if mode == "follow" else (dn, up)


def t_donchian(df, N, mode):
    c = df["close"]
    up = c > df[f"dhi{N}"]
    dn = c < df[f"dlo{N}"]
    return (up, dn) if mode == "follow" else (dn, up)


def t_bb(df, k, mode):
    c = df["close"]
    up_band = df["bbmid"] + k * df["bbstd"]
    dn_band = df["bbmid"] - k * df["bbstd"]
    if mode == "follow":  # 밴드 돌파 추종
        return (c > up_band), (c < dn_band)
    return (c < dn_band), (c > up_band)  # 밴드 터치 역추세


def t_roc(df, L, k, mode):
    rl, sg = df[f"roc{L}"], df[f"rocsig{L}"]
    up = rl > k * sg
    dn = rl < -k * sg
    return (up, dn) if mode == "follow" else (dn, up)


def t_stretch(df, n, k, mode):
    z = (df["close"] - df[f"ema{n}"]) / df["__atr"]
    up = z > k
    dn = z < -k
    return (up, dn) if mode == "follow" else (dn, up)


def t_volspike(df, kr, kv, mode):
    burst = (df["r1"].abs() > kr * df["r1sig"]) & (df["volume"] > kv * df["volma"])
    up = burst & (df["r1"] > 0)
    dn = burst & (df["r1"] < 0)
    return (up, dn) if mode == "follow" else (dn, up)


def t_macdflip(df, tag, mode):
    h = df[f"hist_{tag}"]
    up = (h > 0) & (h.shift(1) <= 0)
    dn = (h < 0) & (h.shift(1) >= 0)
    return (up, dn) if mode == "follow" else (dn, up)


def variants():
    out = []
    for n in (2, 7, 14):
        for lo, hi in ((20, 80), (30, 70)):
            for mode in ("fade", "follow"):
                out.append(("osc", dict(n=n, lo=lo, hi=hi, mode=mode)))
    for f, s in ((5, 20), (8, 21), (10, 50), (20, 100)):
        for mode in ("follow", "fade"):
            out.append(("macross", dict(f=f, s=s, mode=mode)))
    for N in (10, 20, 55):
        for mode in ("follow", "fade"):
            out.append(("donchian", dict(N=N, mode=mode)))
    for k in (2.0, 2.5, 3.0):
        for mode in ("follow", "fade"):
            out.append(("bb", dict(k=k, mode=mode)))
    for L in (5, 20, 60):
        for k in (1.5, 2.5):
            for mode in ("follow", "fade"):
                out.append(("roc", dict(L=L, k=k, mode=mode)))
    for n in (20, 50):
        for k in (2.0, 3.0):
            for mode in ("follow", "fade"):
                out.append(("stretch", dict(n=n, k=k, mode=mode)))
    for kr, kv in ((2.5, 3.0), (3.5, 5.0)):
        for mode in ("follow", "fade"):
            out.append(("volspike", dict(kr=kr, kv=kv, mode=mode)))
    for tag in ("a", "b"):
        for mode in ("follow", "fade"):
            out.append(("macdflip", dict(tag=tag, mode=mode)))
    return out


TEMPLATES = dict(osc=t_oscillator, macross=t_macross, donchian=t_donchian, bb=t_bb,
                 roc=t_roc, stretch=t_stretch, volspike=t_volspike, macdflip=t_macdflip)


def sim_fast(df, sig, tp_mult, sl_mult, max_hold):
    """sim_trades와 동일 시맨틱스(SL우선 비관, 청산봉 다음부터 재진입) — 신호 인덱스 점프로 가속."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    a = df["__atr"].values
    ts = df["timestamp"].values
    n = len(o)
    trades = []
    pos_end = 0  # 마지막 청산봉 (이 봉까지는 재진입 금지)
    for i in np.nonzero(sig)[0]:
        if i <= pos_end or i < 1:
            continue
        s = sig[i]
        ai = a[i - 1]
        if not np.isfinite(ai) or ai <= 0:
            continue
        e = o[i]
        tp = e + s * tp_mult * ai
        sl = e - s * sl_mult * ai
        r = None
        j = i
        while j < n:
            if s > 0:
                if l[j] <= sl:
                    r = (sl - e) / e
                    break
                if h[j] >= tp:
                    r = (tp - e) / e
                    break
            else:
                if h[j] >= sl:
                    r = (e - sl) / e
                    break
                if l[j] <= tp:
                    r = (e - tp) / e
                    break
            if j - i + 1 >= max_hold:
                r = (c[j] - e) / e * s
                break
            j += 1
        if r is None:
            r = (c[-1] - e) / e * s
            j = n - 1
        trades.append((ts[i], r - COST))
        pos_end = j
    return trades


def build_sig(df, tmpl, params):
    fn = TEMPLATES[tmpl]
    kw = dict(params)
    long_c, short_c = fn(df, **kw)
    sig = np.where(long_c.shift(1).fillna(False), 1,
                   np.where(short_c.shift(1).fillna(False), -1, 0))
    return sig


def main():
    tf = sys.argv[1]
    dfs = load_universe(tf)
    for sym in list(dfs):
        dfs[sym] = precompute(dfs[sym])
    hold = HOLD[tf]
    rows = []
    vs = variants()
    total = len(vs) * len(EXITS)
    print(f"tf={tf} universe={len(dfs)} variants={total}")
    done = 0
    for tmpl, params in vs:
        sigs = {sym: build_sig(df, tmpl, params) for sym, df in dfs.items()}
        for tp, sl in EXITS:
            tr = []
            for sym, df in dfs.items():
                tr += sim_fast(df, sigs[sym], tp, sl, hold)
            done += 1
            if not tr:
                continue
            t = pd.DataFrame(tr, columns=["t", "r"])
            t["y"] = pd.to_datetime(t.t).dt.year
            gw = t.r[t.r > 0].sum()
            gl = -t.r[t.r <= 0].sum()
            yr = t.groupby("y").r.sum()
            rows.append(dict(
                tf=tf, tmpl=tmpl, params=json.dumps(params), tp=tp, sl=sl,
                n=len(t), wr=round((t.r > 0).mean() * 100, 1),
                pf=round(gw / gl if gl > 0 else 99, 3),
                avg_bp=round(t.r.mean() * 10000, 1),
                totR=round(t.r.sum() * 100, 1),
                y22=round(yr.get(2022, 0) * 100, 1),
                y23=round(yr.get(2023, 0) * 100, 1),
                y24=round(yr.get(2024, 0) * 100, 1),
                posY=int(sum(yr.get(y, 0) > 0 for y in (2022, 2023, 2024))),
            ))
            if done % 50 == 0:
                print(f"  {done}/{total}")
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"newedge_grid_{tf}.csv", index=False)
    print(f"saved {len(out)} rows -> newedge_grid_{tf}.csv")


if __name__ == "__main__":
    main()
