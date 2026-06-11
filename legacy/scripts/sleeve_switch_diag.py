"""슬리브 on/off 스위치 진단 — 어떤 추세지표를 어떤 TF/기준자산으로 보면
슬리브 좋은날/나쁜날이 갈리나. 버킷별 슬리브 일별수익 평균/Sharpe.

지표(추세강도): ADX(14), Kaufman ER(n), Choppiness(역=추세), 회귀 R²(n)
기준자산: BTC / 슬리브 5심볼 평균
TF: 1d / 주간(W)
가설: 추세약(저ADX/저ER/고Choppiness/저R²) 버킷에서 슬리브 수익 >> 추세강 버킷.
look-ahead 가드: 지표는 전봉까지로 계산(shift), 당일 슬리브 수익과 매칭.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

COST = 0.0010
SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def rsi(c, n=14):
    dd = c.diff()
    up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn)


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def load(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()[["open", "high", "low", "close"]]


def sleeve_daily_pnl(d, tp_mult=3.0, sl_mult=2.0, os_th=30, ob_th=70):
    d = d.copy()
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


# ---- 추세강도 지표 (전부 high=강추세 방향으로 부호 통일) ----
def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus = ((up > dn) & (up > 0)) * up
    minus = ((dn > up) & (dn > 0)) * dn
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*plus.ewm(alpha=1/n, adjust=False).mean()/atr_
    mdi = 100*minus.ewm(alpha=1/n, adjust=False).mean()/atr_
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def er(c, n=10):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def chop_inv(df, n=14):
    """Choppiness Index의 역(100-CHOP)=추세강도. 높을수록 추세."""
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atrsum = tr.rolling(n).sum()
    rng = h.rolling(n).max() - l.rolling(n).min()
    chop = 100*np.log10(atrsum/rng.replace(0, np.nan))/np.log10(n)
    return 100 - chop


def r2(c, n=20):
    """선형회귀 결정계수(추세 직선성). 높을수록 추세."""
    x = np.arange(n)
    xm = x.mean(); xd = x - xm; sxx = (xd**2).sum()
    def f(w):
        ym = w.mean(); yd = w - ym
        b = (xd*yd).sum()/sxx
        ss_res = ((yd - b*xd)**2).sum()
        ss_tot = (yd**2).sum()
        return 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return c.rolling(n).apply(f, raw=True)


def bucket_report(name, sleeve_ret, ind):
    """지표를 전봉 shift(look-ahead 가드) 후 3분위 버킷별 슬리브 수익."""
    ind = ind.shift(1).reindex(sleeve_ret.index)
    j = pd.concat([sleeve_ret, ind], axis=1, keys=["r", "i"]).dropna()
    if len(j) < 100:
        return
    q = j.i.quantile([1/3, 2/3]).values
    lo = j[j.i <= q[0]]; mid = j[(j.i > q[0]) & (j.i <= q[1])]; hi = j[j.i > q[1]]
    def sh(x): return x.r.mean()/x.r.std()*np.sqrt(365) if x.r.std() > 0 else 0.0
    print(f"  {name:22} 약추세(저) 평균{lo.r.mean()*1e4:+6.1f}bp Sh{sh(lo):+5.2f} | "
          f"중 {mid.r.mean()*1e4:+6.1f}bp Sh{sh(mid):+5.2f} | "
          f"강추세(고) {hi.r.mean()*1e4:+6.1f}bp Sh{sh(hi):+5.2f}  "
          f"[격차 {(lo.r.mean()-hi.r.mean())*1e4:+.1f}bp]")


def main():
    data = {s: load(s) for s in SLEEVE}
    btc = load("BTCUSDT")
    # 슬리브 합산 일별수익 (등가중)
    sret = sum(sleeve_daily_pnl(data[s]) for s in SLEEVE)
    sret = sret[sret.index >= "2022-01-01"]
    print(f"슬리브 합산 일수 {len(sret)}, 진입 있던 날 {(sret != 0).sum()}\n")

    # 기준자산별 지표
    def basket(fn):
        """5심볼 지표 평균."""
        return pd.concat([fn(data[s]) for s in SLEEVE], axis=1).mean(axis=1)

    print("=== 기준자산: BTC (1d) ===  [격차>0 = 약추세서 슬리브 우위 = 스위치 신호]")
    bucket_report("ADX(14)", sret, adx(btc, 14))
    bucket_report("ER(10)", sret, er(btc["close"], 10))
    bucket_report("ER(20)", sret, er(btc["close"], 20))
    bucket_report("Chop역(14)", sret, chop_inv(btc, 14))
    bucket_report("R2(20)", sret, r2(btc["close"], 20))

    print("\n=== 기준자산: 슬리브5 평균 (1d) ===")
    bucket_report("ADX(14)", sret, basket(lambda d: adx(d, 14)))
    bucket_report("ER(10)", sret, basket(lambda d: er(d["close"], 10)))
    bucket_report("ER(20)", sret, basket(lambda d: er(d["close"], 20)))
    bucket_report("Chop역(14)", sret, basket(lambda d: chop_inv(d, 14)))
    bucket_report("R2(20)", sret, basket(lambda d: r2(d["close"], 20)))

    # 주간 TF: BTC 주간봉 지표 → 일별로 ffill
    print("\n=== 기준자산: BTC (주간 TF) ===")
    bw = btc.resample("W").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bucket_report("ADX_W(14)", sret, adx(bw, 14).reindex(sret.index, method="ffill"))
    bucket_report("ER_W(10)", sret, er(bw["close"], 10).reindex(sret.index, method="ffill"))
    bucket_report("Chop역_W(14)", sret, chop_inv(bw, 14).reindex(sret.index, method="ffill"))


if __name__ == "__main__":
    main()
