"""슬리브 추세 스위치 임계값 스윕 — 진입 게이트(basket ER(20) > thr면 신규진입 OFF).
기존 포지션은 TP/SL까지 유지(스위치는 ENTRY만 차단). 연도별+plateau+거래수 드롭 확인.
basket/per-symbol 두 변형. look-ahead 가드: ER은 전봉(shift1)."""
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


def er(c, n=20):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def load(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()[["open", "high", "low", "close"]]


def trades_gated(d, gate, tp_mult=3.0, sl_mult=2.0, os_th=30, ob_th=70):
    """gate: DatetimeIndex→bool(진입 허용). 청산은 게이트 무관."""
    d = d.copy()
    d["rsi"] = rsi(d["close"], 14)
    d["atr"] = atr(d, 14)
    arr = d.reset_index()
    out = []
    pos = None
    for i in range(1, len(arr)):
        row = arr.iloc[i]
        t = row["timestamp"]
        if pos is None:
            allow = bool(gate.get(t, True))
            if not allow:
                continue
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
                out.append((t, r - COST)); pos = None
    return out


def summarize(all_trades):
    if not all_trades:
        return None
    tdf = pd.DataFrame(all_trades, columns=["t", "r"]).sort_values("t")
    tdf["year"] = pd.to_datetime(tdf.t).dt.year.clip(upper=2025)
    # 일별 등가중 수익 (같은 날 여러 청산 합산) → equity
    daily = tdf.groupby(pd.to_datetime(tdf.t).dt.normalize()).r.sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).fillna(0)
    eq = (1 + daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = tdf.groupby("year").r.sum()
    posy = sum(1 for y in [2022, 2023, 2024, 2025] if yr.get(y, 0) > 0)
    return {"n": len(tdf), "totR": tdf.r.sum()*100, "sh": sh, "mdd": mdd,
            "posy": posy, "yr": {y: round(yr.get(y, 0)*100) for y in [2022, 2023, 2024, 2025]},
            "final": eq.iloc[-1]}


def main():
    data = {s: load(s) for s in SLEEVE}
    # basket ER(20), shift1 (전봉)
    basket_er = pd.concat([er(data[s]["close"], 20) for s in SLEEVE], axis=1).mean(axis=1).shift(1)
    basket_er = basket_er[basket_er.index >= "2021-06-01"]

    print("=== basket ER(20) 분포 ===")
    print(basket_er.describe(percentiles=[.5, .67, .75, .8, .9]).round(3).to_string())

    # 기준: 게이트 없음
    base = summarize([t for s in SLEEVE for t in trades_gated(data[s], {})])
    print(f"\n=== 게이트 없음 (always-on) ===")
    print(f"  거래{base['n']} 총R{base['totR']:+.0f}% Sh{base['sh']:.2f} MDD{base['mdd']:.0f}% "
          f"양수년{base['posy']}/4 연도{base['yr']}")

    print(f"\n=== basket ER(20) 진입게이트 스윕 (ER>thr면 신규진입 OFF) ===")
    qs = [0.50, 0.60, 0.67, 0.75, 0.80, 0.90]
    print(f"{'thr분위':>7} {'ER값':>6} {'거래':>4} {'총R%':>6} {'Sh':>5} {'MDD%':>6} {'양수년':>5}  연도(22/23/24/25)")
    for q in qs:
        thr = basket_er.quantile(q)
        gate = (basket_er <= thr).to_dict()
        res = summarize([t for s in SLEEVE for t in trades_gated(data[s], gate)])
        y = res["yr"]
        print(f"{q:>7.2f} {thr:>6.3f} {res['n']:>4} {res['totR']:>+6.0f} {res['sh']:>5.2f} "
              f"{res['mdd']:>6.0f} {res['posy']:>4}/4  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")

    # per-symbol 게이트 변형 (각 심볼 자기 ER로)
    print(f"\n=== per-symbol ER(20) 진입게이트 (각 심볼 자기 ER>thr면 OFF) ===")
    print(f"{'thr분위':>7} {'거래':>4} {'총R%':>6} {'Sh':>5} {'MDD%':>6} {'양수년':>5}  연도(22/23/24/25)")
    sym_er = {s: er(data[s]["close"], 20).shift(1) for s in SLEEVE}
    # 공통 분위 임계 (심볼별 자기 분포)
    for q in qs:
        all_t = []
        for s in SLEEVE:
            thr = sym_er[s].quantile(q)
            gate = (sym_er[s] <= thr).to_dict()
            all_t += trades_gated(data[s], gate)
        res = summarize(all_t)
        y = res["yr"]
        print(f"{q:>7.2f} {res['n']:>4} {res['totR']:>+6.0f} {res['sh']:>5.2f} "
              f"{res['mdd']:>6.0f} {res['posy']:>4}/4  {y[2022]:+4} {y[2023]:+4} {y[2024]:+4} {y[2025]:+4}")


if __name__ == "__main__":
    main()
