"""새 엣지 greedy search — 레벨 1: 생존 패밀리 정제 (IS 2022~2024 전용, OOS 봉인 유지).

F3 squeeze 중심 greedy(노브 1개씩) + 진단(심볼 집중도/롱숏 분해/유동성 필터)
+ F7 tsmom 이웃 강건성 + F5 fade 확인 + F1 채널 이웃.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import (
    COST, agg, atr, ema, f1_donchian, f5_leadlag, f7_tsmom, load_universe, rsi, sim_trades,
)


def f3_squeeze_sym(df, pct_th=0.3, N=20, tp=4.0, sl=2.0, hold=30, win=100, liq_min=0.0):
    c = df["close"]
    natr = df["__atr"] / c
    rank = natr.rolling(win).rank(pct=True)
    squeezed = rank.shift(2) < pct_th
    hh = c.rolling(N).max().shift(2)
    ll = c.rolling(N).min().shift(2)
    long_sig = squeezed & (c.shift(1) > hh)
    short_sig = squeezed & (c.shift(1) < ll)
    if liq_min > 0:
        dvol = (df["close"] * df["volume"]).rolling(30).median().shift(1)
        ok = dvol > liq_min
        long_sig &= ok
        short_sig &= ok
    sig = np.where(long_sig.fillna(False), 1, np.where(short_sig.fillna(False), -1, 0))
    return sig


def run_f3(dfs, tag, with_side=False, **kw):
    trades = []  # (ts, r, sym, side)
    for sym, df in dfs.items():
        sig = f3_squeeze_sym(df, **kw)
        tr = sim_trades(df, sig, kw.get("tp", 4.0), kw.get("sl", 2.0), kw.get("hold", 30))
        # sim_trades는 (ts, r)만 반환 — 사이드 정보는 sig에서 재추출 못 하므로 별도 기록
        for t, r in tr:
            trades.append((t, r, sym))
    a = agg([(t, r) for t, r, _ in trades])
    if a is None:
        print(f"{tag:<40} (no trades)")
        return None
    tdf = pd.DataFrame(trades, columns=["t", "r", "sym"])
    by_sym = tdf.groupby("sym").r.sum().sort_values(ascending=False)
    top10 = by_sym.head(10).sum() / max(tdf.r.sum(), 1e-9)
    pos_sym = (by_sym > 0).mean()
    print(
        f"{tag:<40} n={a['n']:>5} wr={a['wr']} pf={a['pf']:<5} totR={a['totR%']:>6} "
        f"avg={a['avgR_bp']:>4}bp posY={a['pos_years']} yr={a['yr']} "
        f"top10sh={top10:.2f} posSym={pos_sym:.2f}"
    )
    return tdf


def sim_trades_sided(df, sig, tp_mult, sl_mult, max_hold):
    """sim_trades와 동일하되 (ts, r, side) 반환."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    ts = df["timestamp"].values
    a = df["__atr"].values
    n = len(df)
    trades = []
    i = 1
    while i < n:
        s = sig[i]
        if s == 0 or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
            i += 1
            continue
        e = o[i]
        tp = e + s * tp_mult * a[i - 1]
        sl = e - s * sl_mult * a[i - 1]
        r = None
        j = i
        while j < n:
            if s > 0 and l[j] <= sl:
                r = (sl - e) / e
                break
            if s > 0 and h[j] >= tp:
                r = (tp - e) / e
                break
            if s < 0 and h[j] >= sl:
                r = (e - sl) / e
                break
            if s < 0 and l[j] <= tp:
                r = (e - tp) / e
                break
            if j - i + 1 >= max_hold:
                r = (c[j] - e) / e * s
                break
            j += 1
        if r is None:
            r = (c[-1] - e) / e * s
            j = n - 1
        trades.append((ts[i], r - COST, "L" if s > 0 else "S"))
        i = j + 1
    return trades


def main():
    dfs = load_universe("1d")
    print(f"universe: {len(dfs)} symbols (IS only)\n")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)

    base = dict(pct_th=0.3, N=20, tp=4.0, sl=2.0, hold=30, win=100, liq_min=0.0)

    print("== F3 greedy: pct_th ==")
    for p in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        run_f3(dfs, f"F3 pct={p}", **{**base, "pct_th": p})

    print("\n== F3 greedy: 채널 N ==")
    for N in [10, 15, 20, 30, 40]:
        run_f3(dfs, f"F3 N={N}", **{**base, "N": N})

    print("\n== F3 greedy: TP/SL ==")
    for tp, sl in [(3.0, 1.5), (3.0, 2.0), (4.0, 2.0), (5.0, 2.5), (6.0, 3.0)]:
        run_f3(dfs, f"F3 tp{tp}/sl{sl}", **{**base, "tp": tp, "sl": sl})

    print("\n== F3 greedy: hold ==")
    for hd in [15, 30, 45, 60]:
        run_f3(dfs, f"F3 hold={hd}", **{**base, "hold": hd})

    print("\n== F3 greedy: 유동성 필터 (30d 중위 달러볼륨) ==")
    for lm in [0, 3e6, 10e6, 30e6]:
        run_f3(dfs, f"F3 liq>{lm/1e6:.0f}M", **{**base, "liq_min": lm})

    print("\n== F3 롱/숏 분해 (base) ==")
    sided = []
    for sym, df in dfs.items():
        sig = f3_squeeze_sym(df, **base)
        sided += sim_trades_sided(df, sig, base["tp"], base["sl"], base["hold"])
    sdf = pd.DataFrame(sided, columns=["t", "r", "side"])
    sdf["year"] = pd.to_datetime(sdf.t).dt.year
    for side in ["L", "S"]:
        sub = sdf[sdf.side == side]
        yr = sub.groupby("year").r.sum()
        print(
            f"  {side}: n={len(sub)} wr={round((sub.r>0).mean()*100)} totR={round(sub.r.sum()*100)} "
            f"yr={{ {', '.join(f'{y}: {round(yr.get(y,0)*100)}' for y in [2022,2023,2024])} }}"
        )

    print("\n== F1 channel 이웃 (N, exit) ==")
    for N in [20, 30, 40]:
        tr = []
        for df in dfs.values():
            tr += f1_donchian(df, N, "channel")
        a = agg(tr)
        print(f"  F1 N={N} channel: n={a['n']} pf={a['pf']} totR={a['totR%']} yr={a['yr']}")

    print("\n== F7 tsmom 이웃 ==")
    for lb in [25, 30, 35, 40, 45]:
        for rb in [3, 5, 7]:
            r = f7_tsmom(dfs, lb, rb)
            print(f"  F7 lb={lb} rb={rb}: sharpe={r['pf']} totR={r['totR%']} yr={r['yr']} mdd={r.get('mdd%')}")

    print("\n== F5 fade 확인 (BTC 큰날 → 알트 다음날 역방향) ==")
    btc = dfs.get("BTCUSDT")
    for th, hold in [(0.02, 1), (0.02, 2), (0.03, 1)]:
        tr = f5_leadlag(dfs, btc, th, hold)
        fade = [(t, -r - 2 * COST + COST) for t, r in tr]  # 부호반전: 원 r에 -COST 포함이므로 보정
        # 원 r = raw*s - COST → raw*s = r + COST → fade raw = -(r+COST) → fade r = -(r+COST) - COST
        fade = [(t, -(r + COST) - COST) for t, r in tr]
        a = agg(fade)
        print(f"  F5fade th={th} h={hold}: n={a['n']} wr={a['wr']} pf={a['pf']} totR={a['totR%']} avg={a['avgR_bp']}bp yr={a['yr']}")


if __name__ == "__main__":
    main()
