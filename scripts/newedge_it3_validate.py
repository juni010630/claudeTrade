"""반복 3 — macross-1d 후보 검증: 1h 해상도 교차검증 + 롱/숏 분해.

1d 신호·진입은 고정한 채, 청산 경로만 1h 봉으로 재시뮬 → 1d 봉내 모호성 부풀림 측정.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import COST, atr, ema
from scripts.newedge_l2 import load_full
from scripts.newedge_it2_macross import macross_sig, HOLD

CACHE = Path(__file__).parent.parent / "data" / "cache"


def trades_1d(df, sig, tp_mult, sl_mult):
    """1d 시뮬 — (entry_idx, side, entry, tp, sl, r_1d, exit_idx) 기록."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    a = df["__atr"].values
    n = len(o)
    out = []
    pos_end = 0
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
            if j - i + 1 >= HOLD:
                r = (c[j] - e) / e * s
                break
            j += 1
        if r is None:
            r = (c[-1] - e) / e * s
            j = n - 1
        out.append(dict(i=i, side=s, e=e, tp=tp, sl=sl, r1d=r - COST, xi=j))
        pos_end = j
    return out


def xval_1h(sym, df1d, trs):
    """동일 트레이드를 1h 경로로 청산 재시뮬."""
    f = CACHE / f"ohlcv_{sym}_1h.parquet"
    if not f.exists():
        return None
    h1 = pd.read_parquet(f)
    h1 = h1[h1.timestamp <= "2026-04-23"]
    hts = h1.timestamp.values
    ho, hh, hl, hc = h1.open.values, h1.high.values, h1.low.values, h1.close.values
    ts1d = df1d.timestamp.values
    pos = np.searchsorted(hts, ts1d)  # 각 1d봉 시작 시각의 1h 인덱스
    out = []
    n1 = len(hts)
    for t in trs:
        k0 = pos[t["i"]]
        if k0 >= n1 or hts[min(k0, n1 - 1)] != ts1d[t["i"]]:
            continue  # 1h 데이터 결손
        kmax = pos[t["xi"]] + 24 if t["xi"] < len(ts1d) else n1 - 1
        kend_hold = k0 + HOLD * 24
        s, e, tp, sl = t["side"], ho[k0], t["tp"], t["sl"]
        # 진입가: 1d 시가 = 해당 1h 시가 (동일 시각) — 차이는 무시 수준이나 1h 시가 사용
        r = None
        k = k0
        while k < n1:
            if s > 0:
                if hl[k] <= sl:
                    r = (sl - e) / e
                    break
                if hh[k] >= tp:
                    r = (tp - e) / e
                    break
            else:
                if hh[k] >= sl:
                    r = (e - sl) / e
                    break
                if hl[k] <= tp:
                    r = (e - tp) / e
                    break
            if k >= kend_hold:
                r = (hc[k] - e) / e * s
                break
            k += 1
        if r is None:
            r = (hc[-1] - e) / e * s
        out.append((t["r1d"], r - COST))
    return out


def main():
    f_, s_, tp_, sl_ = 20, 100, 6.0, 3.0
    if len(sys.argv) > 4:
        f_, s_, tp_, sl_ = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    dfs = load_full("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    pairs = []
    sides = []
    for sym, df in dfs.items():
        sig = macross_sig(df, f_, s_)
        trs = trades_1d(df, sig, tp_, sl_)
        for t in trs:
            sides.append((df.timestamp.values[t["i"]], t["side"], t["r1d"]))
        xv = xval_1h(sym, df, trs)
        if xv:
            pairs += xv
    a = np.array(pairs)
    print(f"== ema{f_}x{s_} tp{tp_}/sl{sl_} 1h 교차검증 (n={len(a)}) ==")
    print(f"1d 경로 avg: {a[:,0].mean()*10000:.0f}bp | 1h 경로 avg: {a[:,1].mean()*10000:.0f}bp | "
          f"괴리: {(a[:,0].mean()-a[:,1].mean())*10000:.1f}bp/건")
    diff = np.abs(a[:, 0] - a[:, 1])
    print(f"per-trade 불일치율(|Δr|>1bp): {(diff>0.0001).mean()*100:.1f}%  max|Δ|={diff.max()*10000:.0f}bp")

    sd = pd.DataFrame(sides, columns=["t", "side", "r"])
    sd["y"] = pd.to_datetime(sd.t).dt.year
    print("\n== 롱/숏 연도별 avg bp ==")
    for s in (1, -1):
        sub = sd[sd.side == s]
        line = " ".join(
            f"{y}:{round(sub[sub.y==y].r.mean()*10000) if len(sub[sub.y==y]) else 0:>5}(n={len(sub[sub.y==y])})"
            for y in (2022, 2023, 2024, 2025, 2026))
        print(f"  {'L' if s>0 else 'S'}: {line}")


if __name__ == "__main__":
    main()
