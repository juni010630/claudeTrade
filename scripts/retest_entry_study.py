"""추세 진입 되돌림(retest) 지정가 스터디 — WR/기대값 vs 러너 유실 정량화.

v17 추세 트레이드(ema_cross/multi_tf)의 시그널가(order_price) 기준,
지정가 = 시그널가 - 방향×k×ATR(1h,14) 를 W시간 대기 → 체결 시 원래 TP/SL로 1h 경로 청산,
미체결 = 트레이드 유실(수익 0). 비교 기준 = 동일 하니스의 시장가 진입(taker).

체결 판정 보수: 롱은 low < limit (터치≠체결), maker 2bp / 시장가 taker5+슬립5bp, 청산은 양쪽 taker+슬립.
TP/SL은 시그널가 앵커 유지 (ema 3.5/1.8, multi 4.0/2.1 ATR).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

CACHE = Path(__file__).parent.parent / "data" / "cache"
MAKER, TAKER, SLIP = 0.0002, 0.0005, 0.0005
MULTS = {"ema_cross": (3.5, 1.8), "multi_tf_breakout": (4.0, 2.1)}
MAX_HOLD_H = 336


def atr_series(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def simulate_exit(arr, j0, side, entry, tp, sl, jmax):
    o, h, l, c = arr
    n = len(o)
    j = j0
    while j < n:
        if side > 0:
            if l[j] <= sl:
                return (sl - entry) / entry, j
            if h[j] >= tp:
                return (tp - entry) / entry, j
        else:
            if h[j] >= sl:
                return (entry - sl) / entry, j
            if l[j] <= tp:
                return (entry - tp) / entry, j
        if j >= jmax:
            return (c[j] - entry) / entry * side, j
        j += 1
    return (c[-1] - entry) / entry * side, n - 1


def main():
    t = pd.read_csv(Path(__file__).parent.parent / "data" / "results" / "v17_trades_full.csv",
                    parse_dates=["entry_time"])
    tr = t[t.strategy.isin(MULTS)].copy()
    print(f"추세 트레이드 {len(tr)}건")

    h1, atrs, idx = {}, {}, {}
    for sym in tr.symbol.unique():
        df = pd.read_parquet(CACHE / f"ohlcv_{sym}_1h.parquet")
        df = df[df.timestamp <= "2026-04-23"].reset_index(drop=True)
        h1[sym] = (df.open.values, df.high.values, df.low.values, df.close.values)
        atrs[sym] = atr_series(df).values
        idx[sym] = {ts: i for i, ts in enumerate(df.timestamp)}

    rows = []
    for _, r in tr.iterrows():
        sym = r.symbol
        k0 = idx[sym].get(r.entry_time)
        if k0 is None or k0 < 20:
            continue
        a = atrs[sym][k0]
        if not np.isfinite(a) or a <= 0:
            continue
        side = 1 if r.direction == "long" else -1
        sig_p = r.order_price
        tp_m, sl_m = MULTS[r.strategy]
        tp = sig_p + side * tp_m * a
        sl = sig_p - side * sl_m * a
        jmax = k0 + MAX_HOLD_H
        arr = h1[sym]
        o, h, l, c = arr
        n = len(o)
        # baseline: 시장가 즉시 (다음 봉부터 청산 체크 — 엔진과 동일하게 진입봉 이후)
        ret_b, _ = simulate_exit(arr, k0 + 1, side, sig_p, tp, sl, jmax)
        ret_b -= (TAKER + SLIP) + (TAKER + SLIP)
        row = dict(year=r.entry_time.year, strategy=r.strategy, base=ret_b)
        for k in (0.0, 0.15, 0.3, 0.5):
            for W in (6, 24):
                lim = sig_p - side * k * a
                fill_j = None
                for j in range(k0 + 1, min(k0 + 1 + W, n)):
                    if (side > 0 and l[j] < lim) or (side < 0 and h[j] > lim):
                        fill_j = j
                        break
                if fill_j is None:
                    row[f"k{k}_W{W}"] = np.nan  # 미체결
                    continue
                ret, _ = simulate_exit(arr, fill_j, side, lim, tp, sl, jmax)
                # 체결봉 내 SL 선행 가능성 보수 처리: 체결봉에서 SL도 닿았으면 SL 우선
                if (side > 0 and l[fill_j] <= sl) or (side < 0 and h[fill_j] >= sl):
                    ret = (sl - lim) / lim * side
                row[f"k{k}_W{W}"] = ret - MAKER - (TAKER + SLIP)
        rows.append(row)

    d = pd.DataFrame(rows)
    print(f"\n분석 표본 {len(d)}건 | baseline: WR={(d.base>0).mean()*100:.1f}% avg={d.base.mean()*10000:.0f}bp")
    print(f"{'변형':<12}{'체결%':>7}{'WR(체결)':>9}{'avg/체결':>9}{'avg/시그널':>10}  연도별 avg/시그널(bp)")
    for k in (0.0, 0.15, 0.3, 0.5):
        for W in (6, 24):
            col = d[f"k{k}_W{W}"]
            fill = col.notna().mean() * 100
            wr = (col.dropna() > 0).mean() * 100
            avg_f = col.dropna().mean() * 10000
            per_sig = col.fillna(0)
            yline = " ".join(f"{y}:{per_sig[d.year==y].mean()*10000:>4.0f}" for y in (2022, 2023, 2024, 2025, 2026))
            print(f"k={k} W={W:<4}{fill:>6.1f}%{wr:>8.1f}%{avg_f:>8.0f}bp{per_sig.mean()*10000:>9.0f}bp  {yline}")
    yb = " ".join(f"{y}:{d.base[d.year==y].mean()*10000:>4.0f}" for y in (2022, 2023, 2024, 2025, 2026))
    print(f"{'baseline':<12}{'100.0%':>7}{(d.base>0).mean()*100:>8.1f}%{d.base.mean()*10000:>8.0f}bp{d.base.mean()*10000:>9.0f}bp  {yb}")


if __name__ == "__main__":
    main()
