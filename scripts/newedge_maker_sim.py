"""신규 macross-1d 챔피언의 maker-first 집행 시뮬 — 체결률/역선택/폴백 포함 정직 비용.

진입: 1d 시가 O에 post-only 지정가. 1h 경로로 체결 판정(롱: low<O 관통 시 체결, 숏: high>O).
미체결 W시간 경과 → 시장가 폴백(해당 1h 종가, taker+슬립). 체결가 기준 PnL, TP/SL 레벨은
시그널 정의(O 기준 ATR) 유지. 출구비용: TP=지정가(maker), SL/hold=시장가(taker+슬립).

비교축: all-taker 기준(기존 10bp RT) vs maker-first (W ∈ {4,12,24}h), 슬립 민감도.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import atr
from scripts.newedge_l2 import load_full
from scripts.newedge_it2_macross import macross_sig
from scripts.newedge_it3_validate import trades_1d

CACHE = Path(__file__).parent.parent / "data" / "cache"

MAKER = 0.0002
TAKER = 0.0005


def sim_sym(sym, df1d, trs, h1, window_h, slip):
    hts = h1.timestamp.values
    ho, hh, hl, hc = h1.open.values, h1.high.values, h1.low.values, h1.close.values
    ts1d = df1d.timestamp.values
    pos = np.searchsorted(hts, ts1d)
    n1 = len(hts)
    out = []
    for t in trs:
        k0 = pos[t["i"]]
        if k0 >= n1 or hts[min(k0, n1 - 1)] != ts1d[t["i"]]:
            continue
        s, O, tp, sl = t["side"], t["e"], t["tp"], t["sl"]
        # 1) 진입 체결 시뮬
        filled = False
        entry = None
        entry_cost = None
        k_fill = None
        for k in range(k0, min(k0 + window_h, n1)):
            if (s > 0 and hl[k] < O) or (s < 0 and hh[k] > O):
                filled = True
                entry = O
                entry_cost = MAKER
                k_fill = k
                break
        if not filled:
            kf = min(k0 + window_h, n1 - 1)
            entry = hc[kf]
            entry_cost = TAKER + slip
            k_fill = kf
            # 폴백 가격이 SL 너머면 진입 포기 (라이브 가드 가정)
            if (s > 0 and entry >= tp) or (s < 0 and entry <= tp):
                continue
            if (s > 0 and entry <= sl) or (s < 0 and entry >= sl):
                continue
        # 2) 출구: 체결 1h봉 다음부터 TP/SL (SL 우선 비관), hold 60d
        r = None
        exit_cost = None
        kend = k_fill + 60 * 24
        for k in range(k_fill + 1, n1):
            if s > 0:
                if hl[k] <= sl:
                    r = (sl - entry) / entry
                    exit_cost = TAKER + slip
                    break
                if hh[k] >= tp:
                    r = (tp - entry) / entry
                    exit_cost = MAKER
                    break
            else:
                if hh[k] >= sl:
                    r = (entry - sl) / entry
                    exit_cost = TAKER + slip
                    break
                if hl[k] <= tp:
                    r = (entry - tp) / entry
                    exit_cost = MAKER
                    break
            if k >= kend:
                r = (hc[k] - entry) / entry * s
                exit_cost = TAKER + slip
                break
        if r is None:
            r = (hc[-1] - entry) / entry * s
            exit_cost = TAKER + slip
        out.append((ts1d[t["i"]], r - entry_cost - exit_cost, filled))
    return out


def baseline_sym(sym, df1d, trs, h1, slip):
    """all-taker 기준: 시가 시장가 진입(taker+슬립), 출구 동일 로직 1h 경로."""
    hts = h1.timestamp.values
    ho, hh, hl, hc = h1.open.values, h1.high.values, h1.low.values, h1.close.values
    ts1d = df1d.timestamp.values
    pos = np.searchsorted(hts, ts1d)
    n1 = len(hts)
    out = []
    for t in trs:
        k0 = pos[t["i"]]
        if k0 >= n1 or hts[min(k0, n1 - 1)] != ts1d[t["i"]]:
            continue
        s, tp, sl = t["side"], t["tp"], t["sl"]
        entry = ho[k0]
        r = None
        exit_cost = None
        kend = k0 + 60 * 24
        for k in range(k0, n1):
            if s > 0:
                if hl[k] <= sl:
                    r = (sl - entry) / entry
                    exit_cost = TAKER + slip
                    break
                if hh[k] >= tp:
                    r = (tp - entry) / entry
                    exit_cost = TAKER + slip
                    break
            else:
                if hh[k] >= sl:
                    r = (entry - sl) / entry
                    exit_cost = TAKER + slip
                    break
                if hl[k] <= tp:
                    r = (entry - tp) / entry
                    exit_cost = TAKER + slip
                    break
            if k >= kend:
                r = (hc[k] - entry) / entry * s
                exit_cost = TAKER + slip
                break
        if r is None:
            r = (hc[-1] - entry) / entry * s
            exit_cost = TAKER + slip
        out.append((ts1d[t["i"]], r - (TAKER + slip) - exit_cost, True))
    return out


def report(tag, rows):
    t = pd.DataFrame(rows, columns=["t", "r", "filled"])
    t["y"] = pd.to_datetime(t.t).dt.year
    fill = t.filled.mean() * 100
    line = " ".join(
        f"{y}:{round(t[t.y==y].r.mean()*10000) if len(t[t.y==y]) else 0:>5}" for y in (2022, 2023, 2024, 2025, 2026))
    print(f"{tag:<28} n={len(t):>5} maker체결={fill:5.1f}% avg={round(t.r.mean()*10000):>5}bp | {line}")
    return t.r.mean()


def main():
    slip = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0005
    dfs = load_full("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"slip(taker측)={slip*10000:.0f}bp, maker={MAKER*10000:.0f}bp, taker={TAKER*10000:.0f}bp")
    all_trs = {}
    h1map = {}
    for sym, df in dfs.items():
        sig = macross_sig(df, 20, 100)
        trs = trades_1d(df, sig, 6.0, 3.0)
        if not trs:
            continue
        f = CACHE / f"ohlcv_{sym}_1h.parquet"
        if not f.exists():
            continue
        h1 = pd.read_parquet(f)
        h1map[sym] = h1[h1.timestamp <= "2026-04-23"].reset_index(drop=True)
        all_trs[sym] = trs

    rows = []
    for sym, trs in all_trs.items():
        rows += baseline_sym(sym, dfs[sym], trs, h1map[sym], slip)
    base = report("all-taker (기준)", rows)

    for W in (4, 12, 24):
        rows = []
        for sym, trs in all_trs.items():
            rows += sim_sym(sym, dfs[sym], trs, h1map[sym], W, slip)
        avg = report(f"maker-first W={W}h", rows)
        print(f"    → 절감: {round((avg-base)*10000):+d}bp/건")


if __name__ == "__main__":
    main()
