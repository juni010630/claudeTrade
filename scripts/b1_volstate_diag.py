"""B1 진단 — 변동성/유동성 '상태'가 v16 거래품질을 가르나 + IS↔OOS 지속되나.
시간대(per-hour corr~0=실패)와 달리 상태기반이 지속되면 robust 필터 후보.
무차단 v16 거래를 진입봉 상태(실현변동성, 거래량비)로 버킷, 전반/후반 따로."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def atr_pct(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return (tr.ewm(alpha=1/n, adjust=False).mean() / c)


def get_trades():
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open("config/final_v16_slwide.yaml")))
    p.pop("strategy_block_hours", None)
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    rows = [(r.symbol, r.strategy, pd.Timestamp(r.entry_time), r.pnl, r.size_usd)
            for r in eng.ledger.records]
    return pd.DataFrame(rows, columns=["symbol", "strategy", "entry_time", "pnl", "size_usd"]), p["symbols"]


def state_series(sym):
    """심볼 1h봉의 진입시점 상태: 실현변동성 백분위, 거래량비."""
    d = pd.read_parquet(f"data/cache/ohlcv_{sym}_1h.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.set_index("timestamp").sort_index()
    ap = atr_pct(d, 14)
    vol_pctile = ap.rolling(200).apply(lambda x: (x.iloc[-1] >= x).mean(), raw=False)
    vr = d["volume"] / d["volume"].rolling(20).mean()
    return vol_pctile, vr


def bucket(df, col):
    """col 3분위별 거래당 평균손익(R 근사=pnl/size) + 건수."""
    df = df.dropna(subset=[col])
    if len(df) < 30:
        return None
    df = df.copy()
    df["rr"] = df["pnl"] / df["size_usd"]
    q = df[col].quantile([1/3, 2/3]).values
    out = []
    for name, mask in [("저", df[col] <= q[0]), ("중", (df[col] > q[0]) & (df[col] <= q[1])), ("고", df[col] > q[1])]:
        s = df[mask]
        out.append((name, len(s), s.rr.mean()*100, (s.rr > 0).mean()*100))
    return out


def main():
    tr, syms = get_trades()
    print(f"무차단 v16 거래 {len(tr)}건\n")

    # 심볼별 상태 매핑
    vps, vrs = {}, {}
    for s in syms:
        vps[s], vrs[s] = state_series(s)
    def lookup(row, series_map):
        s = series_map[row.symbol]
        idx = s.index.asof(row.entry_time)
        return s.loc[idx] if idx is not None and not pd.isna(idx) else np.nan
    tr["volpct"] = tr.apply(lambda r: lookup(r, vps), axis=1)
    tr["volratio"] = tr.apply(lambda r: lookup(r, vrs), axis=1)

    h1 = tr[tr.entry_time < "2024-01-01"]
    h2 = tr[tr.entry_time >= "2024-01-01"]

    for col, label in [("volpct", "실현변동성 백분위"), ("volratio", "거래량비(vol/SMA20)")]:
        print(f"=== {label} 버킷별 거래당 평균손익%(R근사) / 승률 ===")
        for tag, sub in [("전체", tr), ("전반(IS)", h1), ("후반(OOS)", h2)]:
            b = bucket(sub, col)
            if b is None:
                print(f"  {tag}: 표본부족"); continue
            cells = " | ".join(f"{n}:{c}건 {m:+.1f}% 승{w:.0f}%" for n, c, m, w in b)
            print(f"  {tag:9} {cells}")
        print("  → 전반/후반에서 같은 방향(저<고 or 고<저)이면 지속=robust, 뒤집히면 노이즈\n")


if __name__ == "__main__":
    main()
