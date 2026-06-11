"""v16 진단 — 진입 시점 BTC 거시 레짐별 알트 거래 성과. BTC 추세가 알트 성패를 가르나."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pdm = h.diff().where(lambda x: (x > -l.diff()) & (x > 0), 0.0)
    mdm = (-l.diff()).where(lambda x: (x > h.diff()) & (x > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100*pdm.ewm(alpha=1/n, adjust=False).mean()/atr
    mdi = 100*mdm.ewm(alpha=1/n, adjust=False).mean()/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def main():
    p = yaml.safe_load(open("config/final_v16_slwide.yaml"))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    df = eng.ledger.to_dataframe()

    # BTC 1d 거시 지표
    btc = pd.read_parquet("data/cache/ohlcv_BTCUSDT_1d.parquet")
    btc["timestamp"] = pd.to_datetime(btc["timestamp"]); btc = btc.set_index("timestamp").sort_index()
    btc["ema50"] = btc["close"].ewm(span=50, adjust=False).mean()
    btc["adx"] = adx(btc, 14)
    btc["above"] = btc["close"] > btc["ema50"]  # BTC 상승추세
    btc["ret20"] = btc["close"].pct_change(20)   # 20일 모멘텀

    def at(et, col):
        v = btc[btc.index < pd.Timestamp(et).normalize()]
        return float(v[col].iloc[-1]) if len(v) else np.nan

    df["btc_above"] = [at(t, "above") for t in df.entry_time]
    df["btc_adx"] = [at(t, "adx") for t in df.entry_time]
    df["btc_ret20"] = [at(t, "ret20") for t in df.entry_time]
    df = df.dropna(subset=["btc_above", "btc_adx", "btc_ret20"]).copy()
    df["btc_above"] = df["btc_above"].astype(bool)
    # 진입방향이 BTC추세와 같은가
    df["aligned"] = ((df.direction == "long") & df.btc_above) | ((df.direction == "short") & (~df.btc_above))

    def agg(g):
        gw = g.loc[g.pnl > 0, "pnl"].sum(); gl = -g.loc[g.pnl <= 0, "pnl"].sum()
        return pd.Series({"n": len(g), "wr": round((g.pnl > 0).mean()*100),
                          "pnl합": round(g.pnl.sum()), "PF": round(gw/gl, 2) if gl > 0 else 99})

    print(f"v16 총 {len(df)}거래\n=== BTC추세 정렬 여부 (진입방향 vs BTC EMA50 추세) ===")
    print(df.groupby("aligned").apply(agg, include_groups=False).to_string())
    print("\n=== BTC ADX(거시 추세강도) 버킷 ===")
    df["ab"] = pd.cut(df.btc_adx, [0, 20, 30, 100], labels=["약<20", "중20-30", "강30+"])
    print(df.groupby("ab", observed=True).apply(agg, include_groups=False).to_string())
    print("\n=== BTC 20일 모멘텀 부호 × 진입방향 ===")
    df["btc_up"] = df.btc_ret20 > 0
    print(df.groupby(["btc_up", "direction"]).apply(agg, include_groups=False).to_string())


if __name__ == "__main__":
    main()
