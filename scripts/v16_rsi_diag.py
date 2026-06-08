"""v16 거래를 진입 시점 RSI(1h)/ER(1d)별로 갈라 성과 진단 — 과열 진입이 나쁜가."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def rsi(c, n=14):
    dd = c.diff(); up = dd.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-dd.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100-100/(1+up/dn)


def main():
    p = yaml.safe_load(open("config/final_v16_slwide.yaml"))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    df = eng.ledger.to_dataframe()

    # 각 거래 진입 시점 1h RSI, 1d ER 계산 (entry_time까지 — look-ahead 없음)
    cache = {}
    def bars(sym, tf):
        k = (sym, tf)
        if k not in cache:
            d = pd.read_parquet(f"data/cache/ohlcv_{sym}_{tf}.parquet")
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            cache[k] = d.set_index("timestamp").sort_index()
        return cache[k]

    rsis, ers = [], []
    for _, t in df.iterrows():
        et = pd.Timestamp(t.entry_time)
        h1 = bars(t.symbol, "1h"); h1 = h1[h1.index <= et]
        rv = float(rsi(h1["close"], 14).iloc[-1]) if len(h1) > 20 else np.nan
        # 롱이면 RSI 그대로, 숏이면 100-RSI (방향 정규화: 높을수록 "내 방향으로 과열")
        rsis.append(rv if t.direction == "long" else 100-rv)
        d1 = bars(t.symbol, "1d"); d1 = d1[d1.index <= et]["close"]
        er = (abs(d1.iloc[-1]-d1.iloc[-31])/d1.diff().abs().iloc[-30:].sum()) if len(d1) > 31 else np.nan
        ers.append(er)
    df["rsi_dir"] = rsis  # 진입방향 기준 RSI (높을수록 이미 과열 진입)
    df["er"] = ers

    def agg(g):
        gw = g.loc[g.pnl > 0, "pnl"].sum(); gl = -g.loc[g.pnl <= 0, "pnl"].sum()
        return pd.Series({"n": len(g), "wr": round((g.pnl > 0).mean()*100),
                          "pnl합": round(g.pnl.sum()), "PF": round(gw/gl, 2) if gl > 0 else 99})

    print(f"v16 총 {len(df)}거래\n=== 진입 방향기준 RSI 버킷 (높을수록 과열 진입) ===")
    df["rb"] = pd.cut(df.rsi_dir, [0, 40, 50, 60, 70, 100], labels=["<40", "40-50", "50-60", "60-70", "70+"])
    print(df.groupby("rb", observed=True).apply(agg, include_groups=False).to_string())
    print("\n=== 진입 ER 버킷 (낮을수록 횡보) ===")
    df["eb"] = pd.cut(df.er, [0, 0.1, 0.2, 0.3, 1.0], labels=["<0.1", "0.1-0.2", "0.2-0.3", "0.3+"])
    print(df.groupby("eb", observed=True).apply(agg, include_groups=False).to_string())


if __name__ == "__main__":
    main()
