"""v16 진단 — 진입 시점 시장 변동성(ETH ann.vol)별 거래 성과. 고변동 구간이 나쁜가?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def main():
    p = yaml.safe_load(open("config/final_v16_slwide.yaml"))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    df = eng.ledger.to_dataframe()

    # ETH 1d 실현변동성 (30일 ann.) — 시장 전체 변동성 프록시
    eth = pd.read_parquet("data/cache/ohlcv_ETHUSDT_1d.parquet")
    eth["timestamp"] = pd.to_datetime(eth["timestamp"]); eth = eth.set_index("timestamp").sort_index()
    ret = eth["close"].pct_change()
    vol30 = ret.rolling(30).std()*np.sqrt(365)*100  # ann. vol %

    # 각 거래 진입 시점 직전 vol (look-ahead 없음)
    vols = []
    for _, t in df.iterrows():
        et = pd.Timestamp(t.entry_time).normalize()
        v = vol30[vol30.index < et]
        vols.append(float(v.iloc[-1]) if len(v) else np.nan)
    df["vol"] = vols

    def agg(g):
        gw = g.loc[g.pnl > 0, "pnl"].sum(); gl = -g.loc[g.pnl <= 0, "pnl"].sum()
        return pd.Series({"n": len(g), "wr": round((g.pnl > 0).mean()*100),
                          "pnl합": round(g.pnl.sum()), "평균pnl": round(g.pnl.mean()),
                          "PF": round(gw/gl, 2) if gl > 0 else 99})

    print(f"v16 총 {len(df)}거래\n=== 진입시점 ETH 30일 연환산 변동성 버킷별 성과 ===")
    df["vb"] = pd.qcut(df.vol, 5, labels=["최저", "저", "중", "고", "최고"])
    print(df.groupby("vb", observed=True).apply(agg, include_groups=False).to_string())
    print(f"\n변동성 분위 경계: {[round(x) for x in df.vol.quantile([0,.2,.4,.6,.8,1]).tolist()]}")
    # 변동성 vs 평균 절대수익(변동성타게팅 근거: 고변동=거래 변동성 큼?)
    print(f"\n거래 pnl 표준편차 (변동성 버킷별 — 높을수록 타게팅 효과 큼):")
    print(df.groupby("vb", observed=True).pnl.std().round(0).to_string())


if __name__ == "__main__":
    main()
