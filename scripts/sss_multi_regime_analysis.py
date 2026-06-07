"""sss_multi_regime_analysis.py — multi_tf×SSS 거래를 거시국면(1d ADX)으로 분리 가능한지 사전 검증.

각 거래 진입 시점의 ETH 1d ADX(14)를 '전일 마감봉까지'로 계산해 붙임 (look-ahead 없음).
2023 손실군과 2024/25 수익군이 ADX 임계로 분리되면 → 조건부 게이트 구현 가치 있음.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = h.diff().where(lambda x: (x > -l.diff()) & (x > 0), 0.0)
    minus_dm = (-l.diff()).where(lambda x: (x > h.diff()) & (x > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


ledger = pd.read_parquet("/tmp/v15_full_ledger.parquet")
ledger["tier"] = ledger.confluence_score.map(
    lambda s: "SSS" if s >= 6 else "SS" if s >= 5 else "S" if s >= 4 else "A")
ledger["year"] = pd.to_datetime(ledger.entry_time).dt.year.clip(upper=2025)

eth1d = pd.read_parquet("data/cache/ohlcv_ETHUSDT_1d.parquet")
eth1d["timestamp"] = pd.to_datetime(eth1d["timestamp"], utc=True)
eth1d = eth1d.set_index("timestamp").sort_index()
adx1d = adx_series(eth1d)

def adx_at_entry(t):
    """진입 시각 기준 '완결된' 마지막 1d 봉의 ADX (당일 봉 제외 → look-ahead 없음)."""
    t = pd.Timestamp(t)
    done = adx1d[adx1d.index < t.normalize()]  # 전일 이하 마감봉
    return done.iloc[-1] if len(done) else np.nan

ledger["adx1d"] = [adx_at_entry(t) for t in ledger.entry_time]

target = ledger[(ledger.strategy == "multi_tf_breakout") & (ledger.tier == "SSS")].copy()
print(f"multi×SSS 거래: {len(target)}건\n")

def agg(g):
    gw = g.loc[g.pnl > 0, "pnl"].sum(); gl = -g.loc[g.pnl <= 0, "pnl"].sum()
    return pd.Series({"n": len(g), "wr": round((g.pnl > 0).mean() * 100, 0),
                      "pnl": round(g.pnl.sum(), 1), "PF": round(gw / gl, 2) if gl > 0 else 99})

print("== 연도 × 1d ADX 분포 (multi×SSS) ==")
print(target.groupby("year")["adx1d"].describe()[["count", "mean", "25%", "50%", "75%"]].round(1).to_string())

print("\n== 1d ADX 버킷별 성과 (multi×SSS, 전 기간) ==")
target["adx_bucket"] = pd.cut(target.adx1d, [0, 15, 20, 25, 30, 100],
                              labels=["<15", "15-20", "20-25", "25-30", "30+"])
print(target.groupby("adx_bucket", observed=True).apply(agg, include_groups=False).to_string())

print("\n== 임계값별: 게이트 통과(ADX≥k) vs 차단 — 연도별 pnl 합 ==")
for k in [15, 18, 20, 22, 25]:
    keep = target[target.adx1d >= k]; cut = target[target.adx1d < k]
    by = lambda d: {y: round(d[d.year == y].pnl.sum(), 0) for y in [2022, 2023, 2024, 2025]}
    print(f"ADX≥{k}: 통과 {len(keep)}건 {by(keep)} | 차단 {len(cut)}건 {by(cut)}")
