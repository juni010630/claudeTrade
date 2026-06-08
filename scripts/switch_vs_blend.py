"""① v16+슬리브 50:50 혼합 분기별 ② 3일/7일 ADX 스위칭 백테."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml


def bt(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return eng.equity_curve.to_series()


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


def sharpe(r): return r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0
def mdd(r):
    c = (1+r).cumprod(); return ((c-c.cummax())/c.cummax()).min()*100
def totret(r):
    return ((1+r).prod()-1)*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        e16, esl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                    ("config/sleeve_meanrev.yaml", "1d")]))
    d16 = e16.resample("1D").last().pct_change()
    dsl = esl.resample("1D").last().pct_change()
    j = pd.concat([d16, dsl], axis=1, keys=["v16", "sl"]).dropna()

    # ① 50:50 혼합 분기별
    blend = 0.5*j.v16 + 0.5*j.sl
    print("=== ① v16+슬리브 50:50 혼합 — 분기별 수익률(%) ===")
    q = pd.DataFrame({
        "v16": j.v16.resample("QE").apply(totret),
        "슬리브": j.sl.resample("QE").apply(totret),
        "혼합50:50": blend.resample("QE").apply(totret),
    }).round(1)
    q.index = [f"{t.year}Q{t.quarter}" for t in q.index]
    print(q.to_string())
    print(f"\n전체: v16 Sharpe {sharpe(j.v16):.2f}/MDD {mdd(j.v16):.0f}% | "
          f"슬리브 {sharpe(j.sl):.2f}/{mdd(j.sl):.0f}% | 혼합 {sharpe(blend):.2f}/{mdd(blend):.0f}%")

    # ② ADX 스위칭 (ETH 1d ADX, 임계 25, 3일/7일 리밸런스)
    eth = pd.read_parquet("data/cache/ohlcv_ETHUSDT_1d.parquet")
    eth["timestamp"] = pd.to_datetime(eth["timestamp"]); eth = eth.set_index("timestamp").sort_index()
    a = adx(eth, 14).reindex(j.index, method="ffill")
    print("\n=== ② ADX 스위칭 (ETH 1d ADX>25→v16 추세, ≤25→슬리브) vs 혼합 ===")
    for thr in [22, 25, 28]:
        for reb in [3, 7]:
            # reb일마다 국면 갱신
            regime = (a > thr)
            grp = (np.arange(len(j)) // reb)
            reg_reb = regime.groupby(grp).transform("first")  # 그룹 첫날 국면 유지
            switched = np.where(reg_reb.values, j.v16.values, j.sl.values)
            sr = pd.Series(switched, index=j.index)
            print(f"  ADX>{thr} {reb}일리밸런스: 총수익 {totret(sr):+.0f}% Sharpe {sharpe(sr):.2f} MDD {mdd(sr):.0f}% "
                  f"(추세구간 {reg_reb.mean()*100:.0f}%)")
    print(f"  [기준] 혼합50:50: 총수익 {totret(blend):+.0f}% Sharpe {sharpe(blend):.2f} MDD {mdd(blend):.0f}%")
    print(f"  [기준] v16단독:   총수익 {totret(j.v16):+.0f}% Sharpe {sharpe(j.v16):.2f} MDD {mdd(j.v16):.0f}%")


if __name__ == "__main__":
    main()
