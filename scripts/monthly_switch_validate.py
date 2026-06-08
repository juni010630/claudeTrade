"""월간 배분 스위치 과적합 검증 — 인과 임계(트레일링 중앙값) + H1/H2 walk-forward.
고정 0.22(in-sample) 대신 트레일링 ER 중앙값으로 추세/횡보 판정(look-ahead 없음).
양쪽 반쪽에서 정적 50:50 대비 개선되면 robust. 스킴 C/B + 가중 민감도."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
SER = "data/results/funding3_sleeve_daily.parquet"


def run(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    d = eng.equity_curve.to_series().resample("1D").last().pct_change().fillna(0)
    d.index = d.index.tz_localize(None)
    return cfg, d


def er(c, n=20):
    ch = (c - c.shift(n)).abs(); vol = c.diff().abs().rolling(n).sum()
    return ch/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def get_series():
    if Path(SER).exists():
        df = pd.read_parquet(SER); return df["v"], df["s"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        res = dict(ex.map(run, [("config/final_v16_slwide.yaml", "1h"),
                                ("config/sleeve_meanrev.yaml", "1d")]))
    v = res["config/final_v16_slwide.yaml"]; s = res["config/sleeve_meanrev.yaml"]
    df = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()
    Path("data/results").mkdir(exist_ok=True); df.to_parquet(SER)
    return df["v"], df["s"]


def monthly_blend(v, s, wv_of_month):
    months = v.index.to_period("M"); eq = pd.Series(index=v.index, dtype=float); total = 1.0
    for m in months.unique():
        mask = months == m; wv = wv_of_month(m)
        seg = wv*(1+v[mask]).cumprod() + (1-wv)*(1+s[mask]).cumprod()
        eq[mask] = total*seg.values; total = eq[mask].iloc[-1]
    return eq


def st(eq):
    if len(eq) < 2: return (0, 0, 0)
    daily = eq.pct_change().fillna(0)
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    return eq.iloc[-1]/eq.iloc[0]*100-100, sh, mdd


def slice_eq(eq, s, e):
    seg = eq[(eq.index >= s) & (eq.index < e)]
    return seg/seg.iloc[0] if len(seg) else seg


def main():
    v, s = get_series()
    j = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()
    be = pd.concat([er(load_close(x), 20) for x in SLEEVE], axis=1).mean(axis=1)
    if be.index.tz is not None: be.index = be.index.tz_localize(None)
    be = be.reindex(j.v.index, method="ffill")

    # 트레일링 중앙값(252d, 인과) — 전월말 ER vs 그 시점까지 트레일링 중앙값
    trail_med = be.rolling(252, min_periods=60).median()
    months = j.v.index.to_period("M")
    sig = {}  # month -> (전월말 ER, 그 시점 트레일링중앙값)
    for m in months.unique():
        prev_end = j.v.index[months == m][0] - pd.Timedelta(days=1)
        pr = be[be.index <= prev_end]; tm = trail_med[trail_med.index <= prev_end]
        sig[m] = (pr.iloc[-1] if len(pr) else be.iloc[0],
                  tm.dropna().iloc[-1] if len(tm.dropna()) else be.median())

    def wv_causal(scheme):
        def f(m):
            erv, med = sig[m]
            trending = erv > med
            if scheme == "C": return 0.7 if trending else 0.3
            if scheme == "B": return 0.8 if trending else 0.5
        return f

    print("=== 월간 스위치 인과임계(트레일링 중앙값) 검증 ===")
    statics = monthly_blend(j.v, j.s, lambda m: 0.5)
    cC = monthly_blend(j.v, j.s, wv_causal("C"))
    cB = monthly_blend(j.v, j.s, wv_causal("B"))

    def line(tag, eq):
        for span, (a, b) in [("full", ("2022-01-01", "2026-04-23")),
                             ("H1", ("2022-01-01", "2024-01-01")),
                             ("H2", ("2024-01-01", "2026-04-23"))]:
            seg = slice_eq(eq, a, b); r, sh, mdd = st(seg)
            print(f"    {tag:18}[{span:4}] 수익{r:>+7.0f}% Sh{sh:.2f} MDD{mdd:>4.0f}%")

    line("정적 50:50", statics)
    print()
    line("스위치C(인과)", cC)
    print()
    line("스위치B(인과)", cB)
    print("\n판정: 스위치가 H1·H2 양쪽서 정적 대비 MDD↓ 또는 수익↑ 일관 = robust(과적합X).")


if __name__ == "__main__":
    main()
