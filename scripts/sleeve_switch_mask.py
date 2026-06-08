"""스위치 상한 테스트 — 엔진 실제 일별수익에 'basket ER>thr 날엔 플랫' 마스크.
진입게이트가 아니라 노출 자체 제거. Sharpe/MDD/연도 + v16 블렌드 개선 여부.
이상적 상한(무비용 즉시 플랫 가정) — 개선 없으면 스위치 자체 기각."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def er(c, n=20):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def run_engine(cfg, ptf):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return eng.equity_curve.to_series()


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def main():
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        fsl = ex.submit(run_engine, "config/sleeve_meanrev.yaml", "1d")
        fv16 = ex.submit(run_engine, "config/final_v16_slwide.yaml", "1h")
        esl, ev16 = fsl.result(), fv16.result()

    dsl = esl.resample("1D").last().pct_change().fillna(0)
    dsl.index = dsl.index.tz_localize(None)
    dv16 = ev16.resample("1D").last().pct_change().fillna(0)
    dv16.index = dv16.index.tz_localize(None)

    basket_er = pd.concat([er(load_close(s), 20) for s in SLEEVE], axis=1).mean(axis=1).shift(1)
    if basket_er.index.tz is not None:
        basket_er.index = basket_er.index.tz_localize(None)
    basket_er = basket_er.reindex(dsl.index, method="ffill")

    f, sh, mdd, yr = stats(dsl)
    print("=== 슬리브 always-on (엔진 실제) ===")
    print(f"  ${f:,.0f} Sh{sh:.2f} MDD{mdd:.0f}% 연도{ {int(k): round(v) for k,v in yr.items()} }")

    print("\n=== 'basket ER(20)>thr 날엔 슬리브 플랫' 마스크 스윕 ===")
    print(f"{'thr분위':>7} {'ER':>5} {'플랫일%':>6} {'$':>9} {'Sh':>5} {'MDD%':>6}  연도별")
    for q in [0.50, 0.60, 0.67, 0.75, 0.80, 0.90]:
        thr = basket_er.quantile(q)
        on = basket_er <= thr
        masked = dsl.where(on, 0.0)
        f, sh, mdd, yr = stats(masked)
        flat_pct = (~on).mean()*100
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
        print(f"{q:>7.2f} {thr:>5.2f} {flat_pct:>6.0f} {f:>9,.0f} {sh:>5.2f} {mdd:>6.0f}  {ys}")

    # 블렌드(75:25) 개선 여부
    print("\n=== v16:슬리브 75:25 블렌드, 스위치 적용 비교 ===")
    j = pd.concat([dv16, dsl], axis=1, keys=["v", "s"]).dropna()
    er_j = basket_er.reindex(j.index, method="ffill")
    f, sh, mdd, _ = stats(0.75*j.v + 0.25*j.s)
    print(f"  스위치 OFF(always-on): ${f:,.0f} Sh{sh:.2f} MDD{mdd:.0f}%")
    for q in [0.67, 0.75, 0.80]:
        thr = er_j.quantile(q)
        s_on = j.s.where(er_j <= thr, 0.0)
        f, sh, mdd, _ = stats(0.75*j.v + 0.25*s_on)
        print(f"  스위치 ON q{q:.2f}(ER<{thr:.2f}): ${f:,.0f} Sh{sh:.2f} MDD{mdd:.0f}%")


if __name__ == "__main__":
    main()
