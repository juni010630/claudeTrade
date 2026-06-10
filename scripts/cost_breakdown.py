"""A축 집행비용 — 비용 분해 + 슬리피지 스트레스 (프로덕션 무수정, 분석 전용).

스트레스: v17의 default_slippage_bps {5,15,25,35} × {전구간, OOS 2025~} 8런 병렬.
엔진 슬리피지는 MARKET 체결 전체(진입+SL/시간청산)에 적용 → 보수적 스트레스.
분해: 5bps 전구간 런의 ledger에서 commission/slippage/funding vs gross PnL, 연도별.
부산물: data/results/v17_trades_full.csv (A2 maker 연구 입력).

사용: python scripts/cost_breakdown.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
TRADES_OUT = "data/results/v17_trades_full.csv"
END = "2026-04-14"
BPS_LIST = [5.0, 15.0, 25.0, 35.0]


def run_one(args):
    bps, start = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    p["execution"]["default_slippage_bps"] = bps
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(start, tz="UTC"),
                           until=pd.Timestamp(END, tz="UTC")))

    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    sh = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    mdd_bar = (eq / eq.cummax() - 1).min() * 100
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    df = eng.ledger.to_dataframe()
    trades = df if (bps == 5.0 and start == "2022-01-01") else None
    return (bps, start), eq.iloc[-1], sh, mdd_bar, len(df), dict(yearly), trades


def main():
    jobs = [(b, s) for s in ("2022-01-01", "2025-01-01") for b in BPS_LIST]
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_one, jobs))

    print("=== 슬리피지 스트레스 (v17, $100 시작) ===")
    base_trades = None
    for (bps, start), final, sh, mdd, n, yearly, trades in results:
        tag = "전구간" if start == "2022-01-01" else "OOS25~"
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
        print(f"  {tag} slip{bps:>4.0f}bps  ${final:>9,.0f} Sh{sh:5.2f} MDD봉{mdd:6.1f}% {n}건  {ys}")
        if trades is not None:
            base_trades = trades

    t = base_trades
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    Path("data/results").mkdir(exist_ok=True)
    t.to_csv(TRADES_OUT, index=False)

    print(f"\n=== 비용 분해 (5bps 전구간, {len(t)}건 → {TRADES_OUT}) ===")
    gross = t["pnl"] + t["commission"] + t["slippage_cost"] + t["funding_paid"]
    print(f"  gross PnL  ${gross.sum():>10,.0f}")
    print(f"  commission ${t['commission'].sum():>10,.0f}  slippage ${t['slippage_cost'].sum():>10,.0f}"
          f"  funding ${t['funding_paid'].sum():>10,.0f}")
    print(f"  net PnL    ${t['pnl'].sum():>10,.0f}")
    yr = t.groupby(t["entry_time"].dt.year)
    for y, g in yr:
        gg = g["pnl"] + g["commission"] + g["slippage_cost"] + g["funding_paid"]
        print(f"  {y}: gross ${gg.sum():>9,.0f} | comm ${g['commission'].sum():>8,.0f}"
              f" | slip ${g['slippage_cost'].sum():>8,.0f} | fund ${g['funding_paid'].sum():>8,.0f}"
              f" | net ${g['pnl'].sum():>9,.0f}")
    print(f"\n  notional(size_usd): 중앙값 ${t['size_usd'].median():,.0f} / 평균 ${t['size_usd'].mean():,.0f}"
          f" / 합계 ${t['size_usd'].sum():,.0f}")
    print(f"  전략별 건수: {t['strategy'].value_counts().to_dict()}")
    print(f"  청산사유: {t['exit_reason'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
