"""E축 — v17 배분 프론티어 추가 2점 (55:45, 60:40). 정보제공용, 채택 아님.

strategy_capital_fraction: 추세(ema/multi) w, 슬리브(mr) 1-w. 전구간.
기존점: 50:50 = $8,991/Sh1.94/MDD-42.4 (베이스), 40:60류는 RISK_DEPLOYMENT_FRONTIER.md.

사용: python scripts/alloc_v17_points.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml


def run_one(w_trend: float):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open("config/final_v17.yaml"))
    p["strategy_capital_fraction"] = {"ema_cross": w_trend, "multi_tf_breakout": w_trend,
                                      "mean_reversion": round(1 - w_trend, 2)}
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    report = eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                                    until=pd.Timestamp("2026-04-14", tz="UTC")))
    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    return w_trend, round(report.final_equity, 0), round(report.sharpe, 3), \
        round(report.max_drawdown, 1), len(eng.ledger.to_dataframe()), dict(yearly)


def main():
    with ProcessPoolExecutor(max_workers=2) as ex:
        for w, f, sh, mdd, n, yearly in ex.map(run_one, [0.55, 0.60]):
            ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
            print(f"  {int(w*100)}:{int((1-w)*100)}  ${f:>9,.0f} Sh{sh:6.3f} MDD봉{mdd:6.1f}% {n}건  {ys}")


if __name__ == "__main__":
    main()
