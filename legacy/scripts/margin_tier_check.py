"""MarginTierTable 배선 측정 (engine.use_margin_tiers).

off = flat MMR 0.5% (현행) / on = Binance 브래킷 MMR (notional 커질수록 보수적).
off가 기존 v17 수치와 동일해야 하고, on은 청산 판정 현실화 시 수익/MDD 변화 측정.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

CONFIG = "config/final_v17.yaml"
FULL = ("2022-01-01", "2026-04-23")
OOS = ("2025-01-01", "2026-04-14")


def run_one(args: tuple) -> dict:
    tiers_on, period, since_str, until_str = args
    t0 = time.time()
    p = yaml.safe_load(open(CONFIG))
    p.setdefault("engine", {})["use_margin_tiers"] = tiers_on
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(since_str, tz="UTC"),
        until=pd.Timestamp(until_str, tz="UTC")))
    return {
        "tiers": tiers_on, "period": period,
        "final": round(report.final_equity, 2),
        "mdd": round(report.max_drawdown, 2),
        "sharpe": round(report.sharpe, 3),
        "trades": report.total_trades,
        "bankrupt": report.bankrupt,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    tasks = [
        (False, "FULL", *FULL), (True, "FULL", *FULL),
        (False, "OOS", *OOS), (True, "OOS", *OOS),
    ]
    res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            res.append(r)
            tag = "ON " if r["tiers"] else "off"
            print(f"  tiers={tag} {r['period']:4}  ${r['final']:>10,.0f}  "
                  f"MDD{r['mdd']:>7.2f}  Sh{r['sharpe']:>6.3f}  거래{r['trades']:>4}"
                  f"{'  ⚠️파산' if r['bankrupt'] else ''}  ({r['secs']}s)", flush=True)

    by = {(r["tiers"], r["period"]): r for r in res}
    print("\n| 기간 | MMR | 최종$ | MDD | Sharpe | 거래 |")
    print("|---|---|---|---|---|---|")
    for period in ["FULL", "OOS"]:
        for tiers in [False, True]:
            r = by[(tiers, period)]
            print(f"| {period} | {'브래킷' if tiers else 'flat 0.5%'} | {r['final']:,.0f} | "
                  f"{r['mdd']:.2f}% | {r['sharpe']:.3f} | {r['trades']} |")


if __name__ == "__main__":
    main()
