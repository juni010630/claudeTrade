"""갭 SL 비관체결(gap_sl_pessimistic) 측정.

목적 2개:
  1. off = 기존 v17 결과와 완전 동일 확인 (옵션 추가가 기본 경로 무영향 증명)
     기대값: 전구간 $8,991 / Sh 1.94 / MDD -42.4% / 805거래
  2. on  = 갭이 SL을 관통한 봉에서 시가 체결 시 정직 수익/MDD (라이브 기대치)
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
    gap_on, period, since_str, until_str = args
    t0 = time.time()
    p = yaml.safe_load(open(CONFIG))
    p.setdefault("engine", {})["gap_sl_pessimistic"] = gap_on
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
        "gap": gap_on, "period": period,
        "final": round(report.final_equity, 2),
        "ret": round(report.total_return_pct, 1),
        "mdd": round(report.max_drawdown, 2),
        "sharpe": round(report.sharpe, 3),
        "trades": report.total_trades,
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
            tag = "ON " if r["gap"] else "off"
            print(f"  gap={tag} {r['period']:4}  ${r['final']:>10,.0f}  "
                  f"MDD{r['mdd']:>7.2f}  Sh{r['sharpe']:>6.3f}  거래{r['trades']:>4}  ({r['secs']}s)",
                  flush=True)

    by = {(r["gap"], r["period"]): r for r in res}
    print("\n| 기간 | 체결 | 최종$ | MDD | Sharpe | 거래 |")
    print("|---|---|---|---|---|---|")
    for period in ["FULL", "OOS"]:
        for gap in [False, True]:
            r = by[(gap, period)]
            print(f"| {period} | {'갭비관' if gap else '기존'} | {r['final']:,.0f} | "
                  f"{r['mdd']:.2f}% | {r['sharpe']:.3f} | {r['trades']} |")


if __name__ == "__main__":
    main()
