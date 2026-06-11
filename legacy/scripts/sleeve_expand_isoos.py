"""슬리브 확장(현 5 + IS선별 5 = 10심볼) v17 — IS/OOS 검증.

baseline(현 v17) 수치는 ema_tp_walkforward TP3.5 행 재사용 (재실행 금지):
  IS  +599.3% / Sh 1.32 / MDD -42.4
  OOS +1276.3% / Sh 3.44 / MDD -35.5
채택 조건(사전 등록): OOS에서 수익·Sharpe ≥ baseline AND MDD +5pp 이내.
주의: 79개 중 IS 상위 5 선별 = 다중비교 — OOS가 유일한 심판.
"""
from __future__ import annotations

import copy
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
IS = ("2022-01-01", "2024-12-31")
OOS = ("2025-01-01", "2026-04-14")
NEW_SYMBOLS = ["SFPUSDT", "AXSUSDT", "BCHUSDT", "ZECUSDT", "MTLUSDT"]


def run_period(args: tuple) -> dict:
    period, since_str, until_str = args
    t0 = time.time()
    p = copy.deepcopy(yaml.safe_load(open(CONFIG)))
    p["strategies"]["mean_reversion"]["symbols"] = (
        p["strategies"]["mean_reversion"]["symbols"] + NEW_SYMBOLS)
    p["symbols"] = p["symbols"] + NEW_SYMBOLS
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(since_str, tz="UTC"), until=pd.Timestamp(until_str, tz="UTC")))
    return {
        "period": period,
        "ret": round(report.total_return_pct, 1),
        "final": round(report.final_equity, 0),
        "mdd": round(report.max_drawdown, 2),
        "sharpe": round(report.sharpe, 3),
        "trades": report.total_trades,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    tasks = [("IS", *IS), ("OOS", *OOS)]
    res = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(run_period, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            res[r["period"]] = r
            print(f"  확장 {r['period']:4}  수익{r['ret']:>+8.1f}%  MDD{r['mdd']:>7.2f}  "
                  f"Sh{r['sharpe']:>6.3f}  거래{r['trades']:>4}  ({r['secs']}s)", flush=True)

    base = {"IS": {"ret": 599.3, "sharpe": 1.32, "mdd": -42.4},
            "OOS": {"ret": 1276.3, "sharpe": 3.44, "mdd": -35.5}}
    print("\n| 기간 | 구성 | 수익% | Sharpe | MDD |")
    print("|---|---|---|---|---|")
    for period in ["IS", "OOS"]:
        b, e = base[period], res[period]
        print(f"| {period} | baseline | {b['ret']:+.1f} | {b['sharpe']:.2f} | {b['mdd']:.1f} |")
        print(f"| {period} | 확장(+5) | {e['ret']:+.1f} | {e['sharpe']:.2f} | {e['mdd']:.1f} |")

    o, ob = res["OOS"], base["OOS"]
    ok = o["ret"] >= ob["ret"] and o["sharpe"] >= ob["sharpe"] and o["mdd"] >= ob["mdd"] - 5
    print(f"\n채택 판정 (OOS 수익·Sh ≥ base AND MDD +5pp 이내): {'통과 ✓' if ok else '기각 ✗'}")


if __name__ == "__main__":
    main()
