"""슬리브 TP4.0/SL2.0 2단계 검증 — 플라토 + 연도별 + IS/OOS.

1단계(전구간)서 유일 생존: $12,914/MDD-40.7/Sh2.07 (베이스라인 $8,991/-42.4/1.94).
판정(사전선언): ①플라토(3.5~4.5 완만, 4.0 고립 스파이크면 기각)
②연도별 파국 없음 ③IS·OOS 둘 다 베이스라인(IS Sh1.32/MDD-42.4·OOS Sh3.44/MDD-35.5) 비열위.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

CONFIG = "config/final_v17.yaml"

WINDOWS = [("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2026-04-14"),
           ("IS", "2022-01-01", "2024-12-31"), ("OOS", "2025-01-01", "2026-04-14")]

# (이름, tp, start, end)
SPECS = (
    # 플라토 (전구간): 3.0(기록 8991/1.94)·4.0(12914/2.07) 사이/바깥
    [(f"플라토 tp{tp}", tp, None, None) for tp in (3.5, 4.5, 5.0)]
    # tp4.0 연도별 + IS/OOS
    + [(f"tp4.0 {w}", 4.0, s, e) for w, s, e in WINDOWS]
    # 베이스라인(tp3.0) 연도별 — IS/OOS는 기록 재사용
    + [(f"base {w}", 3.0, s, e) for w, s, e in WINDOWS[:4]]
)


def run(spec):
    name, tp, start, end = spec
    p = yaml.safe_load(open(CONFIG))
    p["strategies"]["mean_reversion"]["atr_tp_mult"] = tp
    bt = p.get("backtest", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    since = pd.Timestamp(start or bt.get("start", "2022-01-01"), tz="UTC")
    until_s = end or bt.get("end")
    until = pd.Timestamp(until_s, tz="UTC") if until_s else None
    report = engine.run(loader.iterate(since=since, until=until))
    return {"name": name, "ret": report.total_return_pct, "mdd": report.max_drawdown,
            "sharpe": report.sharpe, "trades": report.total_trades,
            "final": report.final_equity}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=min(13, len(SPECS))) as ex:
        results = list(ex.map(run, SPECS))
    print(f"{'런':16} {'수익%':>10} {'MDD%':>7} {'Sharpe':>7} {'거래':>5} {'최종$':>9}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:16} {r['ret']:>+10.1f} {r['mdd']:>7.1f} {r['sharpe']:>7.2f} "
              f"{r['trades']:>5} {r['final']:>9,.0f}")
