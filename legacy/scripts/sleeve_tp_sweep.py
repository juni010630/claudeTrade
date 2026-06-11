"""슬리브 TP 배수 스윕 (v17 병합 config, 엔진 레벨) — '조금씩 많이' 가설 검증.

mean_reversion.atr_tp_mult {1.5, 2.0, 2.5, 3.0(현행)} 전구간 병렬.
판정(사전선언): Sharpe ≥ 현행 AND MDD 비악화 — 통과 시에만 연도별+실측비용 스트레스.
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
TP_GRID = [1.5, 2.0, 2.5, 3.0]


def run(tp: float) -> dict:
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
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt["end"], tz="UTC") if bt.get("end") else None
    report = engine.run(loader.iterate(since=since, until=until))
    mr = [r for r in engine.ledger.records if r.strategy == "mean_reversion"]
    wins = sum(1 for r in mr if r.pnl > 0)
    return {
        "tp": tp, "final": report.final_equity, "mdd": report.max_drawdown,
        "sharpe": report.sharpe, "trades": report.total_trades,
        "mr_n": len(mr), "mr_wr": 100 * wins / len(mr) if mr else 0,
        "mr_pnl": sum(r.pnl for r in mr),
    }


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=len(TP_GRID)) as ex:
        results = list(ex.map(run, TP_GRID))
    print(f"{'TP':>4} {'최종$':>9} {'MDD%':>7} {'Sharpe':>7} {'거래':>5} "
          f"{'슬리브n':>7} {'슬리브WR%':>9} {'슬리브PnL':>10}")
    print("-" * 66)
    for r in results:
        tag = " ← 현행" if r["tp"] == 3.0 else ""
        print(f"{r['tp']:>4} {r['final']:>9,.0f} {r['mdd']:>7.1f} {r['sharpe']:>7.2f} "
              f"{r['trades']:>5} {r['mr_n']:>7} {r['mr_wr']:>9.1f} {r['mr_pnl']:>+10.1f}{tag}")
