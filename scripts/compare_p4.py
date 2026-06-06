"""Phase 4 — tp_cooldown_hours 스윕 병렬 비교."""
from __future__ import annotations

import sys
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


CONFIGS = {
    "0h (없음)":   "config/p4_cooldown_0h.yaml",
    "2h":          "config/p4_cooldown_2h.yaml",
    "4h":          "config/p4_cooldown_4h.yaml",
    "6h (현재)":   "config/final_v13_eth.yaml",
    "12h":         "config/p4_cooldown_12h.yaml",
    "24h":         "config/p4_cooldown_24h.yaml",
}


def run_one(label: str, config_path: str) -> dict:
    with open(config_path) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    engine = build_engine(p, initial_capital)
    snapshots = loader.iterate(since=since, until=until)
    report = engine.run(snapshots)

    df = engine.ledger.to_dataframe()
    trades = len(df)
    wr = (df["pnl"] > 0).mean() * 100 if trades else 0
    tp_count = (df["exit_reason"] == "tp").sum() if trades else 0

    return {
        "label":    label,
        "final_eq": report.final_equity,
        "sharpe":   report.sharpe,
        "calmar":   report.calmar,
        "mdd":      report.max_drawdown,
        "trades":   trades,
        "tp_count": tp_count,
        "wr":       round(wr, 1),
        "pf":       report.profit_factor,
        "cagr":     report.cagr,
    }


def main() -> None:
    print("6개 설정 병렬 백테스트 시작 (Phase 4 — tp_cooldown 스윕)...\n")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(run_one, label, cfg): label
            for label, cfg in CONFIGS.items()
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            try:
                results[label] = fut.result()
                print(f"  완료: {label}")
            except Exception as e:
                print(f"  실패: {label} — {e}")
                import traceback; traceback.print_exc()

    ordered = [results[k] for k in CONFIGS if k in results]
    base = next((r for r in ordered if r["label"] == "6h (현재)"), None)

    print("\n" + "=" * 95)
    print(f"  {'설정':<12} {'최종자산':>13} {'CAGR':>7} {'Sharpe':>7} {'Calmar':>7} {'MDD':>8}  {'거래':>4} {'TP건':>5} {'WR%':>5} {'PF':>6}")
    print("=" * 95)

    for r in ordered:
        is_base = (r["label"] == "6h (현재)")
        sdiff = "" if is_base else f"({r['sharpe'] - base['sharpe']:+.3f})"
        cdiff = "" if is_base else f"({r['calmar'] - base['calmar']:+.2f})"
        print(
            f"  {r['label']:<12} "
            f"${r['final_eq']:>12,.0f} "
            f"{r['cagr']:>+6.0f}% "
            f"{r['sharpe']:>7.3f}{sdiff:<10} "
            f"{r['calmar']:>7.2f}{cdiff:<8} "
            f"{r['mdd']:>+7.1f}%  "
            f"{r['trades']:>4} "
            f"{r['tp_count']:>5} "
            f"{r['wr']:>5.1f}% "
            f"{r['pf']:>6.3f}"
        )

    print("=" * 95)

    if len(ordered) > 1:
        best_sharpe = max(ordered, key=lambda x: x["sharpe"])
        best_calmar = max(ordered, key=lambda x: x["calmar"])
        print(f"\n  Sharpe 최고: {best_sharpe['label']} ({best_sharpe['sharpe']:.3f})")
        print(f"  Calmar 최고: {best_calmar['label']} ({best_calmar['calmar']:.2f})")


if __name__ == "__main__":
    main()
