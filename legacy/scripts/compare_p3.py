"""Phase 3 — max_hold_hours 스윕 병렬 비교."""
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
    "48h":          "config/p3_hold_48h.yaml",
    "96h":          "config/p3_hold_96h.yaml",
    "168h (1w)":    "config/p3_hold_168h.yaml",
    "336h (현재)":   "config/final_v13_eth.yaml",
    "672h (4w)":    "config/p3_hold_672h.yaml",
    "None (무제한)": "config/p3_hold_none.yaml",
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
    timeouts = (df["exit_reason"] == "timeout").sum() if trades else 0
    avg_hold = (df["exit_time"] - df["entry_time"]).dt.total_seconds().mean() / 3600 if trades else 0

    return {
        "label":      label,
        "final_eq":   report.final_equity,
        "sharpe":     report.sharpe,
        "calmar":     report.calmar,
        "mdd":        report.max_drawdown,
        "trades":     trades,
        "timeouts":   timeouts,
        "avg_hold":   avg_hold,
        "wr":         round(wr, 1),
        "pf":         report.profit_factor,
        "cagr":       report.cagr,
    }


def main() -> None:
    print("6개 설정 병렬 백테스트 시작 (Phase 3 — max_hold_hours 스윕)...\n")

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
    base = next((r for r in ordered if r["label"] == "336h (현재)"), ordered[0] if ordered else None)

    print("\n" + "=" * 105)
    print(f"  {'설정':<14} {'최종자산':>13} {'CAGR':>7} {'Sharpe':>7} {'Calmar':>7} {'MDD':>8}  {'거래':>4} {'timeout':>7} {'avgHold':>8} {'WR%':>5} {'PF':>6}")
    print("=" * 105)

    for r in ordered:
        is_base = (r["label"] == "336h (현재)")
        sdiff = "" if is_base else f"({r['sharpe']-base['sharpe']:+.3f})"
        cdiff = "" if is_base else f"({r['calmar']-base['calmar']:+.2f})"
        print(
            f"  {r['label']:<14} "
            f"${r['final_eq']:>12,.0f} "
            f"{r['cagr']:>+6.0f}% "
            f"{r['sharpe']:>7.3f}{sdiff:<10} "
            f"{r['calmar']:>7.2f}{cdiff:<8} "
            f"{r['mdd']:>+7.1f}%  "
            f"{r['trades']:>4} "
            f"{r['timeouts']:>7} "
            f"{r['avg_hold']:>7.1f}h "
            f"{r['wr']:>5.1f}% "
            f"{r['pf']:>6.3f}"
        )

    print("=" * 105)

    if len(ordered) > 1:
        best_sharpe = max(ordered, key=lambda x: x["sharpe"])
        best_calmar = max(ordered, key=lambda x: x["calmar"])
        print(f"\n  Sharpe 최고: {best_sharpe['label']} ({best_sharpe['sharpe']:.3f})")
        print(f"  Calmar 최고: {best_calmar['label']} ({best_calmar['calmar']:.2f})")


if __name__ == "__main__":
    main()
