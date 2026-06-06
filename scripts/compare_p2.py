"""Phase 2 — 방향 편향 보정 병렬 비교.

Baseline + long_penalty + short_bonus + combined + short_only
"""
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
    "Baseline":              "config/final_v13_eth.yaml",
    "long×0.5":              "config/p2_long_penalty.yaml",
    "short×1.25":            "config/p2_short_bonus.yaml",
    "long×0.5+short×1.25":  "config/p2_combined.yaml",
    "short_only":            "config/p2_short_only.yaml",
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
    long_trades = (df["direction"] == "long").sum() if trades else 0
    short_trades = (df["direction"] == "short").sum() if trades else 0

    return {
        "label":        label,
        "final_equity": report.final_equity,
        "sharpe":       report.sharpe,
        "calmar":       report.calmar,
        "mdd":          report.max_drawdown,
        "trades":       trades,
        "long_t":       long_trades,
        "short_t":      short_trades,
        "wr":           round(wr, 1),
        "pf":           report.profit_factor,
        "cagr":         report.cagr,
    }


def main() -> None:
    print("5개 설정 병렬 백테스트 시작 (Phase 2 — 방향 편향)...\n")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as pool:
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
    base = ordered[0] if ordered else None

    print("\n" + "=" * 100)
    print(f"  {'설정':<26} {'최종자산':>13} {'CAGR':>7} {'Sharpe':>7} {'Calmar':>7} {'MDD':>8}  {'거래':>4} {'L/S':>9} {'WR%':>5} {'PF':>6}")
    print("=" * 100)

    for r in ordered:
        sdiff = f"({r['sharpe']-base['sharpe']:+.3f})" if base and r['label'] != base['label'] else "        "
        cdiff = f"({r['calmar']-base['calmar']:+.2f})" if base and r['label'] != base['label'] else "       "
        print(
            f"  {r['label']:<26} "
            f"${r['final_equity']:>12,.0f} "
            f"{r['cagr']:>+6.0f}% "
            f"{r['sharpe']:>7.3f}{sdiff:<10} "
            f"{r['calmar']:>7.2f}{cdiff:<8} "
            f"{r['mdd']:>+7.1f}%  "
            f"{r['trades']:>4} "
            f"{r['long_t']:>4}L/{r['short_t']:<4}S "
            f"{r['wr']:>5.1f}% "
            f"{r['pf']:>6.3f}"
        )

    print("=" * 100)

    if len(ordered) > 1:
        # 반려 기준: short_only는 수익률이 baseline 대비 "미친듯이" 차이 나지 않으면 반려
        so = results.get("short_only")
        if so and base:
            equity_ratio = so["final_equity"] / base["final_equity"]
            verdict = "채택 후보" if equity_ratio >= 0.8 and so["sharpe"] > base["sharpe"] else "반려 (long 차단 손실 과다)"
            print(f"\n  short_only 판정: 수익 {equity_ratio:.1%} of baseline → {verdict}")

        best_sharpe = max(ordered, key=lambda x: x["sharpe"])
        best_calmar = max(ordered, key=lambda x: x["calmar"])
        print(f"\n  Sharpe 최고: {best_sharpe['label']} ({best_sharpe['sharpe']:.3f})")
        print(f"  Calmar 최고: {best_calmar['label']} ({best_calmar['calmar']:.2f})")


if __name__ == "__main__":
    main()
