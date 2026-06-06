"""버킷 픽스 병렬 비교 스크립트.

Baseline + FixA + FixB + FixAB 를 동시에 백테스트하고 결과 비교.
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
    "Baseline":       "config/final_v13_eth.yaml",
    "FixA (ema≥5)":   "config/bucket_fixA.yaml",
    "FixB (ETH-long)":"config/bucket_fixB.yaml",
    "FixAB (둘다)":    "config/bucket_fixAB.yaml",
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

    r = report
    df = engine.ledger.to_dataframe()
    trades = len(df)
    wr = (df["pnl"] > 0).mean() * 100 if trades else 0

    return {
        "label":        label,
        "final_equity": r.final_equity,
        "total_pct":    (r.final_equity / initial_capital - 1) * 100,
        "sharpe":       r.sharpe,
        "sortino":      r.sortino,
        "calmar":       r.calmar,
        "mdd":          r.max_drawdown,   # already %-scaled in MetricsReport
        "trades":       trades,
        "wr":           round(wr, 1),
        "pf":           r.profit_factor,
        "cagr":         r.cagr,           # already %-scaled in MetricsReport
    }


def main() -> None:
    print("4개 설정 병렬 백테스트 시작...\n")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(run_one, label, cfg): label
            for label, cfg in CONFIGS.items()
        }
        for fut in concurrent.futures.as_completed(futures):
            label = futures[fut]
            try:
                res = fut.result()
                results[label] = res
                print(f"  완료: {label}")
            except Exception as e:
                print(f"  실패: {label} — {e}")
                import traceback; traceback.print_exc()

    # 원래 순서로 정렬
    ordered = [results[k] for k in CONFIGS if k in results]

    print("\n" + "=" * 90)
    print(f"  {'설정':<18} {'최종자산':>12} {'수익률':>9} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} {'MDD':>7} {'거래':>5} {'WR%':>5} {'PF':>6}")
    print("=" * 90)

    baseline = ordered[0] if ordered else None
    for r in ordered:
        sharpe_diff = f"({r['sharpe']-baseline['sharpe']:+.3f})" if baseline and r['label'] != baseline['label'] else ""
        calmar_diff = f"({r['calmar']-baseline['calmar']:+.2f})" if baseline and r['label'] != baseline['label'] else ""
        mdd_diff    = f"({r['mdd']-baseline['mdd']:+.1f}%)" if baseline and r['label'] != baseline['label'] else ""
        print(
            f"  {r['label']:<18} "
            f"${r['final_equity']:>11,.0f} "
            f"{r['total_pct']:>+8.0f}% "
            f"{r['sharpe']:>7.3f}{sharpe_diff:<10} "
            f"{r['sortino']:>7.3f} "
            f"{r['calmar']:>7.2f}{calmar_diff:<8} "
            f"{r['mdd']:>6.1f}%{mdd_diff:<10} "
            f"{r['trades']:>5} "
            f"{r['wr']:>5.1f}% "
            f"{r['pf']:>6.3f}"
        )

    print("=" * 90)

    if len(ordered) > 1:
        best = max(ordered, key=lambda x: x["sharpe"])
        print(f"\n  Sharpe 기준 최고: {best['label']} (Sharpe {best['sharpe']:.3f})")
        best_cal = max(ordered, key=lambda x: x["calmar"])
        print(f"  Calmar 기준 최고: {best_cal['label']} (Calmar {best_cal['calmar']:.2f})")


if __name__ == "__main__":
    main()
