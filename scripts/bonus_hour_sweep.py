"""bonus_hour_sweep.py — 유리 시간대 사이즈 보너스 배율 스윕.

좋은 시간대(PF 높음, WR 높음)에 사이즈를 N배 키웠을 때 성과 비교.

Usage:
    python scripts/bonus_hour_sweep.py
    python scripts/bonus_hour_sweep.py --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

# ── 분석 기반 유리 시간대 ──────────────────────────────────────────────────
# ema_cross:       PF ≥ 3.9 & count ≥ 8  → 0, 5, 12, 15, 22
# multi_tf:        PF ≥ 6.8 & count ≥ 5  → 0, 1, 12, 14, 22
BONUS_HOURS = {
    "ema_cross":         [0, 5, 12, 15, 22],
    "multi_tf_breakout": [0, 1, 12, 14, 22],
}


def _worker(args: tuple) -> dict:
    config_path, initial_capital, since_str, until_str, mult = args

    with open(config_path) as f:
        p = yaml.safe_load(f)

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    engine = build_engine(
        p, initial_capital,
        strategy_size_bonus=BONUS_HOURS,
        strategy_size_bonus_mult=mult,
    )
    report = engine.run(loader.iterate(since=since, until=until))

    # 분기별 수익률 (stability 측정용)
    eq = engine.equity_curve.to_series()
    eq.index = pd.to_datetime(eq.index, utc=True)
    daily = eq.resample("D").last().ffill()
    q_rets = daily.resample("QE").last().pct_change().dropna() * 100
    neg_q = (q_rets < 0).sum()

    return {
        "mult":          mult,
        "sharpe":        round(report.sharpe, 3),
        "mdd_%":         round(report.max_drawdown * 100, 1),
        "calmar":        round(getattr(report, "calmar", float("nan")), 2),
        "win_rate_%":    round(report.win_rate, 1),
        "profit_factor": round(getattr(report, "profit_factor", float("nan")), 2),
        "final_equity":  round(report.final_equity, 0),
        "trades":        report.total_trades,
        "neg_quarters":  int(neg_q),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",  default="config/final_v13_eth.yaml")
    parser.add_argument("--start",   default="2022-01-01")
    parser.add_argument("--end",     default="2026-06-06")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)
    initial_capital = args.capital or p.get("backtest", {}).get("initial_capital", 100)

    # 보너스 없는 기준선 (mult=1.0)
    mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    print(f"유리 시간대: ema_cross {BONUS_HOURS['ema_cross']}")
    print(f"           multi_tf  {BONUS_HOURS['multi_tf_breakout']}")
    print(f"\n{len(mults)}개 배율 스윕 (workers={args.workers}): {mults}\n")

    worker_args = [
        (args.params, initial_capital, args.start, args.end, m)
        for m in mults
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, a): a for a in worker_args}
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
                results.append(row)
                base = " ← 기준선" if row["mult"] == 1.0 else ""
                print(f"  x{row['mult']:.2f}  Sharpe {row['sharpe']:.2f}  "
                      f"MDD {row['mdd_%']:.1f}%  PF {row['profit_factor']:.2f}  "
                      f"WR {row['win_rate_%']:.1f}%  ${row['final_equity']:,.0f}{base}")
            except Exception as e:
                print(f"  실패: {e}")

    df = pd.DataFrame(results).sort_values("mult").reset_index(drop=True)

    print("\n" + "=" * 85)
    print(f"보너스 배율 스윕 결과  (유리시간대: ema={BONUS_HOURS['ema_cross']}, "
          f"multi={BONUS_HOURS['multi_tf_breakout']})")
    print("=" * 85)
    print(df.to_string(index=False))

    best = df.loc[df["sharpe"].idxmax()]
    base = df[df["mult"] == 1.0].iloc[0]
    print("\n" + "=" * 85)
    print(f"최적 배율: x{best['mult']:.2f}  "
          f"Sharpe {best['sharpe']:.2f} (기준 {base['sharpe']:.2f})  "
          f"MDD {best['mdd_%']:.1f}% (기준 {base['mdd_%']:.1f}%)")

    print("\n[Sharpe vs 보너스 배율]")
    max_s = df["sharpe"].max()
    for _, r in df.iterrows():
        bar = int(r["sharpe"] / max_s * 40)
        base_mark = " ← 기준" if r["mult"] == 1.0 else ""
        best_mark = " ← 최적" if r["mult"] == best["mult"] and r["mult"] != 1.0 else ""
        print(f"  x{r['mult']:.2f}  {'█'*bar:<40} {r['sharpe']:.2f}{base_mark}{best_mark}")


if __name__ == "__main__":
    main()
