"""override_sweep.py — config 스칼라 오버라이드 변형들을 풀 백테스트로 비교.

중첩 키는 점 표기 ("risk.max_positions": 8). 전기간 + 연도별 동시 평가.
병렬(ProcessPoolExecutor), 폴링 없음. baseline은 data/results/symbol_add_sweep.csv 재사용 가능.

Usage:
  python3 -u scripts/override_sweep.py --workers 10
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


def sweep_configs() -> dict[str, dict]:
    """name -> {점표기 키: 값} 오버라이드."""
    return {
        "baseline":   {},
        # R4: 전략 파라미터 이웃 (ema 8/21, multi bb 25/2.2, vol 1.8x, rsi 55/45)
        "ema_6_18":   {"strategies.ema_cross.fast_period": 6,
                       "strategies.ema_cross.slow_period": 18},
        "ema_10_26":  {"strategies.ema_cross.fast_period": 10,
                       "strategies.ema_cross.slow_period": 26},
        "mul_bb20":   {"strategies.multi_tf_breakout.bb_std_1h": 2.0},
        "mul_bb24":   {"strategies.multi_tf_breakout.bb_std_1h": 2.4},
        "mul_vol15":  {"strategies.multi_tf_breakout.volume_multiplier": 1.5},
        "mul_vol21":  {"strategies.multi_tf_breakout.volume_multiplier": 2.1},
        "mul_rsi5050": {"strategies.multi_tf_breakout.rsi_long_min": 50.0,
                        "strategies.multi_tf_breakout.rsi_short_max": 50.0},
        "mul_rsi6040": {"strategies.multi_tf_breakout.rsi_long_min": 60.0,
                        "strategies.multi_tf_breakout.rsi_short_max": 40.0},
    }


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "WR%": 0.0, "PF": 0.0}
    wins = (df["pnl"] > 0).sum()
    gw = df.loc[df["pnl"] > 0, "pnl"].sum()
    gl = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
    return {
        "trades": len(df),
        "WR%": round(wins / len(df) * 100, 1),
        "PF": round(gw / gl, 2) if gl > 0 else float("inf"),
    }


def _worker(args: tuple) -> dict:
    name, params_path, overrides, since_str, until_str, period_tag = args
    with open(params_path) as f:
        p = yaml.safe_load(f)
    for dotted, val in overrides.items():
        node = p
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val

    bt = p.get("backtest", {})
    cap = bt.get("initial_capital", 100)
    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None
    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    engine = build_engine(p, cap)
    report = engine.run(loader.iterate(since=since, until=until))
    df = engine.ledger.to_dataframe()
    m = _metrics(df)
    return {
        "name": name, "period": period_tag,
        "final_equity": round(report.final_equity, 1),
        "sharpe": round(report.sharpe, 3),
        "calmar": round(report.calmar, 2),
        "max_dd_%": round(report.max_drawdown, 1),
        **m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="config/final_v13_eth.yaml")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--full-only", action="store_true", help="full 기간만 (스크리닝용)")
    args = ap.parse_args()

    cfgs = sweep_configs()
    if args.skip_baseline:
        cfgs.pop("baseline", None)
    with open(args.params) as f:
        _end = yaml.safe_load(f).get("backtest", {}).get("end")
    periods = [("2022-01-01", _end, "full"),
               ("2022-01-01", "2022-12-31", "2022"), ("2023-01-01", "2023-12-31", "2023"),
               ("2024-01-01", "2024-12-31", "2024"), ("2025-01-01", _end, "2025-26")]
    if args.full_only:
        periods = periods[:1]
    jobs = []
    for name, ov in cfgs.items():
        for since, until, tag in periods:
            jobs.append((name, args.params, ov, since, until, tag))

    print(f"[오버라이드 스윕] {len(cfgs)}개 설정 × {len(periods)}기간 = {len(jobs)}개 백테스트 | workers={args.workers}",
          flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, j): (j[0], j[5]) for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            nm, tag = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']:<12} [{r['period']:<7}] "
                      f"eq ${r['final_equity']:>12,.0f}  Sharpe {r['sharpe']:>6.3f}  "
                      f"Calmar {r['calmar']:>6.2f}  MDD {r['max_dd_%']:>6.1f}%  "
                      f"WR {r['WR%']:>4.1f}%  PF {r['PF']:>6.2f}  n={r['trades']}", flush=True)
            except Exception as e:
                print(f"  ✗ {nm}[{tag}] 실패: {e}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv("data/results/override_sweep.csv", index=False)
    for tag in [pt[2] for pt in periods]:
        sub = res[res.period == tag].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("sharpe", ascending=False)
        print(f"\n{'='*108}\n  [{tag}] 오버라이드 비교\n{'='*108}", flush=True)
        cols = ["name", "final_equity", "sharpe", "calmar", "max_dd_%", "WR%", "PF", "trades"]
        print(sub[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
