"""symbol_add_sweep.py — 심볼 유니버스 확장 후보를 한 개씩 추가하고 풀 백테스트로 비교.

후보: 4년치(2022~) 데이터가 있는 미사용 심볼 (SOL/AVAX/XRP/BNB).
ETHUSDT는 regime 기준이므로 항상 첫 번째 유지. 추가만 하고 기존 유니버스는 불변.
전기간 + 연도별(2022/23/24/25-26) 동시 평가. 병렬(ProcessPoolExecutor), 폴링 없음.

Usage:
  python3 -u scripts/symbol_add_sweep.py --workers 10
  python3 -u scripts/symbol_add_sweep.py --only add_SOLUSDT,add_combo_all
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

CANDIDATES = ["SOLUSDT", "AVAXUSDT", "XRPUSDT", "BNBUSDT"]


def sweep_configs() -> dict[str, list[str]]:
    """name -> 추가 심볼 리스트. baseline = 추가 없음."""
    cfgs = {"baseline": []}
    for sym in CANDIDATES:
        cfgs[f"add_{sym}"] = [sym]
    cfgs["add_combo_all"] = list(CANDIDATES)
    return cfgs


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
    name, params_path, add_syms, since_str, until_str, period_tag = args
    with open(params_path) as f:
        p = yaml.safe_load(f)
    p["symbols"] = list(p["symbols"]) + [s for s in add_syms if s not in p["symbols"]]

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
    ap.add_argument("--only", default=None, help="쉼표구분 설정명만 실행 (baseline 자동 포함)")
    args = ap.parse_args()

    cfgs = sweep_configs()
    if args.only:
        keep = set(args.only.split(",")) | {"baseline"}
        cfgs = {k: v for k, v in cfgs.items() if k in keep}
    with open(args.params) as f:
        _end = yaml.safe_load(f).get("backtest", {}).get("end")
    periods = [("2022-01-01", _end, "full"),
               ("2022-01-01", "2022-12-31", "2022"), ("2023-01-01", "2023-12-31", "2023"),
               ("2024-01-01", "2024-12-31", "2024"), ("2025-01-01", _end, "2025-26")]
    jobs = []
    for name, add in cfgs.items():
        for since, until, tag in periods:
            jobs.append((name, args.params, add, since, until, tag))

    print(f"[심볼 추가 스윕] {len(cfgs)}개 설정 × {len(periods)}기간 = {len(jobs)}개 백테스트 | workers={args.workers}",
          flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, j): (j[0], j[5]) for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            nm, tag = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']:<16} [{r['period']:<7}] "
                      f"eq ${r['final_equity']:>12,.0f}  Sharpe {r['sharpe']:>6.3f}  "
                      f"Calmar {r['calmar']:>6.2f}  MDD {r['max_dd_%']:>6.1f}%  "
                      f"WR {r['WR%']:>4.1f}%  PF {r['PF']:>6.2f}  n={r['trades']}", flush=True)
            except Exception as e:
                print(f"  ✗ {nm}[{tag}] 실패: {e}", flush=True)

    res = pd.DataFrame(results)
    res.to_csv("data/results/symbol_add_sweep.csv", index=False)
    for tag in [pt[2] for pt in periods]:
        sub = res[res.period == tag].copy()
        if sub.empty:
            continue
        base = sub[sub.name == "baseline"]
        sub = sub.sort_values("sharpe", ascending=False)
        print(f"\n{'='*108}\n  [{tag}] 심볼 추가 비교 (baseline 대비)\n{'='*108}", flush=True)
        cols = ["name", "final_equity", "sharpe", "calmar", "max_dd_%", "WR%", "PF", "trades"]
        print(sub[cols].to_string(index=False), flush=True)
        if not base.empty:
            b = base.iloc[0]
            print(f"\n  baseline: eq ${b['final_equity']:,.0f}  Sharpe {b['sharpe']}  "
                  f"Calmar {b['calmar']}  MDD {b['max_dd_%']}%", flush=True)


if __name__ == "__main__":
    main()
