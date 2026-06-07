"""pyramid_5m_xval.py — 피라미딩 5m 교차검증.

1h bar-level 결과(기존 스윕)와 5m sub-bar 순회 결과를 비교.
괴리 ~0% → 1h 결과 신뢰 가능 (트리거 intrabar 순서가 결과를 왜곡하지 않음).

baseline(피라미딩 off)도 함께 돌려 5m 장치 자체의 무왜곡을 재확인.
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

CONFIGS = {
    "base_5m": {},
    "pyr_5m":  {"pyramid": {"enabled": True, "trigger_r": 1.0,
                            "add_fraction": 0.25, "max_adds": 1}},
}


def _worker(args: tuple) -> dict:
    name, overrides, since_str, until_str, tag = args
    with open("config/final_v13_eth.yaml") as f:
        p = yaml.safe_load(f)
    p.update(overrides)
    if "5m" not in p["timeframes"]:
        p["timeframes"] = list(p["timeframes"]) + ["5m"]

    cap = p.get("backtest", {}).get("initial_capital", 100)
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir="data/cache",
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = build_engine(p, cap, subbar_tpsl=True)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(since_str, tz="UTC"),
        until=pd.Timestamp(until_str, tz="UTC") if until_str else None,
    ))
    df = engine.ledger.to_dataframe()
    return {
        "name": name, "period": tag,
        "final_equity": round(report.final_equity, 1),
        "sharpe": round(report.sharpe, 3),
        "max_dd_%": round(report.max_drawdown, 1),
        "trades": len(df),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    with open("config/final_v13_eth.yaml") as f:
        _end = yaml.safe_load(f).get("backtest", {}).get("end")
    periods = [("2022-01-01", _end, "full"),
               ("2022-01-01", "2022-12-31", "2022"), ("2023-01-01", "2023-12-31", "2023"),
               ("2024-01-01", "2024-12-31", "2024"), ("2025-01-01", _end, "2025-26")]
    jobs = [(n, ov, s, u, t) for n, ov in CONFIGS.items() for s, u, t in periods]
    print(f"[5m 교차검증] {len(jobs)}개 백테스트 | workers={args.workers}", flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, j): (j[0], j[4]) for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            nm, tag = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']:<8} [{r['period']:<7}] eq ${r['final_equity']:>12,.0f}  "
                      f"Sharpe {r['sharpe']:>6.3f}  MDD {r['max_dd_%']:>6.1f}%  n={r['trades']}",
                      flush=True)
            except Exception as e:
                print(f"  ✗ {nm}[{tag}] 실패: {e}", flush=True)

    res = pd.DataFrame(results).sort_values(["name", "period"])
    res.to_csv("data/results/pyramid_5m_xval.csv", index=False)
    print("\n" + res.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
