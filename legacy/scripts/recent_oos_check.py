"""최근 홀드아웃 윈도우(옵티마이즈 종료 2026-04-14 이후)에서
merged 시스템 baseline vs 과적합 레이어(시간블록/size_bonus) 교란 비교.

목적: "최근 백테가 좋으면 과적합이 작다"를 검증.
 - 표본 크기(거래 건수)
 - 시간레이어 ±1h 시프트 시 결과 변동 → 윈도우가 그 레이어를 검증할 수 있는가
"""
from __future__ import annotations

import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb

START = "2026-04-24"
END = "2026-06-07"


def _shift(d: dict, k: int) -> dict:
    return {strat: sorted({(h + k) % 24 for h in hrs}) for strat, hrs in d.items()}


def run_one(spec: dict) -> dict:
    name = spec["name"]
    with open(spec["config"]) as f:
        p = yaml.safe_load(f)
    spec["mutate"](p)

    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(START, tz="UTC"), until=pd.Timestamp(END, tz="UTC")))

    df = engine.ledger.to_dataframe()
    by_strat = {}
    if not df.empty:
        by_strat = df.groupby("strategy").size().to_dict()
    return {
        "name": name,
        "ret": report.total_return_pct,
        "mdd": report.max_drawdown,
        "sharpe": report.sharpe,
        "trades": report.total_trades,
        "wr": report.win_rate,
        "by_strat": by_strat,
    }


# 교란기들 (in-place mutate)
def _noop(p): pass
def _block1(p):
    if p.get("strategy_block_hours"): p["strategy_block_hours"] = _shift(p["strategy_block_hours"], 1)
def _block2(p):
    if p.get("strategy_block_hours"): p["strategy_block_hours"] = _shift(p["strategy_block_hours"], 2)
def _bonus1(p):
    if p.get("strategy_size_bonus"): p["strategy_size_bonus"] = _shift(p["strategy_size_bonus"], 1)
def _both1(p): _block1(p); _bonus1(p)
def _noblock(p): p.pop("strategy_block_hours", None)
def _nobonus(p): p.pop("strategy_size_bonus", None)


def main() -> None:
    config = sys.argv[1] if len(sys.argv) > 1 else "config/merged_v16_sleeve.yaml"
    specs = [
        {"name": "baseline",     "mutate": _noop},
        {"name": "block +1h",    "mutate": _block1},
        {"name": "block +2h",    "mutate": _block2},
        {"name": "bonus +1h",    "mutate": _bonus1},
        {"name": "both +1h",     "mutate": _both1},
        {"name": "no block",     "mutate": _noblock},
        {"name": "no bonus",     "mutate": _nobonus},
    ]
    for s in specs:
        s["config"] = config
    print(f"config: {config}")
    with ProcessPoolExecutor(max_workers=7) as ex:
        results = list(ex.map(run_one, specs))
    order = {s["name"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order[r["name"]])

    print(f"\n최근 홀드아웃 윈도우: {START} ~ {END}  ($100 시작)")
    print("=" * 70)
    print(f"{'variant':12} {'수익%':>8} {'MDD%':>8} {'Sharpe':>7} {'거래':>5} {'승률%':>6}  전략별")
    print("-" * 70)
    for r in results:
        bs = " ".join(f"{k[:4]}={v}" for k, v in sorted(r["by_strat"].items()))
        print(f"{r['name']:12} {r['ret']:>8.1f} {r['mdd']:>8.1f} {r['sharpe']:>7.2f} "
              f"{r['trades']:>5} {r['wr']:>6.0f}  {bs}")
    print("=" * 70)
    base = next(r for r in results if r["name"] == "baseline")
    print(f"\nbaseline 거래 {base['trades']}건 / {(pd.Timestamp(END)-pd.Timestamp(START)).days}일")


if __name__ == "__main__":
    main()
