"""TP 쿨다운 3변형 비교 백테 (격리 — 프로덕션 무수정, 변형별 병렬 실행).

변형:
  1) 쿨다운없음(0h)  — tp_cooldown_hours=0
  2) TP 6h(현재)     — tp_cooldown_hours=6 (익절 후에만 6h)
  3) TP+SL 6h        — tp_cooldown_hours=6 + 손절도 6h 쿨다운(monkeypatch)

변형3은 BacktestEngine._close_with_reason를 일시 래핑해 exit_reason=='sl'일 때도
guards.record_tp 호출(프로덕션 파일 불변, run 후 원복). 쿨다운 키=(symbol, strategy).

사용:
  python scripts/cooldown_variants_bt.py 0|1|2   # 변형별 1프로세스(5기간) → /tmp/cooldown_v{N}.json
  python scripts/cooldown_variants_bt.py agg      # 3개 결과 병합·표 출력
변형은 3프로세스로 병렬 실행(상위 셸 `&` + `wait`). 연도별 분리 검증 포함(CLAUDE.md).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

import engine.backtest as eb
from data.loader import DataLoader
from scripts.run_backtest import build_engine

CONFIG = "config/merged_noblock_sleeve.yaml"

PERIODS = [
    ("full",    "2022-01-01", "2026-04-14"),
    ("2022",    "2022-01-01", "2023-01-01"),
    ("2023",    "2023-01-01", "2024-01-01"),
    ("2024",    "2024-01-01", "2025-01-01"),
    ("2025-26", "2025-01-01", "2026-04-14"),
]

VARIANTS = [
    ("쿨다운없음(0h)", 0.0, False),
    ("TP 6h(현재)",    6.0, False),
    ("TP+SL 6h",       6.0, True),
]

_orig_cwr = eb.BacktestEngine._close_with_reason


def _patched_cwr(self, sym, pos, exit_price, exit_reason, snapshot, regime):
    _orig_cwr(self, sym, pos, exit_price, exit_reason, snapshot, regime)
    if exit_reason == "sl":  # 손절도 쿨다운 기록 (TP는 _orig가 이미 기록)
        self.guards.record_tp(sym, pos.strategy, snapshot.timestamp)


def _run_one(p0, loader, cooldown, since, until) -> dict:
    p = copy.deepcopy(p0)
    p.setdefault("risk", {})["tp_cooldown_hours"] = cooldown
    eng = build_engine(p, 100.0)
    rep = eng.run(loader.iterate(
        since=pd.Timestamp(since, tz="UTC"),
        until=pd.Timestamp(until, tz="UTC"),
    ))
    d = rep.to_dict()
    d.pop("strategy_breakdown", None)
    d["bankrupt"] = rep.bankrupt
    return d


def run_variant(vi: int) -> None:
    p0 = yaml.safe_load(open(CONFIG))
    dc = p0.get("data", {})
    loader = DataLoader(
        symbols=p0["symbols"], timeframes=p0["timeframes"],
        primary_tf=p0.get("primary_timeframe", "1h"),
        cache_dir=dc.get("cache_dir", "data/cache"),
        lookback=dc.get("lookback_bars", 300),
    )
    vname, cd, sl_patch = VARIANTS[vi]
    if sl_patch:
        eb.BacktestEngine._close_with_reason = _patched_cwr
    out: dict = {}
    try:
        for pname, since, until in PERIODS:
            d = _run_one(p0, loader, cd, since, until)
            out[f"{vname}|{pname}"] = d
            print(f"[done] {vname:14s} | {pname:8s}: "
                  f"final=${d['final_equity']:>10,.0f}  Sh={d['sharpe']:>5.2f}  "
                  f"MDD={d['max_drawdown']:>6.1f}%  tr={d['total_trades']:>4}  "
                  f"WR={d['win_rate']:.0f}%  PF={d['profit_factor']:.2f}"
                  f"{'  [BANKRUPT]' if d['bankrupt'] else ''}", flush=True)
    finally:
        if sl_patch:
            eb.BacktestEngine._close_with_reason = _orig_cwr
    json.dump(out, open(f"/tmp/cooldown_v{vi}.json", "w"), ensure_ascii=False, indent=2)


def aggregate() -> None:
    results: dict = {}
    for vi in range(len(VARIANTS)):
        results.update(json.load(open(f"/tmp/cooldown_v{vi}.json")))
    json.dump(results, open("/tmp/cooldown_variants.json", "w"), ensure_ascii=False, indent=2)

    print("\n" + "=" * 92)
    print("쿨다운 변형 비교 (merged_noblock_sleeve, $100 시작)")
    print("=" * 92)
    for pname, _, _ in PERIODS:
        print(f"\n── {pname} ──")
        print(f"  {'변형':16s} {'최종$':>12s} {'수익%':>10s} {'Sharpe':>7s} {'MDD%':>8s} "
              f"{'거래':>5s} {'WR%':>5s} {'PF':>5s}")
        for vname, _, _ in VARIANTS:
            d = results[f"{vname}|{pname}"]
            print(f"  {vname:16s} {d['final_equity']:>12,.0f} {d['total_return_pct']:>+9.0f}% "
                  f"{d['sharpe']:>7.2f} {d['max_drawdown']:>+7.1f}% {d['total_trades']:>5} "
                  f"{d['win_rate']:>4.0f}% {d['profit_factor']:>5.2f}"
                  f"{'  BANKRUPT' if d['bankrupt'] else ''}")
    print("\n저장: /tmp/cooldown_variants.json")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    if arg == "agg":
        aggregate()
    else:
        run_variant(int(arg))
