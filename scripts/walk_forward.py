"""워크포워드 검증 실행.

Usage:
    python scripts/walk_forward.py
    python scripts/walk_forward.py --params config/params.yaml
    python scripts/walk_forward.py --is-months 6 --oos-months 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from data.loader import DataLoader
from engine.walk_forward import WalkForwardValidator
from scripts.run_backtest import build_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/params.yaml", help="파라미터 파일 경로")
    parser.add_argument("--is-months", type=int, default=None, help="In-sample 기간 (개월)")
    parser.add_argument("--oos-months", type=int, default=None, help="Out-of-sample 기간 (개월)")
    parser.add_argument("--start", default=None, help="전체 기간 시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="전체 기간 종료일 (YYYY-MM-DD)")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    wf = p.get("walk_forward", {})
    data_cfg = p.get("data", {})

    since = pd.Timestamp(args.start or bt.get("start", "2024-01-01"), tz="UTC")
    until = pd.Timestamp(args.end or bt.get("end", "2024-12-31"), tz="UTC")
    is_months = args.is_months or wf.get("in_sample_months", 6)
    oos_months = args.oos_months or wf.get("out_sample_months", 2)

    initial_capital = bt.get("initial_capital", 100_000)

    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    def engine_factory():
        return build_engine(p, initial_capital)

    validator = WalkForwardValidator(
        engine_factory=engine_factory,
        loader=loader,
        in_sample_months=is_months,
        out_sample_months=oos_months,
    )

    print(f"워크포워드 검증: IS={is_months}개월, OOS={oos_months}개월")
    print(f"전체 기간: {since.date()} ~ {until.date()}")
    print()

    results = validator.run(since, until)
    WalkForwardValidator.print_results(results)

    if results:
        avg_oos = sum(r.oos_report.sharpe for r in results) / len(results)
        avg_ratio = sum(r.degradation_ratio for r in results) / len(results)
        print(f"\n평균 OOS Sharpe: {avg_oos:.3f}")
        print(f"평균 열화비율:   {avg_ratio:.3f}  (1.0 = 오버피팅 없음)")


if __name__ == "__main__":
    main()
