"""검증: 풀 백테스트(dump) vs 리플레이 결과 일치 확인.

Usage:
    python scripts/verify_replay.py
    python scripts/verify_replay.py --params config/final_v13_eth.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(args.start or bt.get("start", "2022-01-01"), tz="UTC")
    until_str = args.end or bt.get("end")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})
    loader_kwargs = dict(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    # ── Phase 1: dump (풀 백테스트 + 시그널 수집) ────────────────
    print("=" * 60)
    print("Phase 1: 풀 백테스트 (dump 모드)")
    print("=" * 60)

    engine_dump = build_engine(p, capital)
    loader = DataLoader(**loader_kwargs)

    t0 = time.perf_counter()
    report_dump, signals_df = engine_dump.run_dump(loader.iterate(since=since, until=until))
    t_dump = time.perf_counter() - t0

    print(f"  시간: {t_dump:.1f}s")
    print(f"  시그널 수: {len(signals_df)}")
    print(f"  거래 수: {report_dump.total_trades}")
    print(f"  최종 자산: ${report_dump.final_equity:,.2f}")
    print(f"  Sharpe: {report_dump.sharpe:.3f}")
    print()

    # 시그널 저장
    sig_path = Path("data/signals_dump.parquet")
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(sig_path, index=False)
    print(f"  시그널 저장: {sig_path} ({sig_path.stat().st_size / 1024:.0f} KB)")
    print()

    # ── Phase 2: replay (시그널 리플레이) ────────────────────────
    print("=" * 60)
    print("Phase 2: 리플레이 (동일 설정)")
    print("=" * 60)

    engine_replay = build_engine(p, capital)
    loader2 = DataLoader(**loader_kwargs)
    loaded_signals = pd.read_parquet(sig_path)

    t0 = time.perf_counter()
    report_replay = engine_replay.run_replay(loaded_signals, loader2.iterate(since=since, until=until))
    t_replay = time.perf_counter() - t0

    print(f"  시간: {t_replay:.1f}s")
    print(f"  거래 수: {report_replay.total_trades}")
    print(f"  최종 자산: ${report_replay.final_equity:,.2f}")
    print(f"  Sharpe: {report_replay.sharpe:.3f}")
    print()

    # ── 비교 ────────────────────────────────────────────────────
    print("=" * 60)
    print("검증 결과")
    print("=" * 60)

    eq_diff = abs(report_dump.final_equity - report_replay.final_equity)
    trade_diff = report_dump.total_trades - report_replay.total_trades
    sharpe_diff = abs(report_dump.sharpe - report_replay.sharpe)

    print(f"  최종 자산 차이:  ${eq_diff:.6f}")
    print(f"  거래 수 차이:    {trade_diff}")
    print(f"  Sharpe 차이:     {sharpe_diff:.6f}")
    print(f"  속도 개선:       {t_dump / t_replay:.1f}x ({t_dump:.1f}s → {t_replay:.1f}s)")
    print()

    if eq_diff < 0.01 and trade_diff == 0:
        print("  ✓ PASS — 결과 완전 일치")
    else:
        print("  ✗ FAIL — 결과 불일치!")
        print(f"    dump:   trades={report_dump.total_trades} equity=${report_dump.final_equity:.2f}")
        print(f"    replay: trades={report_replay.total_trades} equity=${report_replay.final_equity:.2f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
