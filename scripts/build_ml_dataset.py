"""ML 학습 데이터셋 빌더.

백테스트 1회 실행 → ledger(pnl) + entries_df 조인 + 피처 계산
→ data/ml_dataset.parquet 저장.

Usage:
    python scripts/build_ml_dataset.py
    python scripts/build_ml_dataset.py --params config/final_v13_eth.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine
from strategies.ml_filter import compute_features, FEATURE_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml")
    parser.add_argument("--out", default="data/ml_dataset.parquet")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end", "2026-04-14"), tz="UTC")

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    # ML 비활성 상태로 순수 백테 실행 (피처 학습용 — ML 개입 없이 진입 그대로)
    p_clean = _disable_ml(p)
    engine = build_engine(p_clean, initial_capital)
    snapshots = loader.iterate(since=since, until=until)

    print(f"백테스트 실행 중: {since.date()} ~ {until.date()}")
    report, entries_df = engine.run_fill_dump(snapshots)
    ledger_df = engine.ledger.to_dataframe()

    print(f"  진입 수(entries_df): {len(entries_df)}")
    print(f"  거래 수(ledger):     {len(ledger_df)}")
    report.print_summary()

    if entries_df.empty or ledger_df.empty:
        print("[ERROR] 데이터 없음 — 종료")
        sys.exit(1)

    # ── 조인: entries_df.timestamp == ledger_df.entry_time ──────────
    ledger_df["timestamp"] = pd.to_datetime(ledger_df["entry_time"], utc=True)
    entries_df["timestamp"] = pd.to_datetime(entries_df["timestamp"], utc=True)

    joined = entries_df.merge(
        ledger_df[["timestamp", "symbol", "strategy", "direction", "pnl", "exit_reason"]],
        on=["timestamp", "symbol", "strategy", "direction"],
        how="inner",
    )
    print(f"  조인 후 행 수: {len(joined)}  (entries={len(entries_df)}, ledger={len(ledger_df)})")

    mismatch = len(ledger_df) - len(joined)
    if abs(mismatch) > 5:
        print(f"  [WARN] 조인 불일치 {mismatch}건 — symbol_block_directions 등에 의한 정상적 차이일 수 있음")

    # ── 피처 계산 ──────────────────────────────────────────────────
    print("피처 계산 중...")
    primary_tf = p.get("primary_timeframe", "1h")
    primary_delta = pd.Timedelta(primary_tf)

    rows = []
    for _, row in joined.iterrows():
        effective_time = row["timestamp"]
        sym = row["symbol"]

        # DataLoader와 동일한 masking: close_time (open + delta) <= effective_time
        bars_1h = _slice_bars(loader._ohlcv[sym]["1h"], effective_time, "1h")
        bars_4h = _slice_bars(loader._ohlcv[sym]["4h"], effective_time, "4h")
        bars_1d = _slice_bars(loader._ohlcv[sym]["1d"], effective_time, "1d")

        feat = compute_features(bars_1h, bars_4h, bars_1d, len(bars_1h) - 1)
        if feat is None:
            continue

        # 컨텍스트 피처
        feat["direction_long"] = 1 if row["direction"] == "long" else 0
        feat["strategy_ema"] = 1 if row["strategy"] == "ema_cross" else 0

        # 펀딩비 (로더 캐시에서)
        fd = loader._funding.get(sym, pd.DataFrame())
        if not fd.empty:
            mask_f = fd.index <= effective_time
            feat["funding"] = float(fd[mask_f]["rate"].iloc[-1]) if mask_f.any() else 0.0
        else:
            feat["funding"] = 0.0

        feat["timestamp"] = effective_time
        feat["symbol"] = sym
        feat["strategy"] = row["strategy"]
        feat["direction"] = row["direction"]
        feat["score"] = row.get("score", 0)
        feat["tier"] = row.get("tier", "")
        feat["pnl"] = row["pnl"]
        feat["exit_reason"] = row["exit_reason"]
        feat["y"] = int(row["pnl"] > 0)
        rows.append(feat)

    if not rows:
        print("[ERROR] 피처 계산 실패 (데이터 부족?) — 종료")
        sys.exit(1)

    ml_df = pd.DataFrame(rows)
    out = Path(args.out)
    ml_df.to_parquet(out, index=False)

    n = len(ml_df)
    win_rate = ml_df["y"].mean()
    print(f"\n[완료] {out} 저장: {n}행, 승률 {win_rate:.1%}")
    print(f"피처 컬럼: {FEATURE_COLS}")
    print(f"NaN 개수:\n{ml_df[FEATURE_COLS].isna().sum().to_string()}")

    # 게이트 체크
    assert n > 0, "행 수 0"
    assert 0.3 < win_rate < 0.7, f"승률 이상: {win_rate:.1%}"
    assert ml_df[FEATURE_COLS].isna().sum().sum() == 0, "NaN 존재"
    print("\n[GATE] 통과 ✓")


def _disable_ml(p: dict) -> dict:
    """ML 비활성화한 복사본 반환."""
    import copy
    p2 = copy.deepcopy(p)
    p2.setdefault("scorer", {}).setdefault("ml_soft_scoring", {})["enabled"] = False
    return p2


def _slice_bars(full_df: pd.DataFrame, effective_time: pd.Timestamp, tf: str) -> pd.DataFrame:
    """DataLoader 마스킹과 동일: close_time = open + tf_delta <= effective_time."""
    tf_delta = pd.Timedelta(tf)
    mask = (full_df.index + tf_delta) <= effective_time
    return full_df[mask].iloc[-300:].reset_index()


if __name__ == "__main__":
    main()
