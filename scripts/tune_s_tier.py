"""S 티어 튜닝 실험 — 시그널 리플레이 병렬 실행.

3가지 실험:
  1) S 사이징 축소: 25x/22% → 15x/12%
  2) S를 A로 병합: 25x/22% → 10x/10%
  3) ADAUSDT S 차단: score 4 + ADAUSDT 시그널 제거
"""
from __future__ import annotations

import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd

from data.loader import DataLoader
from scripts.run_backtest import build_engine


SIGNAL_PATH = "data/signals_dump.parquet"
CONFIG_PATH = "config/final_v13_eth.yaml"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _run_experiment(name: str, config_patch: dict | None = None,
                    signal_filter: dict | None = None) -> dict:
    """단일 실험 실행 (별도 프로세스에서 호출됨)."""
    p = _load_config()
    if config_patch:
        for key_path, value in config_patch.items():
            parts = key_path.split(".")
            d = p
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value

    bt = p.get("backtest", {})
    capital = bt.get("initial_capital", 100)
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until_str = bt.get("end")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    signals_df = pd.read_parquet(SIGNAL_PATH)
    if signal_filter:
        for filt in signal_filter:
            cond = pd.Series(True, index=signals_df.index)
            for col, val in filt.items():
                cond &= (signals_df[col] == val)
            signals_df = signals_df[~cond]

    engine = build_engine(p, capital)
    t0 = time.perf_counter()
    report = engine.run_replay(signals_df, loader.iterate(since=since, until=until))
    elapsed = time.perf_counter() - t0

    # 티어별 승률 계산
    df = engine.tracker.ledger.to_dataframe()
    tier_stats = {}
    if not df.empty:
        df["win"] = df["pnl"] > 0
        for tier_name, min_score, max_score in [("SS", 5, 99), ("S", 4, 4), ("A", 3, 3)]:
            sub = df[(df["confluence_score"] >= min_score) & (df["confluence_score"] <= max_score)]
            if len(sub) > 0:
                tier_stats[tier_name] = {
                    "trades": len(sub),
                    "wr": sub["win"].mean() * 100,
                    "pnl": sub["pnl"].sum(),
                    "pf": abs(sub[sub["pnl"] > 0]["pnl"].sum() / sub[sub["pnl"] < 0]["pnl"].sum())
                         if sub[sub["pnl"] < 0]["pnl"].sum() != 0 else float("inf"),
                }

    return {
        "name": name,
        "trades": report.total_trades,
        "equity": report.final_equity,
        "sharpe": report.sharpe,
        "mdd": report.max_drawdown,
        "wr": report.win_rate,
        "pf": report.profit_factor,
        "elapsed": elapsed,
        "tier_stats": tier_stats,
    }


# ── 실험 정의 ─────────────────────────────────────────────────
EXPERIMENTS = {
    "baseline": {
        "config_patch": None,
        "signal_filter": None,
    },
    "S_size_down": {
        "config_patch": {
            "leverage_tiers.S.leverage": 15,
            "leverage_tiers.S.size_fraction": 0.12,
        },
        "signal_filter": None,
    },
    "S_merge_to_A": {
        "config_patch": {
            "leverage_tiers.S.leverage": 10,
            "leverage_tiers.S.size_fraction": 0.1,
        },
        "signal_filter": None,
    },
    "S_block_ADA": {
        "config_patch": None,
        "signal_filter": [{"symbol": "ADAUSDT", "tier": "S"}],
    },
}


def main() -> None:
    print(f"시그널 파일: {SIGNAL_PATH}")
    sig = pd.read_parquet(SIGNAL_PATH)
    print(f"총 시그널: {len(sig)}개")
    print(f"실험 {len(EXPERIMENTS)}개 병렬 실행 시작\n")

    results = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_experiment, name, **kwargs): name
            for name, kwargs in EXPERIMENTS.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"  ✓ {name} 완료 ({results[name]['elapsed']:.0f}s)")
            except Exception as e:
                print(f"  ✗ {name} 실패: {e}")

    # ── 결과 비교 ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'실험':<16} {'거래':>5} {'최종자산':>14} {'Sharpe':>7} {'MDD':>7} {'WR':>6} {'PF':>6}")
    print("-" * 80)
    for name in ["baseline", "S_size_down", "S_merge_to_A", "S_block_ADA"]:
        r = results.get(name)
        if r is None:
            continue
        print(f"{r['name']:<16} {r['trades']:>5} ${r['equity']:>12,.2f} {r['sharpe']:>7.3f} "
              f"{r['mdd']:>6.1f}% {r['wr']:>5.1f}% {r['pf']:>6.2f}")

    # 티어별 상세
    print("\n" + "=" * 80)
    print("티어별 상세:")
    for name in ["baseline", "S_size_down", "S_merge_to_A", "S_block_ADA"]:
        r = results.get(name)
        if r is None:
            continue
        print(f"\n  [{r['name']}]")
        for tier in ["SS", "S", "A"]:
            ts = r["tier_stats"].get(tier)
            if ts:
                print(f"    {tier}: {ts['trades']}건, WR {ts['wr']:.1f}%, "
                      f"PF {ts['pf']:.2f}, PnL ${ts['pnl']:,.2f}")


if __name__ == "__main__":
    main()
