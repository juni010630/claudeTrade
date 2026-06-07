"""TP 쿨다운 최적화 — 시그널 리플레이 병렬 실행."""
from __future__ import annotations

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

def _run_experiment(name: str, cooldown_hours: float) -> dict:
    p = _load_config()
    
    # 쿨다운 설정 오버라이드
    if "risk" not in p:
        p["risk"] = {}
    p["risk"]["tp_cooldown_hours"] = cooldown_hours

    bt = p.get("backtest", {})
    capital = bt.get("initial_capital", 100)
    since = pd.Timestamp("2022-01-01", tz="UTC")
    until = pd.Timestamp("2026-06-03", tz="UTC")

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    signals_df = pd.read_parquet(SIGNAL_PATH)
    engine = build_engine(p, capital)
    
    t0 = time.perf_counter()
    report = engine.run_replay(signals_df, loader.iterate(since=since, until=until))
    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "cooldown": cooldown_hours,
        "trades": report.total_trades,
        "equity": report.final_equity,
        "sharpe": report.sharpe,
        "mdd": report.max_drawdown,
        "wr": report.win_rate,
        "pf": report.profit_factor,
        "elapsed": elapsed,
    }

def main() -> None:
    cooldowns = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    
    print(f"시그널 파일: {SIGNAL_PATH}")
    sig = pd.read_parquet(SIGNAL_PATH)
    print(f"총 시그널: {len(sig)}개")
    print(f"TP Cooldown 최적화 시작: {cooldowns}시간\n")

    results = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_run_experiment, f"CD_{cd}h", cd): f"CD_{cd}h"
            for cd in cooldowns
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"  ✓ {name} 완료 ({results[name]['elapsed']:.1f}s)")
            except Exception as e:
                print(f"  ✗ {name} 실패: {e}")

    # 결과 정렬 및 출력
    print("\n" + "=" * 90)
    print(f"{'쿨다운':<10} {'거래수':>8} {'최종자산':>15} {'Sharpe':>8} {'MDD':>8} {'WR':>7} {'PF':>7}")
    print("-" * 90)
    
    sorted_results = sorted(results.values(), key=lambda x: x["cooldown"])
    for r in sorted_results:
        print(f"{r['cooldown']:>8.1f}h {r['trades']:>8} ${r['equity']:>14,.2f} {r['sharpe']:>8.3f} "
              f"{r['mdd']:>7.1f}% {r['wr']:>6.1f}% {r['pf']:>7.2f}")
    print("=" * 90)

if __name__ == "__main__":
    main()