"""delay_entry_sweep.py — 시그널 후 N봉 지연 진입 스윕.

시그널 발생 후 N봉(1h 기준) 이후에 진입했을 때 성과 비교.
TP/SL ATR 거리는 원래 시그널 기준으로 유지 (shift로 새 entry price에 자동 적용).

Usage:
    python scripts/delay_entry_sweep.py                        # 기본
    python scripts/delay_entry_sweep.py --entries data/entries.parquet
    python scripts/delay_entry_sweep.py --tp 3.5 --sl 1.8 --workers 8
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


# ── 워커 (ProcessPoolExecutor용) ─────────────────────────────────────────────
def _worker(args: tuple) -> dict:
    (config_path, entries_path, initial_capital,
     since_str, until_str, n_bars, tp_mult, sl_mult, orig_sl_mult) = args

    with open(config_path) as f:
        p = yaml.safe_load(f)

    df = pd.read_parquet(entries_path).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # TP/SL 재계산 (ATR 거리 기준)
    atr = df["sl_dist"] / orig_sl_mult
    long = df["direction"] == "long"
    df.loc[long,  "tp_price"] = df.loc[long,  "entry_price"] + tp_mult * atr[long]
    df.loc[~long, "tp_price"] = df.loc[~long, "entry_price"] - tp_mult * atr[~long]
    df.loc[long,  "sl_price"] = df.loc[long,  "entry_price"] - sl_mult * atr[long]
    df.loc[~long, "sl_price"] = df.loc[~long, "entry_price"] + sl_mult * atr[~long]

    # N봉 지연: timestamp를 N시간 뒤로 이동
    # replay 엔진이 해당 시점의 market price로 entry하고 TP/SL은 shift 적용
    df["timestamp"] = df["timestamp"] + pd.Timedelta(hours=n_bars)

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    engine = build_engine(p, initial_capital)
    report = engine.run_replay(df, loader.iterate(since=since, until=until))

    # 평균 홀딩시간 계산
    records = engine.ledger.records
    if records:
        hold_hours = [
            (r.exit_time - r.entry_time).total_seconds() / 3600
            for r in records
        ]
        avg_hold = sum(hold_hours) / len(hold_hours)
    else:
        avg_hold = 0.0

    return {
        "delay_bars": n_bars,
        "sharpe": round(report.sharpe, 3),
        "max_dd_%": round(report.max_drawdown * 100, 1),
        "win_rate_%": round(report.win_rate, 1),
        "profit_factor": round(getattr(report, "profit_factor", float("nan")), 2),
        "final_equity": round(report.final_equity, 0),
        "trades": report.total_trades,
        "avg_hold_h": round(avg_hold, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",  default="config/final_v13_eth.yaml")
    parser.add_argument("--entries", default="data/entries.parquet")
    parser.add_argument("--start",   default=None)
    parser.add_argument("--end",     default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--tp",      type=float, default=3.5)
    parser.add_argument("--sl",      type=float, default=1.8)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = args.capital or bt.get("initial_capital", 100)
    since_str = args.start or bt.get("start", "2022-01-01")
    until_str = args.end   or bt.get("end")

    orig_sl_mult = (
        p.get("strategies", {}).get("ema_cross", {}).get("atr_sl_mult", 1.8)
    )

    # ── Phase 1: N=0 기준선 → 평균 홀딩시간 계산 ────────────────────────────
    print("[Phase 1] N=0 기준선 실행 중 (평균 홀딩시간 계산)...")
    baseline = _worker((
        args.params, args.entries, initial_capital,
        since_str, until_str, 0, args.tp, args.sl, orig_sl_mult,
    ))
    avg_hold_h = baseline["avg_hold_h"]
    max_n = max(1, int(avg_hold_h))

    print(f"  → N=0: Sharpe {baseline['sharpe']:.2f}  "
          f"WR {baseline['win_rate_%']:.1f}%  "
          f"최종 ${baseline['final_equity']:,.0f}  "
          f"평균홀딩 {avg_hold_h:.1f}h")
    print(f"\n[Phase 2] N=1..{max_n}봉 스윕 ({max_n}개 조합, workers={args.workers})")

    # ── Phase 2: N=1..max_n 병렬 스윕 ───────────────────────────────────────
    worker_args = [
        (args.params, args.entries, initial_capital,
         since_str, until_str, n, args.tp, args.sl, orig_sl_mult)
        for n in range(1, max_n + 1)
    ]

    results = [baseline]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, a): a for a in worker_args}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                row = fut.result()
                results.append(row)
                print(f"  [{done:>3}/{max_n}] N={row['delay_bars']:>3}봉  "
                      f"Sharpe {row['sharpe']:.2f}  "
                      f"WR {row['win_rate_%']:.1f}%  "
                      f"거래 {row['trades']}건  "
                      f"${row['final_equity']:,.0f}")
            except Exception as e:
                a = futures[fut]
                print(f"  [{done:>3}/{max_n}] N={a[5]}봉 실패: {e}")

    df_res = pd.DataFrame(results).sort_values("delay_bars").reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"지연 진입 스윕 결과  (TP={args.tp}x / SL={args.sl}x / 기준선 N=0)")
    print("=" * 80)
    print(df_res[["delay_bars","sharpe","win_rate_%","profit_factor",
                  "final_equity","trades","avg_hold_h"]].to_string(index=False))

    best = df_res.loc[df_res["sharpe"].idxmax()]
    print("\n" + "=" * 80)
    print(f"최적 지연: N={int(best['delay_bars'])}봉  "
          f"Sharpe {best['sharpe']:.2f}  "
          f"WR {best['win_rate_%']:.1f}%  "
          f"${best['final_equity']:,.0f}")
    print(f"기준선 N=0: Sharpe {baseline['sharpe']:.2f}  "
          f"WR {baseline['win_rate_%']:.1f}%  "
          f"${baseline['final_equity']:,.0f}")

    # Sharpe 시각화 (간단한 ASCII)
    print("\n[Sharpe vs 지연 봉 수]")
    max_sharpe = df_res["sharpe"].max()
    for _, row in df_res.iterrows():
        bar_len = int(row["sharpe"] / max_sharpe * 40)
        marker = " ← 최적" if row["delay_bars"] == best["delay_bars"] else ""
        print(f"  N={int(row['delay_bars']):>3}  {'█' * bar_len:<40} {row['sharpe']:.2f}{marker}")


if __name__ == "__main__":
    main()
