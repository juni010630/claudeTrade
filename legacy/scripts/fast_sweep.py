"""fast_sweep.py — 진입 타점 고정 후 TP/SL 배수 스윕 (리플레이 모드).

흐름:
  Phase 1 (한 번): 풀 백테스트 → 실제 체결 진입 기록 → entries.parquet 저장
  Phase 2 (N개 병렬): entries.parquet 로드 → ATR 역산 → TP/SL 재계산 → 리플레이

Usage:
    # Phase 1+2 (덤프 + 스윕):
    python scripts/fast_sweep.py

    # Phase 2만 (기존 덤프 재사용):
    python scripts/fast_sweep.py --entries data/entries.parquet

    # 기간 지정:
    python scripts/fast_sweep.py --start 2024-01-01 --end 2025-12-31
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


# ── 리플레이 워커 (ProcessPoolExecutor용, 모듈 최상위 필수) ──────────────────
def _replay_worker(args: tuple) -> dict:
    (config_path, entries_path, initial_capital,
     since_str, until_str, tp_mult, sl_mult, orig_sl_mult) = args

    with open(config_path) as f:
        p = yaml.safe_load(f)

    df = pd.read_parquet(entries_path).copy()

    # ATR 역산 (sl_dist = orig_sl_mult * ATR)
    atr = df["sl_dist"] / orig_sl_mult

    long = df["direction"] == "long"
    df.loc[long,  "tp_price"] = df.loc[long,  "entry_price"] + tp_mult * atr[long]
    df.loc[~long, "tp_price"] = df.loc[~long, "entry_price"] - tp_mult * atr[~long]
    df.loc[long,  "sl_price"] = df.loc[long,  "entry_price"] - sl_mult * atr[long]
    df.loc[~long, "sl_price"] = df.loc[~long, "entry_price"] + sl_mult * atr[~long]

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

    return {
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "sharpe": round(report.sharpe, 3),
        "max_dd_%": round(report.max_drawdown * 100, 1),
        "win_rate_%": round(report.win_rate, 1),
        "profit_factor": round(getattr(report, "profit_factor", float("nan")), 2),
        "final_equity": round(report.final_equity, 0),
        "trades": report.total_trades,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params",   default="config/final_v13_eth.yaml")
    parser.add_argument("--entries",  default=None,
                        help="기존 entries.parquet 경로 (지정 시 Phase 1 스킵)")
    parser.add_argument("--out",      default="data/entries.parquet",
                        help="Phase 1 덤프 저장 경로")
    parser.add_argument("--start",    default=None)
    parser.add_argument("--end",      default=None)
    parser.add_argument("--capital",  type=float, default=None)
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = args.capital or bt.get("initial_capital", 100)
    since_str = args.start or bt.get("start", "2022-01-01")
    until_str = args.end   or bt.get("end")

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})

    # ── Phase 1: 풀 백테스트 + 진입 덤프 ────────────────────────────────────
    if args.entries is None:
        print(f"[Phase 1] 풀 백테스트 실행 중... ({since_str} ~ {until_str or '최신'})")
        loader = DataLoader(
            symbols=p["symbols"],
            timeframes=p["timeframes"],
            primary_tf=p.get("primary_timeframe", "1h"),
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            lookback=data_cfg.get("lookback_bars", 300),
        )
        engine = build_engine(p, initial_capital)
        report, entries_df = engine.run_fill_dump(loader.iterate(since=since, until=until))

        if entries_df.empty:
            print("체결된 진입이 없습니다. 설정 확인 필요.")
            return

        entries_df.to_parquet(args.out, index=False)
        print(f"  → {len(entries_df)}건 진입 기록 저장: {args.out}")
        print(f"  → 기준 성과: Sharpe {report.sharpe:.2f}, "
              f"MDD {report.max_drawdown*100:.1f}%, "
              f"최종 ${report.final_equity:,.0f}")
        entries_path = args.out
    else:
        entries_path = args.entries
        entries_df = pd.read_parquet(entries_path)
        print(f"[Phase 1 스킵] 기존 덤프 사용: {entries_path} ({len(entries_df)}건)")

    # ── 스윕 그리드 정의 ────────────────────────────────────────────────────
    # orig_sl_mult: 원래 SL ATR 배수 (ATR 역산용). config에서 읽거나 1.8 기본값.
    orig_sl_mult = (
        p.get("strategies", {}).get("ema_cross", {}).get("atr_sl_mult", 1.8)
    )

    tp_mults = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]   # ← 여기서 조정
    sl_mults = [1.2, 1.5, 1.8, 2.2, 2.5]               # ← 여기서 조정

    combos = [(tp, sl) for tp in tp_mults for sl in sl_mults]
    print(f"\n[Phase 2] {len(combos)}개 조합 리플레이 스윕 (workers={args.workers})")

    worker_args = [
        (args.params, entries_path, initial_capital,
         since_str, until_str, tp, sl, orig_sl_mult)
        for tp, sl in combos
    ]

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_replay_worker, a): a for a in worker_args}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                row = fut.result()
                results.append(row)
                print(f"  [{done}/{len(combos)}] TP={row['tp_mult']} SL={row['sl_mult']} "
                      f"→ Sharpe {row['sharpe']:.2f}  MDD {row['max_dd_%']:.1f}%  "
                      f"WR {row['win_rate_%']:.1f}%  ${row['final_equity']:,.0f}")
            except Exception as e:
                a = futures[fut]
                print(f"  [{done}/{len(combos)}] TP={a[5]} SL={a[6]} 실패: {e}")

    if not results:
        return

    df_res = pd.DataFrame(results).sort_values(["sharpe"], ascending=False)

    print("\n" + "=" * 90)
    print("FAST SWEEP 결과 (Sharpe 내림차순)")
    print("=" * 90)
    print(df_res.to_string(index=False))
    print("=" * 90)

    # Sharpe 히트맵 출력 (TP × SL 행렬)
    pivot = df_res.pivot(index="sl_mult", columns="tp_mult", values="sharpe")
    print("\n[Sharpe 히트맵 — 행: SL 배수, 열: TP 배수]")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
