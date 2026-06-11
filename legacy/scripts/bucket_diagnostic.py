"""버킷 진단 스크립트 — 전체 백테스트 1회 실행 후 거래 기록을 다양한 축으로 분해.

Usage:
    python scripts/bucket_diagnostic.py
    python scripts/bucket_diagnostic.py --params config/final_v13_eth.yaml --start 2022-01-01
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


# ── 티어 분류 (final_v13_eth 기준) ─────────────────────────────────────────
def score_to_tier(score: int) -> str:
    if score >= 5:
        return "SS"
    elif score >= 4:
        return "S"
    elif score >= 3:
        return "A"
    return "B"


# ── 버킷 통계 계산 ─────────────────────────────────────────────────────────
def bucket_stats(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    def _stats(g: pd.DataFrame) -> pd.Series:
        wins = (g["pnl"] > 0).sum()
        losses = (g["pnl"] <= 0).sum()
        gross_win = g.loc[g["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(g.loc[g["pnl"] <= 0, "pnl"].sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        wr = wins / len(g) if len(g) > 0 else 0
        return pd.Series({
            "count": len(g),
            "WR%": round(wr * 100, 1),
            "PF": round(pf, 2),
            "avg_pnl": round(g["pnl"].mean(), 2),
            "total_pnl": round(g["pnl"].sum(), 2),
            "wins": wins,
            "losses": losses,
        })

    result = df.groupby(group_col).apply(_stats).reset_index()
    result = result.sort_values("total_pnl", ascending=False)
    return result


def flag(row: pd.Series) -> str:
    """버킷이 나쁜지 표시."""
    flags = []
    if row["PF"] < 1.0:
        flags.append("PF<1")
    if row["WR%"] < 40:
        flags.append("WR<40%")
    if row["count"] >= 10 and (row["PF"] < 1.0 or row["WR%"] < 40):
        flags.append("★차단후보")
    return " ".join(flags)


def print_bucket(title: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["flag"] = df.apply(flag, axis=1)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = args.capital or bt.get("initial_capital", 100)
    since_str = args.start or bt.get("start", "2022-01-01")
    until_str = args.end or bt.get("end")

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    engine = build_engine(p, initial_capital)

    print(f"백테스트 실행 중: {since_str} ~ {until_str or '최신'}")
    snapshots = loader.iterate(since=since, until=until)
    report = engine.run(snapshots)
    report.print_summary()

    # ── DataFrame 추출 ────────────────────────────────────────────────────
    df = engine.ledger.to_dataframe()
    if df.empty:
        print("거래 기록 없음")
        return

    print(f"\n총 거래: {len(df)}건")

    # 파생 컬럼
    df["tier"] = df["confluence_score"].apply(score_to_tier)
    df["hold_hours"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
    df["day_of_week"] = df["entry_time"].dt.day_name()
    df["regime_str"] = df["regime_at_entry"].apply(
        lambda r: r.regime.value if hasattr(r, "regime") else str(r)
    )

    # hold-time 버킷 (0-12h / 12-48h / 48-96h / 96-336h)
    bins = [0, 12, 48, 96, 336, float("inf")]
    labels = ["0-12h", "12-48h", "48-96h", "96-336h", "336h+"]
    df["hold_bucket"] = pd.cut(df["hold_hours"], bins=bins, labels=labels)

    # ── 버킷별 출력 ───────────────────────────────────────────────────────
    print_bucket("1. 티어별 (SS/S/A)", bucket_stats(df, "tier"))
    print_bucket("2. 심볼별", bucket_stats(df, "symbol"))
    print_bucket("3. 전략별", bucket_stats(df, "strategy"))
    print_bucket("4. 레짐별 (trending/ranging/pre_breakout)", bucket_stats(df, "regime_str"))
    print_bucket("5. 방향별 (long/short)", bucket_stats(df, "direction"))
    print_bucket("6. 청산 이유별 (tp/sl/timeout)", bucket_stats(df, "exit_reason"))
    print_bucket("7. 요일별", bucket_stats(df, "day_of_week"))
    print_bucket("8. 보유 시간 버킷", bucket_stats(df, "hold_bucket"))

    # ── 교차 분석: 티어 × 전략 ────────────────────────────────────────────
    df["tier_x_strategy"] = df["tier"] + " | " + df["strategy"]
    print_bucket("9. 티어 × 전략 교차", bucket_stats(df, "tier_x_strategy"))

    # ── 교차 분석: 심볼 × 방향 ────────────────────────────────────────────
    df["symbol_x_dir"] = df["symbol"] + " " + df["direction"]
    print_bucket("10. 심볼 × 방향 교차", bucket_stats(df, "symbol_x_dir"))

    # ── timeout 청산만 따로 분석 ──────────────────────────────────────────
    timeout_df = df[df["exit_reason"] == "timeout"]
    if not timeout_df.empty:
        print(f"\n{'='*60}")
        print(f"  11. Timeout 청산 상세 ({len(timeout_df)}건)")
        print(f"{'='*60}")
        print(f"  avg hold: {timeout_df['hold_hours'].mean():.1f}h")
        print(f"  avg PnL:  ${timeout_df['pnl'].mean():.2f}")
        print(f"  WR%:      {(timeout_df['pnl'] > 0).mean()*100:.1f}%")
        print(f"  total PnL: ${timeout_df['pnl'].sum():.2f}")
        print("\n  timeout 심볼별:")
        print(bucket_stats(timeout_df, "symbol").to_string(index=False))

    # ── 차단 후보 요약 ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  차단 후보 요약 (count≥10 AND PF<1.0 or WR<40%)")
    print(f"{'='*60}")
    all_buckets = [
        ("tier", bucket_stats(df, "tier")),
        ("symbol", bucket_stats(df, "symbol")),
        ("regime", bucket_stats(df, "regime_str")),
        ("direction", bucket_stats(df, "direction")),
        ("day_of_week", bucket_stats(df, "day_of_week")),
        ("hold_bucket", bucket_stats(df, "hold_bucket")),
        ("tier_x_strategy", bucket_stats(df, "tier_x_strategy")),
        ("symbol_x_dir", bucket_stats(df, "symbol_x_dir")),
    ]
    found_any = False
    for axis, bdf in all_buckets:
        bad = bdf[(bdf["count"] >= 10) & ((bdf["PF"] < 1.0) | (bdf["WR%"] < 40))]
        for _, row in bad.iterrows():
            print(f"  [{axis}] {row.iloc[0]}: count={int(row['count'])}, WR={row['WR%']}%, PF={row['PF']}, total_pnl=${row['total_pnl']:.0f}")
            found_any = True
    if not found_any:
        print("  없음 — 모든 주요 버킷이 WR≥40% & PF≥1.0")


if __name__ == "__main__":
    main()
