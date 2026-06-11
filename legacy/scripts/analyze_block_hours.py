"""analyze_block_hours.py — 전략별 시간대별 성과 분석 (block hours 없이 전체 실행).

Usage:
    python scripts/analyze_block_hours.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def main() -> None:
    with open("config/final_v13_eth.yaml") as f:
        p = yaml.safe_load(f)

    # strategy_block_hours 완전 제거 → 전 시간대 거래 허용
    p_no_block = copy.deepcopy(p)
    p_no_block.pop("strategy_block_hours", None)

    bt = p.get("backtest", {})
    since = pd.Timestamp("2022-01-01", tz="UTC")
    until = pd.Timestamp("2026-06-06", tz="UTC")

    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )

    print("백테스트 실행 중 (block_hours 없음)...")
    engine = build_engine(p_no_block, bt.get("initial_capital", 100))
    report = engine.run(loader.iterate(since=since, until=until))

    records = engine.ledger.records
    print(f"총 거래: {len(records)}건  Sharpe {report.sharpe:.2f}\n")

    # 레코드 → DataFrame
    rows = []
    for r in records:
        rows.append({
            "strategy":  r.strategy,
            "hour":      r.entry_time.hour,
            "win":       1 if r.pnl > 0 else 0,
            "pnl":       r.pnl,
            "size_usd":  r.size_usd,
        })
    df = pd.DataFrame(rows)

    current_blocks = p.get("strategy_block_hours", {})

    for strat in sorted(df["strategy"].unique()):
        s = df[df["strategy"] == strat]
        blocked = set(current_blocks.get(strat, []))

        # 시간대별 집계
        agg = (
            s.groupby("hour")
            .agg(count=("win", "count"), wins=("win", "sum"), pnl_sum=("pnl", "sum"))
            .reindex(range(24), fill_value=0)
        )
        agg["win_rate"] = (agg["wins"] / agg["count"].replace(0, pd.NA) * 100).round(1)

        def profit_factor(hour):
            sub = s[s["hour"] == hour]
            pos = sub[sub["pnl"] > 0]["pnl"].sum()
            neg = abs(sub[sub["pnl"] < 0]["pnl"].sum())
            return round(pos / neg, 2) if neg > 0 else float("nan")

        agg["pf"] = [profit_factor(h) for h in range(24)]

        print(f"{'=' * 65}")
        print(f"전략: {strat}   현재 차단: {sorted(blocked)}")
        print(f"{'=' * 65}")
        print(f"{'UTC':>5} {'거래':>5} {'승률':>7} {'PF':>6} {'손익합($)':>11}  {'현재차단':>4}  {'추천':>4}")
        print("-" * 65)

        for hour in range(24):
            cnt = int(agg.loc[hour, "count"])
            if cnt == 0:
                continue
            wr  = agg.loc[hour, "win_rate"]
            pf  = agg.loc[hour, "pf"]
            pnl = agg.loc[hour, "pnl_sum"]

            cur_marker = "●" if hour in blocked else " "
            # 추천 차단: 3건 이상 & (PF < 1.0 또는 WR < 40%)
            rec_marker = "▲" if (cnt >= 3 and (pf < 1.0 or wr < 40.0)) else " "

            pf_str = f"{pf:.2f}" if not (pf != pf) else " NaN"
            print(
                f"  {hour:02d}:00  {cnt:>5}  {wr:>6.1f}%  {pf_str:>6}  {pnl:>+11.1f}"
                f"  {cur_marker:>4}  {rec_marker:>4}"
            )

        # 차단 중인데 실제로 좋은 시간대
        false_blocks = [
            h for h in blocked
            if int(agg.loc[h, "count"]) >= 3
            and not (agg.loc[h, "pf"] < 1.0 or agg.loc[h, "win_rate"] < 40.0)
        ]
        # 차단 안 됐는데 나쁜 시간대
        missed_blocks = [
            h for h in range(24)
            if h not in blocked
            and int(agg.loc[h, "count"]) >= 3
            and (agg.loc[h, "pf"] < 1.0 or agg.loc[h, "win_rate"] < 40.0)
        ]
        if false_blocks:
            print(f"\n  → 과도 차단 (성과 良): {sorted(false_blocks)}")
        if missed_blocks:
            print(f"  → 누락 차단 (성과 惡): {sorted(missed_blocks)}")
        print()

    print("● 현재 차단중  ▲ 추천 차단 (거래 3건+, PF<1.0 또는 WR<40%)")


if __name__ == "__main__":
    main()
