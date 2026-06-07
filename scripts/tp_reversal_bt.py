"""tp_reversal_bt.py — TP 히트 후 즉시 reverse 포지션 백테스트.

비교:
  A. 기준선  — 즉시 진입, TP/SL 후 종료 (현재 설정)
  B. TP Reversal — TP 히트 시 반대 방향 동일 사이즈 즉시 진입
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def run_and_report(label: str, engine, loader, since, until) -> dict:
    report = engine.run(loader.iterate(since=since, until=until))

    records = engine.ledger.records
    if records:
        hold_h = [(r.exit_time - r.entry_time).total_seconds() / 3600 for r in records]
        avg_h  = sum(hold_h) / len(hold_h)
    else:
        avg_h = 0.0

    eq = engine.equity_curve.to_series()
    eq.index = pd.to_datetime(eq.index, utc=True)
    daily = eq.resample("D").last().ffill()
    q_ret = daily.resample("QE").last().pct_change().dropna() * 100
    neg_q = int((q_ret < 0).sum())

    pf = getattr(report, "profit_factor", float("nan"))
    print(f"\n[{label}]")
    print(f"  Sharpe {report.sharpe:.2f}  MDD {report.max_drawdown*100:.1f}%  "
          f"PF {pf:.2f}  WR {report.win_rate:.1f}%  "
          f"거래 {report.total_trades}건  "
          f"평균홀딩 {avg_h:.1f}h  손실분기 {neg_q}개  "
          f"최종 ${report.final_equity:,.0f}")

    return {
        "label":        label,
        "sharpe":       round(report.sharpe, 3),
        "mdd_%":        round(report.max_drawdown * 100, 1),
        "pf":           round(pf, 2),
        "wr_%":         round(report.win_rate, 1),
        "trades":       report.total_trades,
        "avg_hold_h":   round(avg_h, 1),
        "neg_quarters": neg_q,
        "final_equity": round(report.final_equity, 0),
    }


def main() -> None:
    with open("config/final_v13_eth.yaml") as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp("2022-01-01", tz="UTC")
    until = pd.Timestamp("2026-06-06", tz="UTC")

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )

    results = []

    # A. 기준선
    results.append(run_and_report(
        "A. 기준선 (TP→종료)",
        build_engine(p, initial_capital),
        loader, since, until,
    ))

    # B. TP Reversal
    results.append(run_and_report(
        "B. TP→즉시 Reverse",
        build_engine(p, initial_capital, tp_reversal=True),
        loader, since, until,
    ))

    print("\n" + "=" * 80)
    print("비교 요약")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df[["label","sharpe","mdd_%","pf","wr_%","trades","avg_hold_h",
              "neg_quarters","final_equity"]].to_string(index=False))


if __name__ == "__main__":
    main()
