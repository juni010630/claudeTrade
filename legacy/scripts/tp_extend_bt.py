"""tp_extend_bt.py — 동일 방향 재시그널 시 TP 연장 백테스트.

비교:
  A. 기준선  — 재시그널 무시 (현재 설정)
  B. TP 연장 — 같은 방향 재시그널 나오면 TP를 새 시그널 TP로 연장
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
    hold_h = [(r.exit_time - r.entry_time).total_seconds() / 3600 for r in records] if records else []
    avg_h  = sum(hold_h) / len(hold_h) if hold_h else 0.0

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


def _worker(args: tuple) -> dict:
    label, tp_extend, params_path, capital, since_str, until_str = args
    import yaml
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    with open(params_path) as f:
        p = yaml.safe_load(f)
    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC")
    engine = build_engine(p, capital, tp_extend_on_signal=tp_extend)
    return run_and_report(label, engine, loader, since, until)


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    params_path = "config/final_v13_eth.yaml"
    with open(params_path) as f:
        p = yaml.safe_load(f)
    initial_capital = p.get("backtest", {}).get("initial_capital", 100)
    since_str = "2022-01-01"
    until_str = "2026-06-06"

    jobs = [
        ("A. 기준선 (재시그널 무시)", False, params_path, initial_capital, since_str, until_str),
        ("B. 재시그널 → TP 연장",    True,  params_path, initial_capital, since_str, until_str),
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=2) as exe:
        futs = {exe.submit(_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            res = fut.result()
            results[res["label"]] = res
            print(f"완료: {res['label']}")

    ordered = [results[j[0]] for j in jobs]
    print("\n" + "=" * 80)
    print("비교 요약")
    print("=" * 80)
    df = pd.DataFrame(ordered)
    print(df[["label","sharpe","mdd_%","pf","wr_%","trades","avg_hold_h",
              "neg_quarters","final_equity"]].to_string(index=False))


if __name__ == "__main__":
    main()
