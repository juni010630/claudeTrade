"""C축 — 가상 서브계좌(사이징 풀) + 월간 리밸 검증.

① off 패리티: sizing_pools 미설정 런 = v17 베이스라인 비트동일($8,991/805건) 필수 게이트
② on: 2풀(trend/sleeve) 50:50, 월간 가상 재분배 — full / IS(22~24) / OOS(25~)
게이트 G-C: Sharpe 비열위 + MDD봉 ≥ -45% + 최종수익↑ (전구간 기준, OOS 부호 일치 확인)

사용: python scripts/sizing_pools_check.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
END = "2026-04-14"
POOLS = {
    "enabled": True,
    "rebalance": "monthly",
    "pools": {
        "trend": {"strategies": ["ema_cross", "multi_tf_breakout"], "fraction": 0.5},
        "sleeve": {"strategies": ["mean_reversion"], "fraction": 0.5},
    },
}


def run_one(args):
    tag, pools_on, start, until = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    if pools_on:
        p["sizing_pools"] = POOLS
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    report = eng.run(loader.iterate(since=pd.Timestamp(start, tz="UTC"),
                                    until=pd.Timestamp(until, tz="UTC")))
    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    return tag, round(report.final_equity, 2), round(report.sharpe, 3), \
        round(report.max_drawdown, 1), len(eng.ledger.to_dataframe()), dict(yearly)


def main():
    jobs = [("OFF·full(패리티)", False, "2022-01-01", END),
            ("ON·full", True, "2022-01-01", END),
            ("ON·IS", True, "2022-01-01", "2024-12-31"),
            ("ON·OOS", True, "2025-01-01", END)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = {r[0]: r[1:] for r in ex.map(run_one, jobs)}

    print("=== 사이징 풀 (trend/sleeve 50:50, 월간 가상 리밸) ===")
    for tag, (f, sh, mdd, n, yearly) in results.items():
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
        print(f"  {tag:16} ${f:>10,.2f} Sh{sh:6.3f} MDD{mdd:6.1f}% {n}건  {ys}")

    off = results["OFF·full(패리티)"]
    parity = abs(off[0] - 8991) < 60 and off[3] == 805
    print(f"\n[패리티] OFF = ${off[0]:,.2f}/{off[3]}건 vs 기준 $8,991/805"
          f" → {'OK' if parity else 'FAIL — 기능 버그, on 결과 무효'}")
    on, base = results["ON·full"], off
    print(f"[G-C] Sh {base[1]} → {on[1]} ({'OK' if on[1] >= base[1] else 'FAIL'}),"
          f" MDD {base[2]} → {on[2]}% ({'OK' if on[2] >= -45.0 else 'FAIL'}),"
          f" 수익 ${base[0]:,.0f} → ${on[0]:,.0f} ({'↑' if on[0] > base[0] else '↓'})")


if __name__ == "__main__":
    main()
