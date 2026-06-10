"""B축 — 피라미딩 v17 재검증 (백테만, 프로덕션 무수정).

v14 채택값 고정(스윕 금지): trigger 1.0R / add 25% / 1회 / ema_cross 한정.
게이트 G-B: 전구간 Sharpe ≥ 1.94 AND MDD봉 ≥ -45% AND 참사연도 없음 AND 1h/5m 효과 일치.

런 구성 (ProcessPool):
  병합 v17 1h: base/pyr × {full, IS 22~24, OOS 25~} = 6
  5m 서브바 교차(추세-only, 슬리브 5m 데이터 없음): base/pyr × {1h, subbar} full = 4
  → 교차 판정 = 피라미딩 효과(Δ최종$, ΔMDD)가 1h와 subbar에서 부호·크기 일치

사용: python scripts/pyramid_v17_check.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
PYR = {"enabled": True, "trigger_r": 1.0, "add_fraction": 0.25, "max_adds": 1,
       "strategies": ["ema_cross"]}
END = "2026-04-14"


def run_one(args):
    tag, pyramid_on, start, until, trend_only, subbar = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    if pyramid_on:
        p["pyramid"] = dict(PYR)
    if trend_only:
        p["strategies"]["mean_reversion"]["enabled"] = False
        p["symbols"] = [s for s in p["symbols"]
                        if s not in p["strategies"]["mean_reversion"]["symbols"]]
    if subbar and "5m" not in p["timeframes"]:
        p["timeframes"] = list(p["timeframes"]) + ["5m"]

    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100, subbar_tpsl=True) if subbar else build_engine(p, 100)
    report = eng.run(loader.iterate(since=pd.Timestamp(start, tz="UTC"),
                                    until=pd.Timestamp(until, tz="UTC")))
    eq = eng.equity_curve.to_series()
    daily = eq.resample("1D").last().pct_change().fillna(0)
    yearly = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    return tag, round(report.final_equity, 1), round(report.sharpe, 3), \
        round(report.max_drawdown, 1), len(eng.ledger.to_dataframe()), dict(yearly)


def main():
    jobs = []
    for name, on in (("base", False), ("pyr", True)):
        jobs += [(f"병합·{name}·full", on, "2022-01-01", END, False, False),
                 (f"병합·{name}·IS", on, "2022-01-01", "2024-12-31", False, False),
                 (f"병합·{name}·OOS", on, "2025-01-01", END, False, False)]
    for name, on in (("base", False), ("pyr", True)):
        jobs += [(f"추세만·{name}·1h", on, "2022-01-01", END, True, False),
                 (f"추세만·{name}·5m서브", on, "2022-01-01", END, True, True)]

    with ProcessPoolExecutor(max_workers=8) as ex:
        results = {r[0]: r[1:] for r in ex.map(run_one, jobs)}

    print("=== 피라미딩 v17 (trigger 1.0R / +25% / 1회 / ema_cross) ===")
    for tag, (f, sh, mdd, n, yearly) in results.items():
        ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yearly.items())
        print(f"  {tag:16} ${f:>10,.1f} Sh{sh:6.3f} MDD{mdd:6.1f}% {n}건  {ys}")

    bf, pf = results["병합·base·full"], results["병합·pyr·full"]
    print("\n[G-B 게이트]")
    print(f"  전구간: Sh {bf[1]} → {pf[1]} ({'OK' if pf[1] >= bf[1] else 'FAIL'}),"
          f" MDD {bf[2]} → {pf[2]}% ({'OK' if pf[2] >= -45.0 else 'FAIL ( -45 초과)'})"
          f", 수익 ${bf[0]:,.0f} → ${pf[0]:,.0f}")
    bo, po = results["병합·base·OOS"], results["병합·pyr·OOS"]
    print(f"  OOS:   Sh {bo[1]} → {po[1]}, MDD {bo[2]} → {po[2]}%")
    yrs_b, yrs_p = bf[4], pf[4]
    worst = min(yrs_p.values())
    print(f"  참사연도: pyr 최악 {worst:+.0f}% ({'OK' if worst > -50 else 'FAIL'})")

    d1 = results["추세만·pyr·1h"][0] / results["추세만·base·1h"][0] - 1
    d5 = results["추세만·pyr·5m서브"][0] / results["추세만·base·5m서브"][0] - 1
    print(f"  5m교차: 피라미딩 효과(최종$ 비율) 1h {d1*100:+.1f}% vs 5m서브 {d5*100:+.1f}%"
          f" → {'일치' if abs(d1 - d5) < 0.10 else '괴리(불신)'}")


if __name__ == "__main__":
    main()
