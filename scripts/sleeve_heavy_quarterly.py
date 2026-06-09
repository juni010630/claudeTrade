"""슬리브 비중 확장(5:5~9:1) 분기별 — 단일 merged 엔진(실배포 일치, 블렌드 아님).

슬리브:v16 = 5:5,6:4,7:3,8:2,9:1 (슬리브 50→90%).
merged_v16_sleeve.yaml의 strategy_capital_fraction만 변인:
  v16(추세)=ema_cross+multi_tf_breakout, 슬리브=mean_reversion.
단일 cross-margin 계좌 = 실배포 동일. 블렌드(두 독립백테 일별가중)와 달리
매일리밸 가정 없음([블렌드 사이징 실체] 참조). 격리 스크립트 — 프로덕션 무수정.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from metrics.drawdown import max_drawdown
import scripts.run_backtest as rb

CONFIG = "config/merged_v16_sleeve.yaml"
START, END = "2022-01-01", "2026-04-23"

# 슬리브:v16 → (추세 fraction, 슬리브 fraction)
SPECS = [
    {"name": "5:5", "trend": 0.5, "sleeve": 0.5},
    {"name": "6:4", "trend": 0.4, "sleeve": 0.6},
    {"name": "7:3", "trend": 0.3, "sleeve": 0.7},
    {"name": "8:2", "trend": 0.2, "sleeve": 0.8},
    {"name": "9:1", "trend": 0.1, "sleeve": 0.9},
]


def run_one(spec: dict) -> dict:
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    tf, sf = spec["trend"], spec["sleeve"]
    p["strategy_capital_fraction"] = {
        "ema_cross": tf, "multi_tf_breakout": tf, "mean_reversion": sf,
    }
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(
        since=pd.Timestamp(START, tz="UTC"), until=pd.Timestamp(END, tz="UTC")))

    eq = engine.equity_curve.to_series()
    eqd = eq.resample("1D").last().ffill()
    dret = eqd.pct_change()
    mdd_d = float(((eqd - eqd.cummax()) / eqd.cummax()).min()) * 100

    # 분기별 수익% + 분기내 MDD(일별기준)
    q_ret = dret.resample("QE").apply(lambda r: ((1 + r).prod() - 1) * 100)
    q_mdd = eqd.resample("QE").apply(lambda s: max_drawdown(s) * 100 if len(s) >= 2 else 0.0)
    qkey = lambda idx: [f"{t.year}Q{t.quarter}" for t in idx]
    q_ret.index = qkey(q_ret.index)
    q_mdd.index = qkey(q_mdd.index)

    return {
        "name": spec["name"], "final": report.final_equity,
        "mdd_bar": report.max_drawdown, "mdd_daily": round(mdd_d, 1),
        "sharpe": report.sharpe, "trades": report.total_trades,
        "q_ret": q_ret.round(1), "q_mdd": q_mdd.round(1),
    }


def main() -> None:
    with ProcessPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(run_one, SPECS))
    order = {s["name"]: i for i, s in enumerate(SPECS)}
    results.sort(key=lambda r: order[r["name"]])

    print(f"\n슬리브:v16 배분 (슬리브 비중 ↑) — 단일 merged 엔진 — {START}~{END} ($100→)")
    print("=" * 70)
    print(f"{'슬리브:v16':>9} {'최종$':>9} {'MDD봉%':>8} {'MDD일별%':>9} {'Sharpe':>7} {'거래':>5}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:>9} {r['final']:>9,.0f} {r['mdd_bar']:>8.1f} {r['mdd_daily']:>9.1f} "
              f"{r['sharpe']:>7.2f} {r['trades']:>5}")
    print("=" * 70)

    ret_df = pd.DataFrame({r["name"]: r["q_ret"] for r in results})
    mdd_df = pd.DataFrame({r["name"]: r["q_mdd"] for r in results})

    print("\n[분기별 수익% — 슬리브:v16 배분별]")
    print(ret_df.to_string())
    print("\n[분기별 MDD% (분기내 일별기준) — 슬리브:v16 배분별]")
    print(mdd_df.to_string())

    Path("data/results").mkdir(parents=True, exist_ok=True)
    ret_df.to_csv("data/results/sleeve_heavy_quarterly_ret.csv")
    mdd_df.to_csv("data/results/sleeve_heavy_quarterly_mdd.csv")
    print("\n저장: data/results/sleeve_heavy_quarterly_{ret,mdd}.csv")


if __name__ == "__main__":
    main()
