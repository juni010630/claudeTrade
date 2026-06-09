"""merged 자본배분 프론티어(추세:슬리브) + 딥 플로어(-55%) 검증.

대출시드용 실측 레버: 추세 비중을 낮추면 MDD가 얼마 줄고 Sharpe/수익 얼마 주는가.
capital_fraction만 변인 — ema/multi=추세 비중, mean_reversion=슬리브 비중.
딥 플로어: peak 대비 -55% flatten+쿨다운(TailCutEngine 재사용) — 인샘플 미발동(no-op) 확인용.
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
from scripts.tailcut_backtest import TailCutEngine

CONFIG = "config/merged_v16_sleeve.yaml"
START, END = "2022-01-01", "2026-04-23"


def _yearly(eq: pd.Series) -> dict:
    out = {}
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) >= 2:
            out[int(y)] = (round((seg.iloc[-1]/seg.iloc[0]-1)*100, 0), round(max_drawdown(seg)*100, 0))
    return out


def run_one(spec: dict) -> dict:
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    tf, sf = spec["trend"], spec["sleeve"]
    p["strategy_capital_fraction"] = {"ema_cross": tf, "multi_tf_breakout": tf, "mean_reversion": sf}

    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    if spec["floor"] is not None:
        engine.__class__ = TailCutEngine
        engine._cut_threshold = spec["floor"]
        engine._cut_cooldown_h = 168
        engine._cut_targets = None
        engine._cut_peak = 100.0
        engine._cut_pause_until = None
        engine._cut_events = []
    report = engine.run(loader.iterate(
        since=pd.Timestamp(START, tz="UTC"), until=pd.Timestamp(END, tz="UTC")))
    eq = engine.equity_curve.to_series()
    eqd = eq.resample("1D").last().ffill()
    mdd_d = float(((eqd - eqd.cummax()) / eqd.cummax()).min()) * 100
    return {
        "name": spec["name"], "final": report.final_equity,
        "mdd_bar": report.max_drawdown, "mdd_daily": round(mdd_d, 1),
        "sharpe": report.sharpe, "trades": report.total_trades,
        "cuts": len(getattr(engine, "_cut_events", [])), "yearly": _yearly(eq),
    }


def main() -> None:
    specs = [
        {"name": "50:50 (현재)", "trend": 0.50, "sleeve": 0.50, "floor": None},
        {"name": "45:55",        "trend": 0.45, "sleeve": 0.55, "floor": None},
        {"name": "40:60",        "trend": 0.40, "sleeve": 0.60, "floor": None},
        {"name": "35:65",        "trend": 0.35, "sleeve": 0.65, "floor": None},
        {"name": "30:70",        "trend": 0.30, "sleeve": 0.70, "floor": None},
        {"name": "50:50 +floor55", "trend": 0.50, "sleeve": 0.50, "floor": 0.55},
        {"name": "40:60 +floor55", "trend": 0.40, "sleeve": 0.60, "floor": 0.55},
    ]
    with ProcessPoolExecutor(max_workers=7) as ex:
        results = list(ex.map(run_one, specs))
    order = {s["name"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order[r["name"]])

    print(f"\nmerged 자본배분 프론티어 (추세:슬리브) — {START}~{END} ($100→)")
    print("=" * 76)
    print(f"{'배분':16} {'최종$':>9} {'MDD봉%':>8} {'MDD일별%':>9} {'Sharpe':>7} {'거래':>5} {'컷':>3}")
    print("-" * 76)
    for r in results:
        print(f"{r['name']:16} {r['final']:>9,.0f} {r['mdd_bar']:>8.1f} {r['mdd_daily']:>9.1f} "
              f"{r['sharpe']:>7.2f} {r['trades']:>5} {r['cuts']:>3}")
    print("=" * 76)

    print("\n[연도별 수익% / 연내MDD봉%]")
    years = sorted({y for r in results for y in r["yearly"]})
    print("배분".ljust(16) + "".join(f"{y:>14}" for y in years))
    for r in results:
        if r["name"].endswith("floor55"):
            continue
        row = r["name"].ljust(16)
        for y in years:
            if y in r["yearly"]:
                ret, mdd = r["yearly"][y]
                row += f"{ret:>7.0f}/{mdd:>5.0f} "
            else:
                row += f"{'—':>14}"
        print(row)


if __name__ == "__main__":
    main()
