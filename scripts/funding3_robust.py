"""funding3 {0,8,16} 차단의 인과성 풀샘플 검증.

v16 단독(final_v16_slwide.yaml)에서 block_hours만 교체:
 - 고정: funding3 / 무차단 / 정산직후{1,9,17} / +2h / -1h / +4h
 - 랜덤 대조군: 랜덤 3시간 차단 N개 (funding3가 이 분포를 이겨야 인과)
다른 레이어(size_bonus 등)는 전부 동일 — block_hours만 변인.

판정:
 - funding3 Sharpe가 랜덤 분포 상위 & {1,9,17}이 무차단보다 나쁘면 → 인과 지지
 - funding3가 랜덤 중앙값 근처면 → 특별하지 않음(노이즈)
"""
from __future__ import annotations

import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from metrics.drawdown import max_drawdown
import scripts.run_backtest as rb

CONFIG = "config/final_v16_slwide.yaml"


def _yearly_ret(eq: pd.Series) -> dict:
    out = {}
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) >= 2:
            out[int(y)] = round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 0)
    return out


def run_one(spec: dict) -> dict:
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    hrs = spec["hours"]
    if hrs is None:
        p.pop("strategy_block_hours", None)
    else:
        p["strategy_block_hours"] = {"ema_cross": list(hrs), "multi_tf_breakout": list(hrs)}

    bt = p.get("backtest", {})
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(since=since, until=until))
    eq = engine.equity_curve.to_series()
    return {
        "name": spec["name"], "kind": spec["kind"],
        "final": report.final_equity, "ret": report.total_return_pct,
        "mdd": report.max_drawdown, "sharpe": report.sharpe,
        "trades": report.total_trades, "yearly": _yearly_ret(eq),
    }


def main() -> None:
    fixed = [
        {"name": "funding3 {0,8,16}", "hours": (0, 8, 16),  "kind": "fixed"},
        {"name": "무차단",            "hours": None,         "kind": "fixed"},
        {"name": "정산직후 {1,9,17}", "hours": (1, 9, 17),   "kind": "fixed"},
        {"name": "+2h {2,10,18}",     "hours": (2, 10, 18),  "kind": "fixed"},
        {"name": "-1h {23,7,15}",     "hours": (23, 7, 15),  "kind": "fixed"},
        {"name": "+4h {4,12,20}",     "hours": (4, 12, 20),  "kind": "fixed"},
    ]
    # 랜덤 3시간 대조군 (funding 시각 {0,8,16} 자체는 제외)
    rng = random.Random(20260609)
    rand = []
    seen = {(0, 8, 16)}
    while len(rand) < 12:
        h = tuple(sorted(rng.sample(range(24), 3)))
        if h in seen:
            continue
        seen.add(h)
        rand.append({"name": f"rand {h}", "hours": h, "kind": "random"})
    specs = fixed + rand

    with ProcessPoolExecutor(max_workers=7) as ex:
        results = list(ex.map(run_one, specs))
    by_name = {r["name"]: r for r in results}

    print(f"\nfunding3 인과성 풀샘플 검증 — v16 단독, 2022-01 ~ 2026-04 ($100→)")
    print("=" * 72)
    print(f"{'block_hours':20} {'최종$':>10} {'MDD%':>8} {'Sharpe':>7} {'거래':>5}")
    print("-" * 72)
    for s in fixed:
        r = by_name[s["name"]]
        print(f"{r['name']:20} {r['final']:>10,.0f} {r['mdd']:>8.1f} {r['sharpe']:>7.2f} {r['trades']:>5}")
    print("-" * 72)
    rr = [by_name[s["name"]] for s in rand]
    sh = sorted(x["sharpe"] for x in rr)
    fn = [x["sharpe"] for x in rr if x["final"] > 0]
    fins = sorted(x["final"] for x in rr)
    f3 = by_name["funding3 {0,8,16}"]
    nb = by_name["무차단"]
    n_worse = sum(1 for x in rr if x["sharpe"] < f3["sharpe"])
    print(f"랜덤 3시간 대조군 12개 — Sharpe min/중앙/max: {sh[0]:.2f} / {sh[len(sh)//2]:.2f} / {sh[-1]:.2f}")
    print(f"                       최종$  min/중앙/max: {fins[0]:,.0f} / {fins[len(fins)//2]:,.0f} / {fins[-1]:,.0f}")
    print(f"  → funding3 Sharpe {f3['sharpe']:.2f} 는 랜덤 12개 중 {n_worse}개보다 우수 (상위 {12-n_worse+1}/13 위)")
    print("=" * 72)

    print("\n[연도별 수익% (경로슬라이스) — funding3 가 매년 무차단을 이기는가]")
    years = sorted({y for r in results for y in r["yearly"]})
    print("variant".ljust(20) + "".join(f"{y:>8}" for y in years))
    for nm in ["funding3 {0,8,16}", "무차단", "정산직후 {1,9,17}"]:
        r = by_name[nm]
        row = nm.ljust(20) + "".join(f"{r['yearly'].get(y, 0):>8.0f}" for y in years)
        print(row)
    print("-" * 72)
    diff_row = "f3 - 무차단".ljust(20)
    win = 0
    for y in years:
        d = f3["yearly"].get(y, 0) - nb["yearly"].get(y, 0)
        win += 1 if d > 0 else 0
        diff_row += f"{d:>+8.0f}"
    print(diff_row + f"   ({win}/{len(years)}년 우위)")
    # 랜덤 대조군 연도별 평균 (참고)
    print("\n[랜덤 대조군 연도별 수익% 분포 (min~max)]")
    for y in years:
        vals = sorted(x["yearly"].get(y, 0) for x in rr)
        f3y = f3["yearly"].get(y, 0)
        print(f"  {y}: 랜덤 {vals[0]:>6.0f} ~ {vals[-1]:>6.0f} (중앙 {vals[len(vals)//2]:>6.0f})  |  funding3 {f3y:>6.0f}")


if __name__ == "__main__":
    main()
