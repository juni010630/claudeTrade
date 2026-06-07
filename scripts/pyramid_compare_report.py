"""pyramid_compare_report.py — baseline vs pyr10_f25 상세 비교 리포트.

full 기간 2개 백테스트 병렬 실행 → 원장/equity curve 레벨 비교:
연도별 지표, 분기 수익률, 연속 손절, MDD 회복, 증액 거래 기여.
"""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def _run(pyr: bool):
    with open("config/final_v13_eth.yaml") as f:
        p = yaml.safe_load(f)
    if pyr:
        p["pyramid"] = {"enabled": True, "trigger_r": 1.0, "add_fraction": 0.25, "max_adds": 1}
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=p.get("data", {}).get("lookback_bars", 300))
    eng = build_engine(p, 100)
    report = eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                                    until=pd.Timestamp(p["backtest"]["end"], tz="UTC")))
    df = eng.ledger.to_dataframe()
    eq = eng.equity_curve.to_series()
    return report, df, eq


def streaks(df: pd.DataFrame) -> int:
    s, mx = 0, 0
    for pnl in df.sort_values("exit_time")["pnl"]:
        s = s + 1 if pnl <= 0 else 0
        mx = max(mx, s)
    return mx


def main() -> None:
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        f_base = pool.submit(_run, False)
        f_pyr = pool.submit(_run, True)
        (rb, db, eb), (rp, dp, ep) = f_base.result(), f_pyr.result()

    print("=" * 90)
    print("  baseline vs pyr10_f25 (트리거 +1.0R, 증액 25%) — full 2022-2026")
    print("=" * 90)
    rows = []
    for name, r, d, e in [("baseline", rb, db, eb), ("pyramid", rp, dp, ep)]:
        wins = (d.pnl > 0).sum()
        gw = d.loc[d.pnl > 0, "pnl"].sum(); gl = abs(d.loc[d.pnl <= 0, "pnl"].sum())
        rows.append({
            "config": name, "final_eq": round(r.final_equity), "sharpe": round(r.sharpe, 3),
            "calmar": round(r.calmar, 2), "MDD%": round(r.max_drawdown, 1),
            "WR%": round(wins / len(d) * 100, 1), "PF": round(gw / gl, 2),
            "trades": len(d), "max_연속손실": streaks(d),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # 연도별 equity 배수 (해당 연도 시작/끝 equity 비율)
    print("\n── 연도별 자산 배수 (그 해에만 운용했을 때가 아니라, 복리 경로 내 구간 배수) ──")
    yr_rows = []
    for name, e in [("baseline", eb), ("pyramid", ep)]:
        row = {"config": name}
        for y in [2022, 2023, 2024, 2025, 2026]:
            seg = e[(e.index >= f"{y}-01-01") & (e.index < f"{y+1}-01-01")]
            if len(seg) > 1:
                row[str(y)] = round(seg.iloc[-1] / seg.iloc[0], 2)
        yr_rows.append(row)
    print(pd.DataFrame(yr_rows).to_string(index=False))

    # 분기 수익률
    print("\n── 분기 수익률 (%) ──")
    q = pd.DataFrame({
        "baseline": eb.resample("QE").last().pct_change().mul(100).round(1),
        "pyramid": ep.resample("QE").last().pct_change().mul(100).round(1),
    })
    q.index = q.index.strftime("%Y-Q%q") if hasattr(q.index, "strftime") else q.index
    print(q.to_string())

    # MDD 구간/회복
    for name, e in [("baseline", eb), ("pyramid", ep)]:
        peak = e.cummax()
        dd = (e - peak) / peak
        trough_t = dd.idxmin()
        peak_t = e[:trough_t].idxmax()
        rec = e[trough_t:][e[trough_t:] >= peak[trough_t]]
        rec_t = rec.index[0] if len(rec) else None
        rec_days = (rec_t - trough_t).days if rec_t is not None else -1
        print(f"\n{name}: MDD {dd.min()*100:.1f}%  (피크 {peak_t.date()} → 바닥 {trough_t.date()}"
              f" → 회복 {rec_t.date() if rec_t is not None else '미회복'}, 회복 {rec_days}일)")

    # 증액 거래 기여 (피라미드 런 — entry_price 차이로 식별)
    db2 = db.copy(); dp2 = dp.copy()
    db2["key"] = db2.entry_time.astype(str) + db2.symbol + db2.strategy
    dp2["key"] = dp2.entry_time.astype(str) + dp2.symbol + dp2.strategy
    m = db2.merge(dp2, on="key", suffixes=("_b", "_p"))
    added = m[(m.entry_price_p - m.entry_price_b).abs() > 1e-9]
    print(f"\n── 증액 발생 거래: {len(added)}/{len(m)} (조인 기준) ──")
    aw = (added.pnl_p > 0).sum()
    print(f"증액 거래 WR: {aw/len(added)*100:.1f}%  |  PnL 합 변화(같은 거래): "
          f"${added.pnl_b.sum():,.0f} → ${added.pnl_p.sum():,.0f}")
    yr = added.copy()
    yr["year"] = pd.to_datetime(yr.entry_time_b).dt.year.clip(upper=2025)
    g = yr.groupby("year").apply(
        lambda x: pd.Series({"n": len(x), "win": (x.pnl_p > 0).sum()}), include_groups=False)
    print(g.to_string())


if __name__ == "__main__":
    main()
