"""ML 필터 효과 평가 — baseline vs 보너스 vs 하드컷 백테스트 비교.

3개 백테스트를 병렬로 실행한 뒤 OOS / 연도별 성과 비교표 출력.

Usage:
    python scripts/eval_ml_filter.py
    python scripts/eval_ml_filter.py --model models/ml_filter.pkl --oos-start 2025-01-01
"""
from __future__ import annotations

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader


# ── 단일 백테스트 실행 (subprocess 격리) ─────────────────────────
def _run_one(job: dict) -> dict:
    """별도 프로세스에서 실행 (fork-safe)."""
    from scripts.run_backtest import build_engine
    label = job["label"]
    p = job["params"]
    since = pd.Timestamp(job["since"], tz="UTC")
    until = pd.Timestamp(job["until"], tz="UTC")
    initial_capital = job["initial_capital"]
    data_cfg = p.get("data", {})

    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    engine = build_engine(p, initial_capital)
    snapshots = loader.iterate(since=since, until=until)
    report = engine.run(snapshots)

    ledger_df = engine.ledger.to_dataframe()
    return {
        "label": label,
        "report": report,
        "ledger": ledger_df,
    }


def _year_stats(ledger_df: pd.DataFrame, year: int) -> dict:
    if ledger_df.empty:
        return {}
    df = ledger_df.copy()
    df["entry_year"] = pd.to_datetime(df["entry_time"], utc=True).dt.year
    sub = df[df["entry_year"] == year]
    if sub.empty:
        return {}
    wins = (sub["pnl"] > 0).sum()
    n = len(sub)
    gross_profit = sub[sub["pnl"] > 0]["pnl"].sum()
    gross_loss = (-sub[sub["pnl"] <= 0]["pnl"]).sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "n": n,
        "wr": wins / n,
        "pf": pf,
        "net_pnl": sub["pnl"].sum(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml")
    parser.add_argument("--model", default="models/ml_filter.pkl")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--full-start", default="2022-01-01")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] 모델 없음: {args.model}  — train_ml_filter.py 먼저 실행")
        sys.exit(1)

    with open(args.params) as f:
        p_base = yaml.safe_load(f)

    initial_capital = p_base.get("backtest", {}).get("initial_capital", 100)
    full_end = p_base.get("backtest", {}).get("end", "2026-04-14")

    # ── 3가지 파라미터 세트 ────────────────────────────────────────
    p_baseline = copy.deepcopy(p_base)
    p_baseline.setdefault("scorer", {}).setdefault("ml_soft_scoring", {})["enabled"] = False

    p_bonus = copy.deepcopy(p_base)
    p_bonus["scorer"]["ml_soft_scoring"]["enabled"] = True
    p_bonus["scorer"]["ml_soft_scoring"]["mode"] = "bonus"
    p_bonus["scorer"]["ml_soft_scoring"]["model_path"] = args.model

    p_hardcut = copy.deepcopy(p_base)
    p_hardcut["scorer"]["ml_soft_scoring"]["enabled"] = True
    p_hardcut["scorer"]["ml_soft_scoring"]["mode"] = "hardcut"
    p_hardcut["scorer"]["ml_soft_scoring"]["model_path"] = args.model

    jobs_oos = [
        {"label": "baseline", "params": p_baseline, "since": args.oos_start,
         "until": full_end, "initial_capital": initial_capital},
        {"label": "bonus",    "params": p_bonus,    "since": args.oos_start,
         "until": full_end, "initial_capital": initial_capital},
        {"label": "hardcut",  "params": p_hardcut,  "since": args.oos_start,
         "until": full_end, "initial_capital": initial_capital},
    ]

    years = [2022, 2023, 2024, 2025]
    jobs_yearly = []
    for yr in years:
        for label, p in [("baseline", p_baseline), ("bonus", p_bonus), ("hardcut", p_hardcut)]:
            jobs_yearly.append({
                "label": f"{label}_{yr}", "params": p,
                "since": f"{yr}-01-01", "until": f"{yr+1}-01-01",
                "initial_capital": initial_capital,
            })

    all_jobs = jobs_oos + jobs_yearly
    print(f"백테스트 {len(all_jobs)}개 병렬 실행 중...")

    results: dict[str, dict] = {}
    with ProcessPoolExecutor() as ex:
        futs = {ex.submit(_run_one, j): j["label"] for j in all_jobs}
        for fut in as_completed(futs):
            lbl = futs[fut]
            try:
                res = fut.result()
                results[lbl] = res
                print(f"  완료: {lbl}")
            except Exception as e:
                print(f"  [ERROR] {lbl}: {e}")

    # ── OOS 비교표 ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print(f"OOS 비교 ({args.oos_start} ~ {full_end})")
    print("═" * 70)
    print(f"{'모드':12s}  {'거래수':>6s}  {'WR':>6s}  {'PF':>6s}  "
          f"{'Sharpe':>7s}  {'MDD':>7s}  {'최종자산':>10s}")
    print("─" * 70)

    for lbl in ["baseline", "bonus", "hardcut"]:
        if lbl not in results:
            print(f"{lbl:12s}  (결과 없음)")
            continue
        r = results[lbl]["report"]
        m = r.metrics if hasattr(r, "metrics") else {}
        _print_row(lbl, r)

    # ── 연도별 비교 ─────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("연도별 PF (baseline → bonus → hardcut)")
    print("─" * 70)
    print(f"{'연도':6s}  {'baseline':>10s}  {'bonus':>10s}  {'hardcut':>10s}")
    print("─" * 70)

    for yr in years:
        row = [str(yr)]
        for lbl in ["baseline", "bonus", "hardcut"]:
            key = f"{lbl}_{yr}"
            if key not in results or results[key]["ledger"].empty:
                row.append("   —   ")
            else:
                st = _year_stats(results[key]["ledger"], yr)
                if st:
                    row.append(f"{st['pf']:>6.2f} ({st['n']}건)")
                else:
                    row.append("   —   ")
        print(f"{row[0]:6s}  {row[1]:>10s}  {row[2]:>10s}  {row[3]:>10s}")

    # ── 수용 기준 체크 ─────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("수용 기준 체크 (OOS 기준)")
    _check_gates(results)


def _print_row(label: str, report) -> None:
    try:
        s = report.stats
        print(f"{label:12s}  {s.get('num_trades', 0):>6d}  "
              f"{s.get('win_rate', 0):>5.1%}  "
              f"{s.get('profit_factor', 0):>6.2f}  "
              f"{s.get('sharpe', 0):>7.2f}  "
              f"{s.get('max_drawdown', 0):>6.1%}  "
              f"${s.get('final_equity', 0):>9,.0f}")
    except Exception:
        # MetricsReport API가 다를 수 있음 — 폴백
        print(f"{label:12s}  (상세 통계 파싱 실패)")


def _check_gates(results: dict) -> None:
    def _sharpe(lbl: str):
        if lbl not in results:
            return None
        try:
            return results[lbl]["report"].stats.get("sharpe", 0)
        except Exception:
            return None

    base_sh = _sharpe("baseline")
    bonus_sh = _sharpe("bonus")
    hcut_sh = _sharpe("hardcut")

    print(f"  baseline Sharpe:  {base_sh}")
    print(f"  bonus    Sharpe:  {bonus_sh}  {'✓' if bonus_sh and base_sh and bonus_sh >= base_sh else '✗'}")
    print(f"  hardcut  Sharpe:  {hcut_sh}  {'✓' if hcut_sh and base_sh and hcut_sh >= base_sh else '✗'}")
    print("\n→ 어느 쪽이든 baseline 대비 개선이면 eval_ml_filter.py STEP 4 통과.")
    print("→ 개선 없으면 ML 필터 폐기 (적응형 TP/SL과 동일 기각 기준).")


if __name__ == "__main__":
    main()
