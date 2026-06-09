"""ema_cross TP 단독 walk-forward 검증 (oat_followup의 full-sample +44% 재검증).

사전 등록 판정 규칙 (실행 전 고정):
  1. IS(2022-01-01~2024-12-31)에서 Sharpe 최고 TP를 선택 (test 미참조).
  2. 채택 조건: 선택된 TP가 OOS(2025-01-01~2026-04-14)에서 baseline(3.5) 대비
     수익%·Sharpe 둘 다 ≥, MDD 크게 악화 없음(+5pp 이내).
  3. 연도별(2022/23/24/25 독립 $100) 무붕괴: 어느 해도 baseline 대비 수익 큰 폭 악화 없음.
  하나라도 미달 → 기각, 현 3.5 유지.

baseline = config/final_v17.yaml (현 라이브). 프로덕션 무수정(build_engine 재사용).
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
import scripts.run_backtest as rb
from scripts.oat_ablation import apply_ops, CONFIG

IS = ("2022-01-01", "2024-12-31")
OOS = ("2025-01-01", "2026-04-14")
YEARS = [2022, 2023, 2024, 2025]
TP_VALUES = [2.5, 3.0, 3.5, 4.0]  # 3.5 = baseline


def run_period(args: tuple) -> dict:
    tp, period, since_str, until_str = args
    t0 = time.time()
    p = yaml.safe_load(open(CONFIG))
    apply_ops(p, [[["strategies", "ema_cross", "atr_tp_mult"], tp]])
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC")
    engine = rb.build_engine(p, 100.0)
    report = engine.run(loader.iterate(since=since, until=until))
    return {
        "tp": tp, "period": period,
        "ret": round(report.total_return_pct, 1),
        "final": round(report.final_equity, 0),
        "mdd": round(report.max_drawdown, 1),
        "sharpe": round(report.sharpe, 3),
        "wr": round(report.win_rate, 1),
        "trades": report.total_trades,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    tasks = []
    for tp in TP_VALUES:
        tasks.append((tp, "IS", IS[0], IS[1]))
        tasks.append((tp, "OOS", OOS[0], OOS[1]))
        for yr in YEARS:
            tasks.append((tp, str(yr), f"{yr}-01-01", f"{yr+1}-01-01"))
    print(f"ema TP walk-forward — {len(TP_VALUES)}TP × (IS+OOS+{len(YEARS)}년) = {len(tasks)}백테")
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(run_period, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                res.append(r)
                print(f"  [{done}/{len(tasks)}] TP{r['tp']}  {r['period']:4}  "
                      f"수익{r['ret']:>+8.1f}%  MDD{r['mdd']:>6.1f}  Sh{r['sharpe']:>5.2f}  "
                      f"거래{r['trades']:>4}  ({r['secs']}s)", flush=True)
            except Exception as e:
                t = futs[fut]
                print(f"  [{done}/{len(tasks)}] TP{t[0]} {t[1]} 실패: {e}", flush=True)
    print(f"\n총 소요: {time.time()-t0:.0f}s")

    by = {(r["tp"], r["period"]): r for r in res}
    Path("EMA_TP_WF_RESULTS.json").write_text(
        json.dumps({"IS": IS, "OOS": OOS, "results": res}, ensure_ascii=False, indent=2))

    # IS 최적 선택 (사전 규칙: Sharpe)
    is_rows = [(tp, by[(tp, "IS")]) for tp in TP_VALUES]
    pick = max(is_rows, key=lambda x: x[1]["sharpe"])[0]

    L = ["# ema TP walk-forward (IS 도출 → OOS 검증)", "",
         f"baseline = final_v17 (ema TP 3.5). IS = {IS[0]}~{IS[1]} · OOS = {OOS[0]}~{OOS[1]}.",
         f"규칙(사전등록): IS Sharpe 최고 선택 → OOS에서 수익·Sharpe ≥ baseline AND 연도별 무붕괴.", "",
         "| TP | IS 수익% | IS Sh | IS MDD | OOS 수익% | OOS Sh | OOS MDD | 2022 | 2023 | 2024 | 2025 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tp in TP_VALUES:
        i, o = by[(tp, "IS")], by[(tp, "OOS")]
        yr_cells = " | ".join(f"{by[(tp, str(y))]['ret']:+.0f}%" for y in YEARS)
        mark = " **(IS pick)**" if tp == pick else (" (base)" if tp == 3.5 else "")
        L.append(f"| {tp}{mark} | {i['ret']:+.1f} | {i['sharpe']:.2f} | {i['mdd']:.1f} | "
                 f"{o['ret']:+.1f} | {o['sharpe']:.2f} | {o['mdd']:.1f} | {yr_cells} |")
    L += ["", f"_IS Sharpe 최적 = TP {pick}_"]
    Path("EMA_TP_WF.md").write_text("\n".join(L))
    print(f"\nIS pick = TP {pick}")
    print("저장: EMA_TP_WF_RESULTS.json, EMA_TP_WF.md")


if __name__ == "__main__":
    main()
