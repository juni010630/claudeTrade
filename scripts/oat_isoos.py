"""강화 클러스터 3나사(SS게이트6 / same_dir4 / vol_thr2.2)의 조합을 IS/OOS 홀드아웃 검증.

조합 = 3C2(3) + 3C3(1) = 4개 + baseline 참조.
각 config를 IS·OOS 각각 독립 백테($100 fresh)로 돌림 — DataLoader가 since 이전 lookback(300봉)을
항상 제공하므로 OOS 2025-01 시작도 1d EMA200까지 완전 워밍업(cold-start 없음).
판정: IS·OOS '둘 다' baseline보다 나아야 견고. IS만 이기면 과적합.

baseline=merged_noblock_sleeve(v16+무차단+슬리브50:50). 프로덕션 무수정(build_engine 재사용).
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

COMBOS = [
    ("baseline",          []),
    ("SS6+sd4",           [[["scorer","tier_ss_min_score"],6], [["risk","max_same_direction"],4]]),
    ("SS6+vol2.2",        [[["scorer","tier_ss_min_score"],6], [["scorer","volume_ratio_threshold"],2.2]]),
    ("sd4+vol2.2",        [[["risk","max_same_direction"],4],  [["scorer","volume_ratio_threshold"],2.2]]),
    ("SS6+sd4+vol2.2",    [[["scorer","tier_ss_min_score"],6], [["risk","max_same_direction"],4],
                           [["scorer","volume_ratio_threshold"],2.2]]),
]


def run_period(args: tuple) -> dict:
    name, ops, period, since_str, until_str = args
    t0 = time.time()
    p = yaml.safe_load(open(CONFIG))
    apply_ops(p, ops)
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
        "name": name, "period": period,
        "ret": round(report.total_return_pct, 1),   # 수익률 %
        "final": round(report.final_equity, 0),
        "mdd": round(report.max_drawdown, 1),
        "sharpe": round(report.sharpe, 3),
        "wr": round(report.win_rate, 1),
        "trades": report.total_trades,
        "secs": round(time.time() - t0, 1),
    }


def main() -> None:
    tasks = []
    for name, ops in COMBOS:
        tasks.append((name, ops, "IS",  IS[0],  IS[1]))
        tasks.append((name, ops, "OOS", OOS[0], OOS[1]))
    print(f"IS/OOS 검증 — {len(COMBOS)}config × 2기간 = {len(tasks)}백테")
    print(f"  IS  = {IS[0]} ~ {IS[1]}")
    print(f"  OOS = {OOS[0]} ~ {OOS[1]}")
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
                print(f"  [{done}/{len(tasks)}] {r['name']:16} {r['period']:3}  "
                      f"수익{r['ret']:>+8.1f}%  MDD{r['mdd']:>6.1f}  Sh{r['sharpe']:>5.2f}  "
                      f"거래{r['trades']:>4}  ({r['secs']}s)", flush=True)
            except Exception as e:
                t = futs[fut]
                print(f"  [{done}/{len(tasks)}] {t[0]} {t[2]} 실패: {e}", flush=True)
    print(f"\n총 소요: {time.time()-t0:.0f}s")

    by = {(r["name"], r["period"]): r for r in res}
    base_is, base_oos = by[("baseline","IS")], by[("baseline","OOS")]

    Path("OAT_ISOOS_RESULTS.json").write_text(
        json.dumps({"IS": IS, "OOS": OOS, "results": res}, ensure_ascii=False, indent=2))

    L = ["# 강화 클러스터 조합 — IS/OOS 홀드아웃 검증", "",
         f"baseline=merged_noblock_sleeve. 각 셀 = 독립 $100 백테(지표 워밍업 완전).",
         f"IS = {IS[0]}~{IS[1]} · OOS = {OOS[0]}~{OOS[1]}. **수익률 = 해당 기간 총수익%**.",
         f"판정: IS·OOS **둘 다** baseline 초과해야 견고. Δ = baseline 대비 수익%p.", "",
         "| 조합 | IS 수익% | IS Δ | IS MDD | IS Sh | OOS 수익% | OOS Δ | OOS MDD | OOS Sh | IS거래 | OOS거래 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _ in COMBOS:
        i, o = by[(name,"IS")], by[(name,"OOS")]
        di = i["ret"] - base_is["ret"]
        do = o["ret"] - base_oos["ret"]
        L.append(f"| {name} | {i['ret']:+.1f} | {di:+.1f} | {i['mdd']:.1f} | {i['sharpe']:.2f} | "
                 f"{o['ret']:+.1f} | {do:+.1f} | {o['mdd']:.1f} | {o['sharpe']:.2f} | {i['trades']} | {o['trades']} |")
    L += ["", f"_baseline IS {base_is['ret']:+.1f}% (Sh {base_is['sharpe']:.2f}, MDD {base_is['mdd']:.1f}) · "
          f"OOS {base_oos['ret']:+.1f}% (Sh {base_oos['sharpe']:.2f}, MDD {base_oos['mdd']:.1f})_"]
    Path("OAT_ISOOS.md").write_text("\n".join(L))
    print("저장: OAT_ISOOS_RESULTS.json, OAT_ISOOS.md")


if __name__ == "__main__":
    main()
