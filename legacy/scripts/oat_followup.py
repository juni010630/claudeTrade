"""OAT 후속 검증 — OAT/fast_sweep가 가리킨 2개 리드를 풀엔진·연도별로 교차검증.

리드1) 짧은 TP: fast_sweep 정점 TP2.5/SL1.8(uniform). 풀엔진·전략별로 ema/multi TP를 줄이면
        매년 이기는가? (uniform·진입고정 한계 제거)
리드2) 강화 클러스터: SS게이트6·same_dir4·vol2.2를 *쌓으면* 계속 좋은가 과필터인가? (OAT는 단독뿐)
통합) 짧은TP + 강화 + BTC숏허용을 합친 후보가 연도별로 견고한가.

run_one은 oat_ablation 재사용(같은 baseline=merged_noblock_sleeve, 같은 ops 포맷, 연도별 포함).
※ 여전히 풀샘플 — 진짜 OOS는 별도 walk-forward 필요. 여기선 '연도별 일관성'까지만.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.oat_ablation import run_one, DEL  # baseline·엔진·연도별 동일


def build_specs() -> list:
    S = []
    def add(name, cat, d, ops): S.append({"name": name, "cat": cat, "dir": d, "ops": ops})

    add("BASELINE", "base", "—", [])

    # ── 리드1: 짧은 TP (풀엔진·전략별) ──
    add("ema TP 3.5→2.5",        "tp", "shorter", [[["strategies","ema_cross","atr_tp_mult"], 2.5]])
    add("multi TP 4.0→3.0",      "tp", "shorter", [[["strategies","multi_tf_breakout","atr_tp_mult"], 3.0]])
    add("multi TP 4.0→2.5",      "tp", "shorter", [[["strategies","multi_tf_breakout","atr_tp_mult"], 2.5]])
    add("ema2.5 + multi3.0",     "tp", "shorter", [[["strategies","ema_cross","atr_tp_mult"], 2.5],
                                                   [["strategies","multi_tf_breakout","atr_tp_mult"], 3.0]])
    add("ema2.5 + multi2.5",     "tp", "shorter", [[["strategies","ema_cross","atr_tp_mult"], 2.5],
                                                   [["strategies","multi_tf_breakout","atr_tp_mult"], 2.5]])

    # ── 리드2: 강화 클러스터 점진 스택 (과필터 검사) ──
    add("SS6",                   "stack", "tighten", [[["scorer","tier_ss_min_score"], 6]])
    add("SS6 + same_dir4",       "stack", "tighten", [[["scorer","tier_ss_min_score"], 6],
                                                      [["risk","max_same_direction"], 4]])
    add("SS6+sd4+vol2.2",        "stack", "tighten", [[["scorer","tier_ss_min_score"], 6],
                                                      [["risk","max_same_direction"], 4],
                                                      [["scorer","volume_ratio_threshold"], 2.2]])

    # ── 통합 후보 ──
    add("짧은TP+SS6+sd4",        "combo", "integ", [[["strategies","ema_cross","atr_tp_mult"], 2.5],
                                                    [["strategies","multi_tf_breakout","atr_tp_mult"], 3.0],
                                                    [["scorer","tier_ss_min_score"], 6],
                                                    [["risk","max_same_direction"], 4]])
    add("통합+BTC숏허용",         "combo", "integ", [[["strategies","ema_cross","atr_tp_mult"], 2.5],
                                                    [["strategies","multi_tf_breakout","atr_tp_mult"], 3.0],
                                                    [["scorer","tier_ss_min_score"], 6],
                                                    [["risk","max_same_direction"], 4],
                                                    [["symbol_block_directions","BTCUSDT"], DEL]])
    return S


def main() -> None:
    specs = build_specs()
    print(f"OAT 후속 검증 — {len(specs)}개 변형 (풀엔진·연도별)")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(run_one, s): s for s in specs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{done}/{len(specs)}] {r['name']:20} ${r['final']:>9,.0f}  "
                      f"MDD{r['mdd']:>6.1f}  Sh{r['sharpe']:>5.2f}  WR{r['wr']:>5.1f}  거래{r['trades']:>4}  "
                      f"({r['secs']}s)", flush=True)
            except Exception as e:
                s = futs[fut]
                print(f"  [{done}/{len(specs)}] {s['name']:20} 실패: {e}", flush=True)
    print(f"\n총 소요: {time.time()-t0:.0f}s")

    base = next(r for r in results if r["name"] == "BASELINE")
    years = sorted({y for r in results for y in r["yearly"]})
    spec_order = {s["name"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: spec_order.get(r["name"], 99))

    Path("OAT_FOLLOWUP_RESULTS.json").write_text(
        json.dumps({"baseline": base, "years": years, "variants": results}, ensure_ascii=False, indent=2))

    L = [f"# OAT 후속 검증 — 짧은 TP & 강화 클러스터 (풀엔진·연도별)", "",
         f"baseline=merged_noblock_sleeve, 전구간 $100, 봉MDD. 풀샘플(진짜 OOS 아님 — 연도별 일관성까지).", "",
         f"**BASELINE: ${base['final']:,.0f} / Sh {base['sharpe']:.2f} / MDD {base['mdd']:.1f}% / "
         f"WR {base['wr']}% / 거래 {base['trades']}**", "",
         "| 변형 | 최종$ | Δ$% | MDD% | ΔMDD | Sharpe | WR% | 거래 | " + " | ".join(years) + " |",
         "|" + "---|"*(7+len(years))]
    for r in results:
        d_final = (r["final"]/base["final"]-1)*100 if base["final"] else 0
        d_mdd = r["mdd"]-base["mdd"]
        d_sh = r["sharpe"]-base["sharpe"]
        yc = " | ".join(f"{r['yearly'].get(y,'—')}" for y in years)
        L.append(f"| {r['name']} | {r['final']:,.0f} | {d_final:+.0f}% | {r['mdd']:.1f} | {d_mdd:+.1f} | "
                 f"{r['sharpe']:.2f} | {r['wr']} | {r['trades']} | {yc} |")
    L += ["", "_연도=경로슬라이스 수익%. 모든 연도 ≥ baseline이어야 '견고'. 단일연도 구동이면 과적합 의심._"]
    Path("OAT_FOLLOWUP.md").write_text("\n".join(L))
    print("저장: OAT_FOLLOWUP_RESULTS.json, OAT_FOLLOWUP.md")


if __name__ == "__main__":
    main()
