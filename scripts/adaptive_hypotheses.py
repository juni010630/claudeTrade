"""adaptive_hypotheses.py — 사전 지정 적응형 TP/SL 가설들을 전 기간 리플레이로 비교.

왜 그리드 서치 대신 가설 검정인가:
  단일 리플레이가 ~270초(사실상 풀 백테). 81규칙×확장폴드 그리드는 수 시간 → 비현실적.
  규칙을 **사전 지정**(in-sample 선택편향 없음)하면 전 기간 1회 리플레이로 정당하게 비교 가능.
  MDD 감소 목표에 맞춘 소수 가설만 검정 → 좋아 보이면 연도별 강건성만 추가 확인.

버킷(adx_split, bbw_split 기준): hiadx/loadx × lobw/hibw 4분면.
규칙 = 버킷→(tp_scale, sl_scale), 베이스라인 대비 배율. 항등=(1.0,1.0).

Usage:
  python3 -u scripts/adaptive_hypotheses.py --entries data/entries_ctx.parquet
  python3 -u scripts/adaptive_hypotheses.py --entries data/entries_ctx.parquet --by-year combo_A
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from scripts.adaptive_tpsl import BUCKETS, apply_adaptive_tpsl
from scripts.walkforward_tpsl import _replay_worker

HI = ("hiadx_lobw", "hiadx_hibw")   # 강추세
LO = ("loadx_lobw", "loadx_hibw")   # 약추세/전환
LOBW = ("hiadx_lobw", "loadx_lobw")  # 저변동성(수축)
HIBW = ("hiadx_hibw", "loadx_hibw")  # 고변동성(확장)


def _rule(**overrides) -> dict:
    """기본 항등에서 지정 버킷만 (tp_scale, sl_scale) 덮어쓰기."""
    r = {b: (1.0, 1.0) for b in BUCKETS}
    r.update(overrides)
    return r


def _set(buckets, tp=1.0, sl=1.0) -> dict:
    return {b: (tp, sl) for b in buckets}


# ── 가설 규칙들 (MDD 감소 가설 중심) ────────────────────────────────────────
def build_hypotheses() -> dict[str, dict]:
    H: dict[str, dict] = {}
    H["baseline"] = _rule()                                   # 현재 고정 TP/SL
    H["loadx_sl0.7"] = _rule(**_set(LO, sl=0.7))              # 약추세 SL 좁힘(빨리 손절)
    H["loadx_sl0.5"] = _rule(**_set(LO, sl=0.5))              # 약추세 SL 더 좁힘
    H["loadx_tp0.7_sl0.7"] = _rule(**_set(LO, tp=0.7, sl=0.7))  # 약추세 빠른 스캘프
    H["hiadx_tp1.3"] = _rule(**_set(HI, tp=1.3))             # 강추세 익절 멀리(추세 태우기)
    H["combo_A"] = _rule(hiadx_lobw=(1.3, 1.0), hiadx_hibw=(1.3, 1.0),
                         loadx_lobw=(1.0, 0.7), loadx_hibw=(1.0, 0.7))
    H["hibw_sl0.7"] = _rule(**_set(HIBW, sl=0.7))            # 고변동 SL 좁힘
    H["hibw_sl1.3"] = _rule(**_set(HIBW, sl=1.3))            # 고변동 SL 넓힘(노이즈 회피)
    H["sl0.7_all"] = _rule(**_set(BUCKETS, sl=0.7))          # 전역 SL 좁힘(대조용)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="config/final_v13_eth.yaml")
    ap.add_argument("--entries", default="data/entries_ctx.parquet")
    ap.add_argument("--adx-split", type=float, default=25.0)
    ap.add_argument("--bbw-split", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--by-year", default=None,
                    help="지정 규칙명(+baseline)을 연도별로 분해 평가")
    args = ap.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)
    cap = p["backtest"].get("initial_capital", 100)
    df = pd.read_parquet(args.entries)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    H = build_hypotheses()

    def make_args(name, rule, since, until):
        applied = apply_adaptive_tpsl(
            df[(df.timestamp >= pd.Timestamp(since, tz="UTC")) &
               (df.timestamp <= pd.Timestamp(until, tz="UTC"))],
            rule, adx_split=args.adx_split, bbw_split=args.bbw_split)
        recs = applied.assign(timestamp=applied["timestamp"].astype(str)).to_dict("records")
        return (name, args.params, recs, str(since), str(until), cap)

    if args.by_year:
        names = ["baseline", args.by_year]
        years = [("2022-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31"),
                 ("2024-01-01", "2024-12-31"), ("2025-01-01", "2026-04-14")]
        jobs = []
        for nm in names:
            for s, e in years:
                tag = f"{nm}@{s[:4]}"
                a = make_args(tag, H[nm], s, e)
                jobs.append((tag,) + a[1:])
        print(f"[연도별] {args.by_year} vs baseline | {len(jobs)}개 리플레이 | workers={args.workers}",
              flush=True)
    else:
        full = ("2022-01-01", "2026-04-14")
        jobs = [make_args(nm, rule, *full) for nm, rule in H.items()]
        print(f"[가설 비교] {len(jobs)}개 규칙 전 기간 리플레이 | workers={args.workers}", flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_replay_worker, j): j[0] for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['tag']:<22} Calmar {r['calmar']:>8.2f}  "
                      f"Sharpe {r['sharpe']:>6.2f}  MDD {r['max_dd_%']:>6.1f}%  "
                      f"ret {r['total_return']*100:>+10.1f}%  WR {r['win_rate_%']:>4.1f}%  "
                      f"trades {r['trades']}", flush=True)
            except Exception as e:
                print(f"  ✗ {futs[fut]} 실패: {e}", flush=True)

    res = pd.DataFrame(results)
    sort_col = "tag" if args.by_year else "calmar"
    res = res.sort_values(sort_col, ascending=bool(args.by_year))
    print("\n" + "=" * 100, flush=True)
    print(res[["tag", "calmar", "sharpe", "max_dd_%", "total_return", "win_rate_%", "trades"]]
          .to_string(index=False), flush=True)
    print("=" * 100, flush=True)
    if not args.by_year and not res.empty:
        base = res[res.tag == "baseline"].iloc[0]
        print(f"\n베이스라인: Calmar {base['calmar']:.2f}  MDD {base['max_dd_%']:.1f}%  "
              f"Sharpe {base['sharpe']:.2f}", flush=True)
        better = res[(res.calmar > base["calmar"]) & (res["max_dd_%"] > base["max_dd_%"])]
        if better.empty or (better.tag == "baseline").all():
            print("→ Calmar↑ + MDD개선 동시 충족 규칙 없음. 적응형 미채택이 정답.", flush=True)
        else:
            print("→ 후보(Calmar↑ & MDD개선):", flush=True)
            print(better[better.tag != "baseline"][["tag", "calmar", "max_dd_%", "sharpe"]]
                  .to_string(index=False), flush=True)
            print("  ※ 단일 기간 결과 → --by-year 로 연도별 강건성 확인 필수.", flush=True)


if __name__ == "__main__":
    main()
