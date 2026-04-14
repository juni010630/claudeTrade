"""여러 워커의 WF 최적화 결과를 하나로 합치고 분석."""
from __future__ import annotations
import json
import statistics
from pathlib import Path

BASE = Path(__file__).parent.parent


def load_all() -> list[dict]:
    files = list(BASE.glob("optimize_wf_aggressive_results*.json"))
    all_results = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            for r in data:
                r["_source"] = f.name
            all_results.extend(data)
            print(f"  {f.name}: {len(data)}개")
        except Exception as e:
            print(f"  {f.name}: 로드 실패 ({e})")
    return all_results


def analyze(results: list[dict]) -> None:
    valid = [r for r in results if r["score"] > -990]
    invalid = len(results) - len(valid)

    print(f"\n총 {len(results)}개 트라이얼  (유효 {len(valid)}개, MDD초과/미달 {invalid}개)")

    if not valid:
        print("유효한 결과 없음")
        return

    top10 = sorted(valid, key=lambda r: r["score"], reverse=True)[:10]
    print("\n=== Top-10 ===")
    for r in top10:
        syms = "+".join(s.replace("USDT", "") for s in r["symbols"])
        src = r.get("_source", "?")
        print(
            f"  #{r['trial']:>3}  score={r['score']:+.4f}  "
            f"train_sh={r['train']['sharpe']:.2f}  "
            f"test_sh={r['test']['sharpe']:.2f}  "
            f"test_ret={r['test']['return']:+.1f}%  "
            f"MDD={r['test']['mdd']:.1f}%  "
            f"[{syms}]  ({src})"
        )

    returns = [r["test"]["return"] for r in valid]
    sharpes = [r["test"]["sharpe"] for r in valid]
    mdds = [r["test"]["mdd"] for r in valid]
    pos_ret = [r for r in valid if r["test"]["return"] > 0]

    print("\n=== 전체 통계 ===")
    print(f"  검증 수익률:  평균={statistics.mean(returns):+.1f}%  최대={max(returns):+.1f}%  최소={min(returns):+.1f}%")
    print(f"  검증 Sharpe:  평균={statistics.mean(sharpes):.3f}  최대={max(sharpes):.3f}  최소={min(sharpes):.3f}")
    print(f"  검증 MDD:     평균={statistics.mean(mdds):.1f}%  최악={min(mdds):.1f}%")
    print(f"  양수 수익 비율: {len(pos_ret)}/{len(valid)} ({100*len(pos_ret)/len(valid):.0f}%)")

    # 스코어 > -2.0 인 결과만 파라미터 패턴 분석
    strong = [r for r in valid if r["score"] > -2.0]
    if strong:
        print(f"\n=== score > -2.0 결과 {len(strong)}개 공통 패턴 ===")
        from collections import Counter
        # 심볼 빈도
        sym_counter: Counter = Counter()
        for r in strong:
            for s in r["symbols"]:
                sym_counter[s.replace("USDT", "")] += 1
        print("  자주 등장한 코인:", dict(sym_counter.most_common(7)))
        # 파라미터 빈도
        param_vals: dict[str, Counter] = {}
        for r in strong:
            for k, (_, new_val) in r.get("params_diff", {}).items():
                if k == "symbols":
                    continue
                if k not in param_vals:
                    param_vals[k] = Counter()
                param_vals[k][str(new_val)] += 1
        for k, c in sorted(param_vals.items()):
            if len(c) > 0:
                top_val = c.most_common(1)[0]
                print(f"  {k}: 최빈값 {top_val[0]} ({top_val[1]}회)")

    # 병합 저장
    out = BASE / "optimize_wf_aggressive_results_merged.json"
    out.write_text(json.dumps(valid, indent=2, ensure_ascii=False))
    print(f"\n병합 결과 저장: {out.name} ({len(valid)}개)")


if __name__ == "__main__":
    print("=== WF Aggressive 결과 병합 ===")
    print("로드 중:")
    results = load_all()
    analyze(results)
