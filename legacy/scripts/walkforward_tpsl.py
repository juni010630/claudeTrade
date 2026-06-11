"""walkforward_tpsl.py — 적응형 TP/SL 규칙의 purged 워크포워드 OOS 검증.

핵심:
  - 적응형 TP/SL(국면/변동성 조건부)을 train 폴드에서 Calmar 최대화로 탐색
  - 선택된 규칙을 **held-out test 폴드**에서 평가 (OOS)
  - 동일 test 폴드에 고정 베이스라인(현재 3.5/4.0/1.8)도 통과시켜 사과 대 사과 비교
  - 거래 보유기간(max_hold_hours)만큼 train 꼬리를 embargo 하여 누수 차단

규칙 적용(apply_adaptive_tpsl)은 부모에서 벡터연산으로 수행하고,
워커는 결과 entries 로 run_replay 만 돌린다 → 워커는 규칙에 무관, 단순/병렬화 용이.

Usage:
  # 덤프부터 (Phase 1) + 워크포워드:
  python scripts/walkforward_tpsl.py --params config/final_v13_eth.yaml

  # 기존 컨텍스트 덤프 재사용:
  python scripts/walkforward_tpsl.py --entries data/entries_ctx.parquet

  # 하네스 무결성 체크 (항등 규칙 == 풀 백테스트):
  python scripts/walkforward_tpsl.py --check-integrity
"""
from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.adaptive_tpsl import (
    BUCKETS,
    IDENTITY_RULE,
    apply_adaptive_tpsl,
)
from scripts.run_backtest import build_engine


# ── 리플레이 워커 (ProcessPoolExecutor용, 모듈 최상위 필수) ──────────────────
def _replay_worker(args: tuple) -> dict:
    """이미 tp/sl 이 적용된 entries records 로 단일 리플레이 → 지표 dict."""
    (tag, config_path, records, since_str, until_str, capital) = args

    with open(config_path) as f:
        p = yaml.safe_load(f)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    engine = build_engine(p, capital)
    report = engine.run_replay(df, loader.iterate(since=since, until=until))

    # 주의: report.max_drawdown 은 이미 ×100 된 퍼센트값 (metrics/report.py:71).
    #       report.calmar 은 CAGR/|MDD| (metrics/returns.py) — 폴드 길이 무관 비교 가능.
    total_return = report.final_equity / capital - 1.0
    return {
        "tag": tag,
        "calmar": round(report.calmar, 3),
        "total_return": round(total_return, 4),
        "sharpe": round(report.sharpe, 3),
        "max_dd_%": round(report.max_drawdown, 1),
        "win_rate_%": round(report.win_rate, 1),
        "profit_factor": round(getattr(report, "profit_factor", float("nan")), 2),
        "final_equity": round(report.final_equity, 1),
        "trades": report.total_trades,
    }


# ── 규칙 생성 (버킷 모드별 그리드) ──────────────────────────────────────────
def _build_rules(bucket_mode: str, scales: tuple[float, ...]):
    """bucket_mode 별로 탐색할 규칙(버킷→(tp_scale, sl_scale)) 목록 생성.

    global : 전 버킷 동일 스케일      → |scales|^2 개
    adx    : adx 고/저 2그룹 독립      → |scales|^4 개
    adx_bbw: 4버킷 독립 (폭발 주의)    → |scales|^8 개  ← 비권장(좌표하강 필요)
    """
    import itertools

    pairs = list(itertools.product(scales, scales))  # (tp_scale, sl_scale)

    if bucket_mode == "global":
        return [{b: pr for b in BUCKETS} for pr in pairs]

    if bucket_mode == "adx":
        hi = ("hiadx_lobw", "hiadx_hibw")
        lo = ("loadx_lobw", "loadx_hibw")
        rules = []
        for pr_hi in pairs:
            for pr_lo in pairs:
                rule = {}
                for b in hi:
                    rule[b] = pr_hi
                for b in lo:
                    rule[b] = pr_lo
                rules.append(rule)
        return rules

    if bucket_mode == "adx_bbw":
        rules = []
        for combo in itertools.product(pairs, repeat=len(BUCKETS)):
            rules.append({b: combo[i] for i, b in enumerate(BUCKETS)})
        return rules

    raise ValueError(f"알 수 없는 bucket_mode: {bucket_mode}")


def _rule_summary(rule: dict) -> str:
    """규칙을 간결 문자열로 (대표 버킷만)."""
    hi = rule["hiadx_lobw"]
    lo = rule["loadx_lobw"]
    return f"hiADX(tp×{hi[0]},sl×{hi[1]}) / loADX(tp×{lo[0]},sl×{lo[1]})"


# ── 폴드 분할 ───────────────────────────────────────────────────────────────
def _make_folds(t_start: pd.Timestamp, t_end: pd.Timestamp, k: int):
    """[t_start, t_end] 를 K+1 등분 → 확장윈도우 K폴드.

    fold i: train=[t_start, seg_i], test=(seg_i, seg_{i+1}], i=0..K-1
    (첫 train 은 첫 세그먼트 1개)
    """
    edges = pd.date_range(t_start, t_end, periods=k + 2)
    folds = []
    for i in range(k):
        train_start = t_start
        test_start = edges[i + 1]
        test_end = edges[i + 2]
        folds.append((train_start, test_start, test_end))
    return folds


def _dispatch(pool, worker_args):
    """병렬 리플레이 — as_completed 로 완료 즉시 수집 (폴링 금지)."""
    results = []
    futures = {pool.submit(_replay_worker, a): a for a in worker_args}
    for fut in concurrent.futures.as_completed(futures):
        try:
            results.append(fut.result())
        except Exception as e:
            a = futures[fut]
            print(f"  리플레이 실패 (tag={a[0]}): {e}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="config/final_v13_eth.yaml")
    parser.add_argument("--entries", default=None,
                        help="컨텍스트 포함 entries.parquet (없으면 Phase1 덤프 실행)")
    parser.add_argument("--out", default="data/entries_ctx.parquet",
                        help="Phase1 덤프 저장 경로")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--embargo-hours", type=float, default=None,
                        help="기본=config max_hold_hours (거래 보유기간 누수 차단)")
    parser.add_argument("--bucket-mode", default="adx",
                        choices=["global", "adx", "adx_bbw"])
    parser.add_argument("--scales", default="0.7,1.0,1.3",
                        help="tp/sl 스케일 그리드 (베이스라인=1.0 중심)")
    parser.add_argument("--adx-split", type=float, default=25.0)
    parser.add_argument("--bbw-split", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--check-integrity", action="store_true",
                        help="항등 규칙 풀 리플레이 == 풀 백테스트 무결성 검증만 수행")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    capital = args.capital or bt.get("initial_capital", 100)
    since_str = args.start or bt.get("start", "2022-01-01")
    until_str = args.end or bt.get("end")

    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    embargo = args.embargo_hours
    if embargo is None:
        embargo = p.get("engine", {}).get("max_hold_hours",
                  p.get("risk", {}).get("max_hold_hours", 336))
    embargo_td = pd.Timedelta(hours=float(embargo))

    data_cfg = p.get("data", {})

    def _loader():
        return DataLoader(
            symbols=p["symbols"],
            timeframes=p["timeframes"],
            primary_tf=p.get("primary_timeframe", "1h"),
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            lookback=data_cfg.get("lookback_bars", 300),
        )

    # ── Phase 1: 컨텍스트 포함 진입 덤프 ────────────────────────────────────
    if args.entries is None:
        print(f"[Phase 1] 풀 백테스트 + 컨텍스트 덤프 ({since_str} ~ {until_str or '최신'})")
        engine = build_engine(p, capital)
        full_report, entries_df = engine.run_fill_dump(
            _loader().iterate(since=since, until=until))
        if entries_df.empty:
            print("체결 진입 없음. 설정 확인 필요.")
            return
        missing = {"regime", "adx", "bb_width_pct"} - set(entries_df.columns)
        if missing:
            print(f"덤프에 컨텍스트 컬럼 누락: {missing} — 엔진 갱신 필요.")
            return
        entries_df.to_parquet(args.out, index=False)
        print(f"  → {len(entries_df)}건 저장: {args.out}")
        print(f"  → 풀 백테 기준: Sharpe {full_report.sharpe:.2f}, "
              f"MDD {full_report.max_drawdown*100:.1f}%, "
              f"최종 ${full_report.final_equity:,.0f}")
    else:
        entries_df = pd.read_parquet(args.entries)
        full_report = None
        print(f"[Phase 1 스킵] 덤프 재사용: {args.entries} ({len(entries_df)}건)")

    entries_df["timestamp"] = pd.to_datetime(entries_df["timestamp"], utc=True)

    # ── 무결성 체크: 항등 규칙 풀 리플레이 == 풀 백테스트 ───────────────────
    if args.check_integrity:
        print("\n[무결성] 항등 규칙(스케일 1.0) 풀 리플레이 vs 풀 백테스트")
        if full_report is None:
            print("  풀 백테 기준 생성 중...")
            eng = build_engine(p, capital)
            full_report, _ = eng.run_fill_dump(_loader().iterate(since=since, until=until))
        ident = apply_adaptive_tpsl(
            entries_df, IDENTITY_RULE,
            adx_split=args.adx_split, bbw_split=args.bbw_split)
        recs = ident.assign(timestamp=ident["timestamp"].astype(str)).to_dict("records")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as pool:
            res = _dispatch(pool, [("identity", args.params, recs, since_str, until_str, capital)])
        r = res[0]
        # full_report.max_drawdown 은 이미 퍼센트값 (×100 금지)
        print(f"  풀 백테스트 : Sharpe {full_report.sharpe:.3f}  "
              f"MDD {full_report.max_drawdown:.1f}%  최종 ${full_report.final_equity:,.1f}")
        print(f"  항등 리플레이: Sharpe {r['sharpe']:.3f}  "
              f"MDD {r['max_dd_%']:.1f}%  최종 ${r['final_equity']:,.1f}  거래 {r['trades']}")
        d_sh = abs(r["sharpe"] - full_report.sharpe)
        d_mdd = abs(r["max_dd_%"] - full_report.max_drawdown)
        d_eq_pct = abs(r["final_equity"] - full_report.final_equity) / max(full_report.final_equity, 1) * 100
        # look-ahead 판정은 Sharpe/MDD 일치로. 최종자산 괴리는 run_replay 의 리스크/사이징
        # 재실행에서 오는 고유 경로편차(기존 fast_sweep 등도 공유) → 정보용으로만 출력.
        ok = d_sh < 0.10 and d_mdd < 3.0
        print(f"  → Sharpe 괴리 {d_sh:.3f}, MDD 괴리 {d_mdd:.1f}%p, 최종자산 괴리 {d_eq_pct:.1f}% "
              f"(리플레이 경로편차, 정보용)")
        print(f"  → 무결성(look-ahead 無): {'✅ 통과' if ok else '⚠️ 점검 필요'} "
              f"(Sharpe/MDD 일치 기준)")
        return

    # ── 폴드 구성 ───────────────────────────────────────────────────────────
    t0, t1 = entries_df["timestamp"].min(), entries_df["timestamp"].max()
    folds = _make_folds(t0, t1, args.folds)
    scales = tuple(float(x) for x in args.scales.split(","))
    rules = _build_rules(args.bucket_mode, scales)
    print(f"\n[워크포워드] {args.folds}폴드 | 버킷={args.bucket_mode} | "
          f"규칙 {len(rules)}개/폴드 | embargo {embargo}h | workers={args.workers}")

    fold_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fi, (tr_start, te_start, te_end) in enumerate(folds):
            # train: test_start 이전 진입, embargo 만큼 꼬리 제거 (보유기간 누수 차단)
            tr_mask = entries_df["timestamp"] <= (te_start - embargo_td)
            te_mask = (entries_df["timestamp"] > te_start) & (entries_df["timestamp"] <= te_end)
            train_df = entries_df[tr_mask]
            test_df = entries_df[te_mask]
            print(f"\n── 폴드 {fi+1}/{args.folds} | "
                  f"train≤{(te_start-embargo_td).date()} ({len(train_df)}건) | "
                  f"test {te_start.date()}~{te_end.date()} ({len(test_df)}건)")
            if len(train_df) < 20 or len(test_df) < 5:
                print("  표본 부족 → 스킵")
                continue

            # 1) train 그리드 탐색 (Calmar 최대)
            tr_since, tr_until = str(tr_start), str(te_start - embargo_td)
            train_args = []
            for ri, rule in enumerate(rules):
                applied = apply_adaptive_tpsl(
                    train_df, rule, adx_split=args.adx_split, bbw_split=args.bbw_split)
                recs = applied.assign(
                    timestamp=applied["timestamp"].astype(str)).to_dict("records")
                train_args.append((ri, args.params, recs, tr_since, tr_until, capital))
            tr_res = _dispatch(pool, train_args)
            tr_res = [r for r in tr_res if r["trades"] > 0]
            if not tr_res:
                print("  train 유효 결과 없음 → 스킵")
                continue
            best = max(tr_res, key=lambda r: r["calmar"])
            best_rule = rules[best["tag"]]
            print(f"  best(train Calmar {best['calmar']}): {_rule_summary(best_rule)}")

            # 2) test 폴드 OOS 평가: 적응형(best) vs 베이스라인(항등)
            te_since, te_until = str(te_start), str(te_end)
            adapt = apply_adaptive_tpsl(
                test_df, best_rule, adx_split=args.adx_split, bbw_split=args.bbw_split)
            base = apply_adaptive_tpsl(
                test_df, IDENTITY_RULE, adx_split=args.adx_split, bbw_split=args.bbw_split)
            oos_args = [
                ("adapt", args.params,
                 adapt.assign(timestamp=adapt["timestamp"].astype(str)).to_dict("records"),
                 te_since, te_until, capital),
                ("base", args.params,
                 base.assign(timestamp=base["timestamp"].astype(str)).to_dict("records"),
                 te_since, te_until, capital),
            ]
            oos = {r["tag"]: r for r in _dispatch(pool, oos_args)}
            a, b = oos.get("adapt"), oos.get("base")
            if a and b:
                print(f"  OOS 적응형 : Calmar {a['calmar']}  Sharpe {a['sharpe']}  "
                      f"MDD {a['max_dd_%']}%  ret {a['total_return']*100:+.1f}%  WR {a['win_rate_%']}%")
                print(f"  OOS 베이스 : Calmar {b['calmar']}  Sharpe {b['sharpe']}  "
                      f"MDD {b['max_dd_%']}%  ret {b['total_return']*100:+.1f}%  WR {b['win_rate_%']}%")
                fold_rows.append({
                    "fold": fi + 1,
                    "rule": _rule_summary(best_rule),
                    "a_calmar": a["calmar"], "b_calmar": b["calmar"],
                    "a_mdd": a["max_dd_%"], "b_mdd": b["max_dd_%"],
                    "a_sharpe": a["sharpe"], "b_sharpe": b["sharpe"],
                    "a_ret": a["total_return"], "b_ret": b["total_return"],
                })

    # ── 집계 ────────────────────────────────────────────────────────────────
    if not fold_rows:
        print("\n유효 폴드 없음.")
        return
    res = pd.DataFrame(fold_rows)
    print("\n" + "=" * 92)
    print("OOS 워크포워드 결과 (적응형 a vs 베이스라인 b)")
    print("=" * 92)
    print(res.to_string(index=False))
    print("-" * 92)
    wins = (res["a_calmar"] > res["b_calmar"]).sum()
    mdd_better = (res["a_mdd"] > res["b_mdd"]).sum()  # MDD는 음수 → 큰 값이 덜 깊음
    print(f"적응형 Calmar 우위 폴드: {wins}/{len(res)}  | MDD 개선 폴드: {mdd_better}/{len(res)}")
    print(f"평균 Calmar  적응형 {res['a_calmar'].mean():.3f} vs 베이스 {res['b_calmar'].mean():.3f}")
    print(f"평균 MDD%    적응형 {res['a_mdd'].mean():.1f} vs 베이스 {res['b_mdd'].mean():.1f}")
    print(f"평균 Sharpe  적응형 {res['a_sharpe'].mean():.3f} vs 베이스 {res['b_sharpe'].mean():.3f}")
    print("=" * 92)
    print("\n채택 게이트: OOS 평균 Calmar↑ + 평균 MDD↑(덜 깊음) + 대다수 폴드 우위 + 규칙 안정")


if __name__ == "__main__":
    main()
