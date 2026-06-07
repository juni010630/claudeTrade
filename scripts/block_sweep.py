"""block_sweep.py — 버킷 진단에서 나온 차단 후보를 한 개씩 단독 차단하고 풀 백테스트로 비교.

각 후보를 독립적으로 차단(교차 없음) → engine.run() 풀 백테스트(리플레이 아님, 충실도 100%).
전기간(2022-2026) + 2025-2026 OOS 동시 평가. 병렬(ProcessPoolExecutor), 폴링 없음.

Usage:
  python3 -u scripts/block_sweep.py --params config/final_v13_eth.yaml --workers 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


# ── 차단 후보 (단독 차단, 진입 시점 속성만 사용 → look-ahead 없음) ──────────────
def block_configs() -> dict[str, dict]:
    """name -> config 오버라이드(dict). baseline = 빈 dict."""
    return {
        "baseline":         {},
        "block_eth_long":   {"symbol_block_directions": {"ETHUSDT": ["long"]}},
        "block_arb_long":   {"symbol_block_directions": {"ARBUSDT": ["long"]}},
        "block_S_emacross": {"strategy_block_tiers": {"ema_cross": ["S"]}},
        "block_S_all":      {"strategy_block_tiers": {"ema_cross": ["S"], "multi_tf_breakout": ["S"]}},
        "block_wednesday":  {"block_weekdays": [2]},  # 0=월 .. 2=수
        # 아래는 거짓 양성(WR<40%지만 PF 2.51, +$483K) — 차단 시 손실 확인용 대조군
        "block_btc_short":  {"symbol_block_directions": {"BTCUSDT": ["short"]}},
        # 교차 차단 (OOS 양대 승자 결합)
        "Sema+btc_short":   {"strategy_block_tiers": {"ema_cross": ["S"]},
                             "symbol_block_directions": {"BTCUSDT": ["short"]}},
        "Sema+eth_long":    {"strategy_block_tiers": {"ema_cross": ["S"]},
                             "symbol_block_directions": {"ETHUSDT": ["long"]}},
    }


def _metrics(report, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"trades": 0, "WR%": 0.0, "PF": 0.0}
    wins = (df["pnl"] > 0).sum()
    gw = df.loc[df["pnl"] > 0, "pnl"].sum()
    gl = abs(df.loc[df["pnl"] <= 0, "pnl"].sum())
    return {
        "trades": len(df),
        "WR%": round(wins / len(df) * 100, 1),
        "PF": round(gw / gl, 2) if gl > 0 else float("inf"),
    }


def _worker(args: tuple) -> dict:
    name, params_path, overrides, since_str, until_str, period_tag = args
    with open(params_path) as f:
        p = yaml.safe_load(f)
    # 차단 오버라이드 병합 (symbol_block_directions 등은 기존 키 병합 필요)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            merged = copy.deepcopy(p[k]);
            for sk, sv in v.items():
                merged[sk] = list(set(merged.get(sk, [])) | set(sv))
            p[k] = merged
        else:
            p[k] = v

    bt = p.get("backtest", {})
    cap = bt.get("initial_capital", 100)
    since = pd.Timestamp(since_str, tz="UTC")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None
    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 300),
    )
    engine = build_engine(p, cap)
    report = engine.run(loader.iterate(since=since, until=until))
    df = engine.ledger.to_dataframe()
    m = _metrics(report, df)
    return {
        "name": name, "period": period_tag,
        "final_equity": round(report.final_equity, 1),
        "sharpe": round(report.sharpe, 3),
        "calmar": round(report.calmar, 2),
        "max_dd_%": round(report.max_drawdown, 1),
        **m,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="config/final_v13_eth.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--by-year", action="store_true", help="연도별(2022/23/24/25-26) 분해 평가")
    ap.add_argument("--only", default=None, help="쉼표구분 설정명만 실행 (baseline 자동 포함)")
    args = ap.parse_args()

    cfgs = block_configs()
    if args.only:
        keep = set(args.only.split(",")) | {"baseline"}
        cfgs = {k: v for k, v in cfgs.items() if k in keep}
    with open(args.params) as f:
        _bt = yaml.safe_load(f).get("backtest", {})
    _end = _bt.get("end")  # config 기간 준수 → 문서 baseline과 정렬
    if args.by_year:
        periods = [("2022-01-01", "2022-12-31", "2022"), ("2023-01-01", "2023-12-31", "2023"),
                   ("2024-01-01", "2024-12-31", "2024"), ("2025-01-01", _end, "2025-26")]
    else:
        periods = [("2022-01-01", _end, "full"), ("2025-01-01", _end, "2025-26")]
    jobs = []
    for name, ov in cfgs.items():
        for since, until, tag in periods:
            jobs.append((name, args.params, ov, since, until, tag))

    print(f"[차단 스윕] {len(cfgs)}개 설정 × {len(periods)}기간 = {len(jobs)}개 백테스트 | workers={args.workers}",
          flush=True)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, j): (j[0], j[5]) for j in jobs}
        for fut in concurrent.futures.as_completed(futs):
            nm, tag = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  ✓ {r['name']:<18} [{r['period']:<7}] "
                      f"eq ${r['final_equity']:>12,.0f}  Sharpe {r['sharpe']:>6.3f}  "
                      f"Calmar {r['calmar']:>6.2f}  MDD {r['max_dd_%']:>6.1f}%  "
                      f"WR {r['WR%']:>4.1f}%  PF {r['PF']:>6.2f}  n={r['trades']}", flush=True)
            except Exception as e:
                print(f"  ✗ {nm}[{tag}] 실패: {e}", flush=True)

    res = pd.DataFrame(results)
    for tag in [pt[2] for pt in periods]:
        sub = res[res.period == tag].copy()
        if sub.empty:
            continue
        base = sub[sub.name == "baseline"].iloc[0]
        sub = sub.sort_values("calmar", ascending=False)
        print(f"\n{'='*108}\n  [{tag}] 차단 후보 비교 (baseline 대비)\n{'='*108}", flush=True)
        cols = ["name", "final_equity", "sharpe", "calmar", "max_dd_%", "WR%", "PF", "trades"]
        print(sub[cols].to_string(index=False), flush=True)
        print(f"\n  baseline: eq ${base['final_equity']:,.0f}  Sharpe {base['sharpe']}  "
              f"Calmar {base['calmar']}  MDD {base['max_dd_%']}%  WR {base['WR%']}%  PF {base['PF']}", flush=True)
        better = sub[(sub.name != "baseline") & (sub.calmar > base["calmar"]) &
                     (sub["WR%"] >= base["WR%"])]
        if better.empty:
            print("  → Calmar↑ + WR↑ 동시 충족 단독 차단 없음.", flush=True)
        else:
            print("  → 후보(Calmar↑ & WR↑):", ", ".join(better.name.tolist()), flush=True)


if __name__ == "__main__":
    main()
