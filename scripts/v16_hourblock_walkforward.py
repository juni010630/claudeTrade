"""시간대 차단 walk-forward 과적합 검증 (shift보다 엄격).
1) 전략·진입시간별 거래수 (표본충분성)
2) IS 시간별손익→차단셋 도출→OOS 적용. IS차단 vs 무차단 vs 원본차단 OOS 성과.
   IS도출 차단이 OOS서 무차단보다 우월하면 진짜효과, 아니면 과적합. 양방향.
3) IS vs OOS 시간별 평균손익 상관 (시간효과 지속성)."""
from __future__ import annotations

import concurrent.futures
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

BASE = "config/final_v16_slwide.yaml"
STRATS = ["ema_cross", "multi_tf_breakout"]


def bt(blocks, start, end, want_trades=False):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = copy.deepcopy(yaml.safe_load(open(BASE)))
    if blocks == "none":
        p.pop("strategy_block_hours", None)
    elif blocks != "orig":
        p["strategy_block_hours"] = blocks
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(start, tz="UTC"), until=pd.Timestamp(end, tz="UTC")))
    eq = eng.equity_curve.to_series()
    d = eq.resample("1D").last().pct_change().dropna()
    sh = d.mean()/d.std()*np.sqrt(365) if d.std() > 0 else 0
    out = {"final": eq.iloc[-1], "sh": sh}
    if want_trades:
        rows = [(r.strategy, pd.Timestamp(r.entry_time).hour, pd.Timestamp(r.entry_time), r.pnl)
                for r in eng.ledger.records]
        out["trades"] = pd.DataFrame(rows, columns=["strategy", "hour", "entry_time", "pnl"])
    return out


def _bt_dispatch(args):
    return bt(*args)


def derive_blocks(trades, n_by_strat):
    """IS 거래에서 전략별 시간 평균손익 최악 K(원본개수)시간 차단."""
    blocks = {}
    for st in STRATS:
        t = trades[trades.strategy == st]
        m = t.groupby("hour").pnl.mean()
        # 거래 없는 시간은 중립(0)으로 — 최악 선정에서 자연 제외되게 큰 값
        m = m.reindex(range(24)).fillna(1e9)
        worst = list(m.nsmallest(n_by_strat[st]).index)
        blocks[st] = sorted(int(h) for h in worst)
    return blocks


def main():
    orig = yaml.safe_load(open(BASE))["strategy_block_hours"]
    n_by = {st: len(orig[st]) for st in STRATS}
    print(f"원본 차단 개수: { {st: n_by[st] for st in STRATS} } (ema 24중, multi 24중)\n")

    # 무차단 풀 백테 → 거래 시간 통계
    full = bt("none", "2022-01-01", "2026-04-23", want_trades=True)
    tr = full["trades"]

    print("=== 진입 시간별 거래 수 (무차단, 표본충분성) ===")
    for st in STRATS:
        c = tr[tr.strategy == st].groupby("hour").size().reindex(range(24)).fillna(0).astype(int)
        print(f"{st}: 총{c.sum()}건, 시간당 평균 {c.sum()/24:.1f}건")
        print("  " + " ".join(f"{h:02d}:{c[h]}" for h in range(24)))

    splits = {
        "전반IS→후반OOS": ("2022-01-01", "2024-01-01", "2024-01-01", "2026-04-23"),
        "후반IS→전반OOS": ("2024-01-01", "2026-04-23", "2022-01-01", "2024-01-01"),
    }

    print("\n=== Walk-forward: IS도출 차단을 OOS 적용 ===")
    for name, (is_s, is_e, oos_s, oos_e) in splits.items():
        is_tr = tr[(tr.entry_time >= is_s) & (tr.entry_time < is_e)]
        wf_blocks = derive_blocks(is_tr, n_by)
        # OOS 3종 병렬
        with concurrent.futures.ProcessPoolExecutor(max_workers=3) as ex:
            r_none, r_wf, r_orig = list(ex.map(
                _bt_dispatch,
                [("none", oos_s, oos_e), (wf_blocks, oos_s, oos_e), ("orig", oos_s, oos_e)]))
        print(f"\n[{name}]  IS도출 차단: {wf_blocks}")
        print(f"  OOS 무차단      : ${r_none['final']:>12,.0f}  Sh{r_none['sh']:.2f}")
        print(f"  OOS IS도출차단  : ${r_wf['final']:>12,.0f}  Sh{r_wf['sh']:.2f}   "
              f"{'✅IS선택 일반화' if r_wf['sh'] > r_none['sh'] else '❌IS선택 무효(과적합)'}")
        print(f"  OOS 원본차단    : ${r_orig['final']:>12,.0f}  Sh{r_orig['sh']:.2f}   (원본은 풀데이터 도출=OOS오염 참고용)")

    # 시간별 평균손익 지속성 상관
    print("\n=== IS vs OOS 시간별 평균손익 상관 (지속성; 0근처=노이즈) ===")
    h1 = tr[tr.entry_time < "2024-01-01"]
    h2 = tr[tr.entry_time >= "2024-01-01"]
    for st in STRATS:
        a = h1[h1.strategy == st].groupby("hour").pnl.mean().reindex(range(24))
        b = h2[h2.strategy == st].groupby("hour").pnl.mean().reindex(range(24))
        both = pd.concat([a, b], axis=1).dropna()
        corr = both.iloc[:, 0].corr(both.iloc[:, 1]) if len(both) > 3 else float("nan")
        print(f"  {st}: corr={corr:+.2f}  (공통시간 {len(both)}개)")


if __name__ == "__main__":
    main()
