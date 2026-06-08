"""최근 기간 v13 vs v16 백테 병렬 비교 (실거래 대조용)."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

START, END, CAP = "2026-04-14", "2026-06-08", 100.0


def run(cfg_path):
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg_path))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, CAP)
    r = eng.run(loader.iterate(since=pd.Timestamp(START, tz="UTC"),
                               until=pd.Timestamp(END, tz="UTC")))
    df = eng.ledger.to_dataframe()
    rows = []
    eq = CAP
    for _, t in df.iterrows() if len(df) else []:
        pct_of_eq = t.pnl / eq * 100
        eq += t.pnl
        rows.append((str(t.entry_time)[:16], t.symbol, t.strategy, t.direction,
                     t.exit_reason, t.pnl, pct_of_eq))
    return {"final": r.final_equity, "sharpe": r.sharpe, "mdd": r.max_drawdown,
            "n": len(df), "wins": int((df.pnl > 0).sum()) if len(df) else 0, "trades": rows}


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        f13 = ex.submit(run, "config/final_v13_eth.yaml")
        f16 = ex.submit(run, "config/final_v16_slwide.yaml")
        v13, v16 = f13.result(), f16.result()

    for tag, res in [("v13 (실거래 config)", v13), ("v16 (현 라이브)", v16)]:
        wr = res["wins"] / res["n"] * 100 if res["n"] else 0
        print(f"\n{'='*70}\n[{tag}] {START}~{END} 시작 ${CAP:.0f}")
        print(f"최종 ${res['final']:,.2f} ({(res['final']/CAP-1)*100:+.1f}%) | "
              f"Sharpe {res['sharpe']:.2f} | MDD {res['mdd']:.1f}% | "
              f"{res['n']}건 승률 {wr:.0f}%")
        print(f"{'진입시각':16} {'심볼':9} {'전략':18} {'방향':5} {'청산':14} {'PnL':>9} {'당시equity%':>10}")
        for et, sym, strat, d, reason, pnl, pct in res["trades"]:
            print(f"{et:16} {sym:9} {strat:18} {d:5} {reason:14} ${pnl:+8.2f} {pct:+9.1f}%")


if __name__ == "__main__":
    main()
