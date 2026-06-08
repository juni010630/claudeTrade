"""4/24~현재: 슬리브(평균회귀) vs v16(추세) 동기간 백테 + 거래목록."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

START, END, CAP = "2026-04-24", "2026-06-08", 100.0


def bt(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, CAP)
    r = eng.run(loader.iterate(since=pd.Timestamp(START, tz="UTC"),
                               until=pd.Timestamp(END, tz="UTC")))
    df = eng.ledger.to_dataframe()
    trades = []
    if len(df):
        for _, t in df.iterrows():
            trades.append((str(t.entry_time)[:10], t.symbol, t.direction, t.exit_reason, round(t.pnl, 2)))
    return {"cfg": cfg.split("/")[-1], "final": round(r.final_equity, 1),
            "ret": round((r.final_equity/CAP-1)*100, 1), "n": len(df),
            "wr": round((df.pnl > 0).mean()*100) if len(df) else 0, "trades": trades}


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        v16, sl = list(ex.map(bt, [("config/final_v16_slwide.yaml", "1h"),
                                   ("config/sleeve_meanrev.yaml", "1d")]))
    for r in [v16, sl]:
        print(f"\n=== {r['cfg']} ({START}~{END}, $100) ===")
        print(f"최종 ${r['final']} ({r['ret']:+.1f}%) | {r['n']}거래 승률 {r['wr']}%")
        for et, sym, d, reason, pnl in r["trades"]:
            print(f"  {et} {sym:9} {d:5} {reason:13} ${pnl:+7.2f}")


if __name__ == "__main__":
    main()
