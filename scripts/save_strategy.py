"""확정 전략을 백테 1회 돌려 strategy_registry.json에 저장 — 이후 재백테 없이 비교.

사용:
  python3 scripts/save_strategy.py <키이름> <config경로> [--primary-tf 1h] [--status experimental] [--desc "설명"]
예:
  python3 scripts/save_strategy.py sleeve_meanrev config/sleeve_meanrev.yaml --primary-tf 1d --desc "RSI평균회귀 슬리브"
조회만:
  python3 scripts/save_strategy.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

REG = Path("data/strategy_registry.json")
START, END, CAP = "2022-01-01", "2026-04-23", 100.0


def load_reg():
    return json.loads(REG.read_text()) if REG.exists() else {"strategies": {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?")
    ap.add_argument("config", nargs="?")
    ap.add_argument("--primary-tf", default="1h")
    ap.add_argument("--status", default="experimental")
    ap.add_argument("--desc", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    reg = load_reg()
    if args.list or not args.key:
        print(f"{'키':22} {'status':12} {'최종$':>12} {'Sharpe':>6} {'MDD봉%':>7} {'거래':>5}")
        for k, v in reg["strategies"].items():
            print(f"{k:22} {v.get('status',''):12} {v.get('final_usd_from_100',0):>12,.0f} "
                  f"{v.get('sharpe',0):>6.2f} {v.get('mdd_bar_pct',0):>7.1f} {v.get('trades',0):>5}")
        return

    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(args.config))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"],
                        primary_tf=args.primary_tf, cache_dir="data/cache", lookback=300)
    eng = build_engine(p, CAP)
    r = eng.run(loader.iterate(since=pd.Timestamp(START, tz="UTC"), until=pd.Timestamp(END, tz="UTC")))
    df = eng.ledger.to_dataframe()
    # 일별 MDD
    eqd = eng.equity_curve.to_series().resample("1D").last().ffill()
    mdd_d = ((eqd-eqd.cummax())/eqd.cummax()).min()*100
    df["year"] = pd.to_datetime(df.entry_time).dt.year.clip(upper=2025)
    yr = df.groupby("year").pnl  # not used for usd; keep simple
    gw = df.loc[df.pnl > 0, "pnl"].sum(); gl = -df.loc[df.pnl <= 0, "pnl"].sum()
    reg["strategies"][args.key] = {
        "config": args.config, "status": args.status, "desc": args.desc,
        "period": f"{START}~{END}",
        "final_usd_from_100": round(r.final_equity),
        "sharpe": round(r.sharpe, 3), "calmar": round(r.calmar, 2),
        "mdd_bar_pct": round(r.max_drawdown, 1), "mdd_daily_pct": round(mdd_d, 1),
        "win_rate_pct": round((df.pnl > 0).mean()*100, 1),
        "profit_factor": round(gw/gl, 2) if gl > 0 else 99,
        "trades": len(df),
    }
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=2))
    print(f"저장됨: {args.key} → 최종 ${r.final_equity:,.0f} Sharpe {r.sharpe:.2f} "
          f"MDD봉 {r.max_drawdown:.1f}% MDD일별 {mdd_d:.1f}% 거래 {len(df)}")


if __name__ == "__main__":
    main()
