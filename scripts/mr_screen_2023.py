"""mean_reversion 변형 2023년 단독 빠른 스크리닝 (풀 스윕 전 1차 관문)."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

VARIANTS = {
    "mr_4h":        {"signal_tf": "4h", "max_adx": 25.0, "rsi_oversold": 30.0, "rsi_overbought": 70.0, "atr_tp_mult": 1.5, "atr_sl_mult": 1.5},
    "mr_4h_strict": {"signal_tf": "4h", "max_adx": 25.0, "rsi_oversold": 25.0, "rsi_overbought": 75.0, "atr_tp_mult": 1.5, "atr_sl_mult": 1.5},
    "mr_4h_tp1":    {"signal_tf": "4h", "max_adx": 25.0, "rsi_oversold": 30.0, "rsi_overbought": 70.0, "atr_tp_mult": 1.0, "atr_sl_mult": 2.0},
    "mr_1h_strict": {"signal_tf": "1h", "max_adx": 20.0, "rsi_oversold": 25.0, "rsi_overbought": 75.0, "bb_std": 2.5, "atr_tp_mult": 1.5, "atr_sl_mult": 1.5},
}


def run(item):
    name, mrcfg = item
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    with open("config/final_v15_gate.yaml") as f:
        p = yaml.safe_load(f)
    base = {"enabled": True, "use_regime_gate": False, "bb_period": 20, "bb_std": 2.0}
    base.update(mrcfg)
    p["strategies"]["mean_reversion"] = base
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    r = eng.run(loader.iterate(since=pd.Timestamp("2023-01-01", tz="UTC"),
                               until=pd.Timestamp("2023-12-31", tz="UTC")))
    df = eng.ledger.to_dataframe()
    mr = df[df.strategy == "mean_reversion"]
    gw = mr.loc[mr.pnl > 0, "pnl"].sum(); gl = -mr.loc[mr.pnl <= 0, "pnl"].sum()
    return (name, len(mr), round(mr.pnl.sum(), 1), round(gw / gl, 2) if gl > 0 else 99.0,
            round(r.final_equity, 1))


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for name, n, pnl, pf, eq in ex.map(run, VARIANTS.items()):
            print(f"{name:14s} mr거래 {n:3d}건  mr_pnl {pnl:+7.1f}  PF {pf:5.2f}  | 전체 2023 eq ${eq}",
                  flush=True)
    print("참고: v15 baseline 2023 eq $87")


if __name__ == "__main__":
    main()
