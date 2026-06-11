"""p7_w_hi(RSI 강모멘텀 가중 v16) + 슬리브 배분 곡선 — 레버리지 MDD를 슬리브로 헤지."""
from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

# p7_w_hi config 생성 (v16 + RSI period7 강모멘텀 ×1.5)
_p = yaml.safe_load(open("config/final_v16_slwide.yaml"))
_p["rsi_momentum"] = {"period": 7,
                      "weight": {"low_thr": 0, "low_mult": 1.0, "high_thr": 70, "high_mult": 1.5}}
with open("config/p7_whi.yaml", "w") as f:
    yaml.dump(_p, f, allow_unicode=True, sort_keys=True)


def bt(args):
    cfg, ptf = args
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine
    p = yaml.safe_load(open(cfg))
    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf=ptf,
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp("2022-01-01", tz="UTC"),
                           until=pd.Timestamp("2026-04-23", tz="UTC")))
    return eng.equity_curve.to_series()


def sharpe(r): return r.mean()/r.std()*np.sqrt(365) if r.std() > 0 else 0
def mdd(r):
    c = (1+r).cumprod(); return ((c-c.cummax())/c.cummax()).min()*100


def main():
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        ep7, esl = list(ex.map(bt, [("config/p7_whi.yaml", "1h"),
                                    ("config/sleeve_meanrev.yaml", "1d")]))
    dp7 = ep7.resample("1D").last().pct_change()
    dsl = esl.resample("1D").last().pct_change()
    j = pd.concat([dp7, dsl], axis=1, keys=["p7", "sl"]).dropna()
    print(f"p7_w_hi vs 슬리브 일별 상관: {j.p7.corr(j.sl):+.3f}\n")
    print(f"{'배분(p7:슬)':>12} | {'총수익%':>14} {'Sharpe':>6} {'MDD%':>6}")
    for w in [100, 90, 80, 70, 60, 50]:
        x = w/100
        r = x*j.p7 + (1-x)*j.sl
        eq = (1+r).cumprod().iloc[-1]
        print(f"{w:>4}:{100-w:<3}     | {(eq-1)*100:>14,.0f} {sharpe(r):>6.2f} {mdd(r):>6.0f}")


if __name__ == "__main__":
    main()
