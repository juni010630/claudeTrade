"""반복 2 — 1d EMA크로스 follow 패밀리 정밀 측량 + OOS 판정.

mode 'is'  : 확장 IS 스윕 (쌍 9종 × 출구 3종 × 유동성 2종) — OOS 봉인 유지
mode 'oos' : 지정 설정의 전구간 트레이드 연도별 + 포트폴리오 시뮬 + v17 상관 + 비용 스트레스
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import COST, atr, ema, load_universe
from scripts.newedge_grid import sim_fast
from scripts.newedge_l2 import load_full, portfolio_sim, v17_daily, yearly

PAIRS = [(5, 20), (8, 21), (10, 50), (15, 75), (10, 100), (20, 50), (20, 100), (30, 150), (50, 200)]
EXITS = [(4.0, 2.0), (6.0, 3.0), (8.0, 4.0)]
HOLD = 60


def macross_sig(df, f, s, liq_min=0.0):
    c = df["close"]
    fa, sl_ = ema(c, f), ema(c, s)
    up = (fa > sl_) & (fa.shift(1) <= sl_.shift(1))
    dn = (fa < sl_) & (fa.shift(1) >= sl_.shift(1))
    if liq_min > 0:
        dvol = (c * df["volume"]).rolling(30).median().shift(1)
        ok = dvol > liq_min
        up &= ok
        dn &= ok
    return np.where(up.shift(1).fillna(False), 1, np.where(dn.shift(1).fillna(False), -1, 0))


def stats_yearly(tr, years):
    t = pd.DataFrame(tr, columns=["t", "r"])
    t["y"] = pd.to_datetime(t.t).dt.year
    gw = t.r[t.r > 0].sum()
    gl = -t.r[t.r <= 0].sum()
    out = dict(n=len(t), wr=round((t.r > 0).mean() * 100), pf=round(gw / gl if gl > 0 else 99, 3),
               avg_bp=round(t.r.mean() * 10000), posY=0)
    for y in years:
        s = t[t.y == y]
        out[f"y{y%100}"] = round(s.r.mean() * 10000) if len(s) else 0
        out[f"n{y%100}"] = len(s)
    out["posY"] = sum(1 for y in years if t[t.y == y].r.sum() > 0)
    return out


def run_is():
    dfs = load_universe("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"universe={len(dfs)} (IS only)")
    rows = []
    for f, s in PAIRS:
        for liq in (0.0, 10e6):
            sigs = {sym: macross_sig(df, f, s, liq) for sym, df in dfs.items()}
            for tp, sl in EXITS:
                tr = []
                for sym, df in dfs.items():
                    tr += sim_fast(df, sigs[sym], tp, sl, HOLD)
                st = stats_yearly(tr, [2022, 2023, 2024])
                rows.append(dict(f=f, s=s, liq=int(liq / 1e6), tp=tp, sl=sl, **st))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.sort_values("pf", ascending=False).to_string(index=False))
    out.to_csv(Path(__file__).parent.parent / "data" / "results" / "newedge_it2_is.csv", index=False)


def run_oos(f, s, tp, sl, liq):
    dfs = load_full("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    print(f"== OOS 개봉: ema{f}x{s} tp{tp}/sl{sl} liq>{liq/1e6:.0f}M ==")
    sigs = {sym: macross_sig(df, f, s, liq) for sym, df in dfs.items()}
    tr = []
    for sym, df in dfs.items():
        tr += sim_fast(df, sigs[sym], tp, sl, HOLD)
    st = stats_yearly(tr, [2022, 2023, 2024, 2025, 2026])
    print("트레이드 레벨 연도별 avg bp:", st)

    sigmap = {sym: (df, sigs[sym]) for sym, df in dfs.items()}
    for cost in (0.0010, 0.0030):
        eq, tdf, stats = portfolio_sim(sigmap, tp, sl, HOLD, risk_f=0.02, max_pos=8, cost_rt=cost)
        oos_eq = eq[eq.index >= "2025-01-01"]
        oos_ret = oos_eq.pct_change().dropna()
        oos_sh = oos_ret.mean() / oos_ret.std() * np.sqrt(365) if oos_ret.std() > 0 else 0
        oos_mdd = ((oos_eq - oos_eq.cummax()) / oos_eq.cummax()).min()
        v17 = v17_daily()
        mine = eq.pct_change().dropna()
        j = pd.concat([mine, v17.reindex(mine.index).fillna(0)], axis=1).dropna()
        j.columns = ["new", "v17"]
        print(f"[cost={cost*10000:.0f}bp] FULL {stats} yearly={yearly(eq)}")
        print(f"  OOS: ret={round((oos_eq.iloc[-1]/oos_eq.iloc[0]-1)*100,1)}% sharpe={round(oos_sh,2)} "
              f"mdd={round(oos_mdd*100,1)}%  v17corr={round(j.new.corr(j.v17),3)}")


if __name__ == "__main__":
    if sys.argv[1] == "is":
        run_is()
    else:
        f, s, tp, sl = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        liq = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
        run_oos(f, s, tp, sl, liq)
