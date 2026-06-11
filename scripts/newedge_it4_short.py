"""반복 4 — 알트 숏 수확기 검증: macross-1d short-only + 실제 펀딩 반영.

롱/숏 분해 진단에서 숏이 엔진으로 드러남 → short-only 분기.
펀딩: 보유기간 내 실제 8h rate 합산, 숏 = +Σrate (수취), 롱 = -Σrate.
출력: side(both/short) × 펀딩(on/off) × liq 조합의 IS/OOS 트레이드+포트폴리오.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import atr
from scripts.newedge_l2 import load_full, portfolio_sim, v17_daily, yearly
from scripts.newedge_it2_macross import macross_sig, HOLD
from scripts.newedge_it3_validate import trades_1d

CACHE = Path(__file__).parent.parent / "data" / "cache"


def load_funding(sym):
    f = CACHE / f"funding_{sym}_8h.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    return pd.Series(d.rate.values, index=d.timestamp.dt.floor("h").values)


def trade_table(dfs, f_, s_, tp_, sl_, side, liq, use_funding):
    rows = []
    for sym, df in dfs.items():
        sig = macross_sig(df, f_, s_, liq)
        if side == "short":
            sig = np.where(sig < 0, sig, 0)
        trs = trades_1d(df, sig, tp_, sl_)
        if not trs:
            continue
        fund = load_funding(sym) if use_funding else None
        ts = df.timestamp.values
        for t in trs:
            r = t["r1d"]
            if fund is not None:
                t0, t1 = ts[t["i"]], ts[t["xi"]]
                fr = fund[(fund.index > t0) & (fund.index <= t1)].sum()
                r += (fr if t["side"] < 0 else -fr)
            rows.append((ts[t["i"]], sym, t["side"], r))
    return pd.DataFrame(rows, columns=["t", "sym", "side", "r"])


def print_stats(tag, tdf):
    tdf = tdf.copy()
    tdf["y"] = pd.to_datetime(tdf.t).dt.year
    gw = tdf.r[tdf.r > 0].sum()
    gl = -tdf.r[tdf.r <= 0].sum()
    line = " ".join(
        f"{y}:{round(tdf[tdf.y==y].r.mean()*10000) if len(tdf[tdf.y==y]) else 0:>5}(n={len(tdf[tdf.y==y])})"
        for y in (2022, 2023, 2024, 2025, 2026))
    print(f"{tag}: n={len(tdf)} wr={round((tdf.r>0).mean()*100)} pf={round(gw/gl if gl>0 else 99,3)} "
          f"avg={round(tdf.r.mean()*10000)}bp | {line}")


def short_portfolio(dfs, f_, s_, tp_, sl_, liq, cost, fund_map):
    """portfolio_sim 재사용 위해 short-only sig 구성. 펀딩은 사후 일별 보정 불가 →
    트레이드 레벨에서 이미 검증했으므로 포트폴리오는 펀딩 제외(보수) 수치."""
    sigmap = {}
    for sym, df in dfs.items():
        sig = macross_sig(df, f_, s_, liq)
        sigmap[sym] = (df, np.where(sig < 0, sig, 0))
    eq, tdf, stats = portfolio_sim(sigmap, tp_, sl_, HOLD, risk_f=0.02, max_pos=8, cost_rt=cost)
    oos_eq = eq[eq.index >= "2025-01-01"]
    oos_ret = oos_eq.pct_change().dropna()
    oos_sh = oos_ret.mean() / oos_ret.std() * np.sqrt(365) if oos_ret.std() > 0 else 0
    oos_mdd = ((oos_eq - oos_eq.cummax()) / oos_eq.cummax()).min()
    v17 = v17_daily()
    mine = eq.pct_change().dropna()
    j = pd.concat([mine, v17.reindex(mine.index).fillna(0)], axis=1).dropna()
    j.columns = ["new", "v17"]
    print(f"[short-only 포트폴리오 cost={cost*10000:.0f}bp liq>{liq/1e6:.0f}M] FULL {stats} yearly={yearly(eq)}")
    print(f"  OOS: ret={round((oos_eq.iloc[-1]/oos_eq.iloc[0]-1)*100,1)}% sharpe={round(oos_sh,2)} "
          f"mdd={round(oos_mdd*100,1)}% v17corr={round(j.new.corr(j.v17),3)}")
    return eq


def main():
    f_, s_, tp_, sl_ = 20, 100, 6.0, 3.0
    dfs = load_full("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)

    print("== 트레이드 레벨: side × 펀딩 × liq ==")
    for side in ("both", "short"):
        for uf in (False, True):
            for liq in (0.0, 10e6):
                t = trade_table(dfs, f_, s_, tp_, sl_, side, liq, uf)
                print_stats(f"{side:5} fund={'Y' if uf else 'N'} liq>{liq/1e6:>2.0f}M", t)

    print("\n== short-only 포트폴리오 (펀딩 미반영=보수) ==")
    eq = short_portfolio(dfs, f_, s_, tp_, sl_, 0.0, 0.0010, None)
    short_portfolio(dfs, f_, s_, tp_, sl_, 10e6, 0.0010, None)
    short_portfolio(dfs, f_, s_, tp_, sl_, 10e6, 0.0030, None)
    eq.to_frame("eq").to_parquet(Path(__file__).parent.parent / "data" / "results" / "newedge_short_eq.parquet")

    # 쌍둥이 corroboration
    print("\n== 쌍둥이 10x100 short-only ==")
    t = trade_table(dfs, 10, 100, 6.0, 3.0, "short", 0.0, True)
    print_stats("10x100 fund=Y liq>0", t)
    short_portfolio(dfs, 10, 100, 6.0, 3.0, 0.0, 0.0010, None)


if __name__ == "__main__":
    main()
