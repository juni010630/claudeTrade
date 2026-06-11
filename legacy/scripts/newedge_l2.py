"""새 엣지 greedy search — 레벨 2: 포트폴리오 시뮬 + OOS 개봉 + 비용 스트레스 + v17 상관.

레벨 1에서 확정된 설정만 받는다 (여기서 파라미터 추가 탐색 금지 — walk-forward 원칙).
포트폴리오 시뮬: equity 복리, 슬롯 캡, 신호당 risk fraction, 일별 MTM equity curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from scripts.newedge_l0 import atr
from scripts.newedge_l1 import f3_squeeze_sym

CACHE = Path(__file__).parent.parent / "data" / "cache"
FULL_END = "2026-04-23"


def load_full(tf="1d", start_cut="2022-01-01"):
    import os
    out = {}
    for f in sorted(os.listdir(CACHE)):
        if not (f.startswith("ohlcv_") and f.endswith(f"_{tf}.parquet")):
            continue
        sym = f[6:-(len(tf) + 9)]
        df = pd.read_parquet(CACHE / f)
        df = df[df.timestamp <= FULL_END]
        if len(df) < 200 or str(df.timestamp.iloc[0])[:10] > start_cut:
            continue
        out[sym] = df.reset_index(drop=True)
    return out


def collect_signals(dfs, cfg):
    """전 심볼 신호·가격 테이블 생성. 반환: {sym: (df, sig array)}"""
    out = {}
    for sym, df in dfs.items():
        sig = f3_squeeze_sym(df, **cfg)
        out[sym] = (df, sig)
    return out


def portfolio_sim(sigmap, tp_mult, sl_mult, max_hold, *, risk_f=0.02, max_pos=8,
                  cost_rt=0.0010, start="2022-01-01", end="2026-04-23"):
    """일별 이벤트 루프: 신호일 시가 진입(슬롯 여유 시), ATR 고정 TP/SL(SL우선)+보유캡.
    사이징: notional = equity * risk_f / (sl_dist/entry)  (SL 도달 시 equity의 risk_f 손실).
    일별 MTM equity curve → Sharpe/MDD. 비용 cost_rt 왕복, 진입 시 절반/청산 시 절반."""
    # 날짜축
    all_dates = sorted(set().union(*[set(df.timestamp.dt.floor("D")) for df, _ in sigmap.values()]))
    all_dates = [d for d in all_dates if str(d)[:10] >= start and str(d)[:10] <= end]
    date_ix = {d: k for k, d in enumerate(all_dates)}
    # 심볼별 날짜→행 인덱스
    arrs = {}
    for sym, (df, sig) in sigmap.items():
        ix = {}
        for i, t in enumerate(df.timestamp.dt.floor("D")):
            ix[t] = i
        arrs[sym] = (df.open.values, df.high.values, df.low.values, df.close.values,
                     df["__atr"].values, sig, ix, df.timestamp.values)

    equity = 1.0
    eq_curve = []
    open_pos = {}  # sym -> dict(side, e, tp, sl, qty_frac(notional/equity at entry as $), entry_i, entry_d)
    trades = []
    for d in all_dates:
        # 1) 기존 포지션 청산 체크 (당일 봉)
        for sym in list(open_pos.keys()):
            o, h, l, c, a, sig, ix, ts = arrs[sym]
            if d not in ix:
                continue
            i = ix[d]
            p = open_pos[sym]
            s = p["side"]
            r = None
            if s > 0:
                if l[i] <= p["sl"]:
                    r = (p["sl"] - p["e"]) / p["e"]
                elif h[i] >= p["tp"]:
                    r = (p["tp"] - p["e"]) / p["e"]
            else:
                if h[i] >= p["sl"]:
                    r = (p["e"] - p["sl"]) / p["e"]
                elif l[i] <= p["tp"]:
                    r = (p["e"] - p["tp"]) / p["e"]
            if r is None and (date_ix[d] - p["entry_dx"]) >= max_hold:
                r = (c[i] - p["e"]) / p["e"] * s
            if r is not None:
                pnl = p["notional"] * (r - cost_rt)
                equity += pnl
                trades.append((d, sym, s, r - cost_rt, pnl))
                del open_pos[sym]
        # 2) 신규 진입 (당일 시가, t-1 신호)
        for sym, (o, h, l, c, a, sig, ix, ts) in arrs.items():
            if sym in open_pos or len(open_pos) >= max_pos:
                continue
            if d not in ix:
                continue
            i = ix[d]
            s = int(sig[i])
            if s == 0 or i < 1 or not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
                continue
            e = o[i]
            sl_dist = sl_mult * a[i - 1] / e
            if sl_dist <= 0:
                continue
            notional = equity * risk_f / sl_dist
            notional = min(notional, equity * 3.0)  # 노셔널 캡 (저변동 폭주 방지)
            open_pos[sym] = dict(side=s, e=e, tp=e + s * tp_mult * a[i - 1],
                                 sl=e - s * sl_mult * a[i - 1], notional=notional,
                                 entry_dx=date_ix[d])
            # 당일 진입 즉시 청산 체크 (진입봉 내 TP/SL)
            p = open_pos[sym]
            r = None
            if s > 0:
                if l[i] <= p["sl"]:
                    r = (p["sl"] - e) / e
                elif h[i] >= p["tp"]:
                    r = (p["tp"] - e) / e
            else:
                if h[i] >= p["sl"]:
                    r = (e - p["sl"]) / e
                elif l[i] <= p["tp"]:
                    r = (e - p["tp"]) / e
            if r is not None:
                pnl = p["notional"] * (r - cost_rt)
                equity += pnl
                trades.append((d, sym, s, r - cost_rt, pnl))
                del open_pos[sym]
        # 3) 일별 MTM
        mtm = equity
        for sym, p in open_pos.items():
            o, h, l, c, a, sig, ix, ts = arrs[sym]
            if d in ix:
                p["last_c"] = c[ix[d]]
            mtm += p["notional"] * ((p.get("last_c", p["e"]) - p["e"]) / p["e"] * p["side"])
        eq_curve.append((d, mtm))
        if mtm <= 0:
            break

    eq = pd.Series(dict(eq_curve)).sort_index()
    ret = eq.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    tdf = pd.DataFrame(trades, columns=["d", "sym", "side", "r", "pnl"])
    return eq, tdf, dict(final=round(eq.iloc[-1], 3), sharpe=round(sharpe, 2),
                         mdd=round(mdd * 100, 1), n=len(tdf),
                         wr=round((tdf.r > 0).mean() * 100) if len(tdf) else 0)


def yearly(eq):
    out = {}
    for y in [2022, 2023, 2024, 2025, 2026]:
        sub = eq[(eq.index >= f"{y}-01-01") & (eq.index <= f"{y}-12-31")]
        if len(sub) < 30:
            continue
        out[y] = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100)
    return out


def v17_daily():
    t = pd.read_csv(Path(__file__).parent.parent / "data" / "results" / "v17_trades_full.csv",
                    parse_dates=["exit_time"])
    pnl = t.groupby(t.exit_time.dt.floor("D")).pnl.sum()
    return pnl


def report(name, sigmap, cfg, cost_rt, save=False):
    eq, tdf, stats = portfolio_sim(sigmap, cfg["tp"], cfg["sl"], cfg["hold"],
                                   risk_f=0.02, max_pos=8, cost_rt=cost_rt)
    is_eq = eq[eq.index <= "2024-12-31"]
    is_ret = is_eq.pct_change().dropna()
    is_sh = is_ret.mean() / is_ret.std() * np.sqrt(365) if is_ret.std() > 0 else 0
    is_mdd = ((is_eq - is_eq.cummax()) / is_eq.cummax()).min()
    oos_eq = eq[eq.index >= "2025-01-01"]
    oos_ret = oos_eq.pct_change().dropna()
    oos_sh = oos_ret.mean() / oos_ret.std() * np.sqrt(365) if oos_ret.std() > 0 else 0
    oos_mdd = ((oos_eq - oos_eq.cummax()) / oos_eq.cummax()).min()
    print(f"\n[{name}] cost={cost_rt*10000:.0f}bp  FULL: {stats}  yearly={yearly(eq)}")
    print(f"  IS : x{round(is_eq.iloc[-1],2)} sharpe={round(is_sh,2)} mdd={round(is_mdd*100,1)}%")
    print(f"  OOS: ret={round((oos_eq.iloc[-1]/oos_eq.iloc[0]-1)*100,1)}% "
          f"sharpe={round(oos_sh,2)} mdd={round(oos_mdd*100,1)}%")
    v17 = v17_daily()
    mine = eq.pct_change().dropna()
    j = pd.concat([mine, v17.reindex(mine.index).fillna(0)], axis=1).dropna()
    j.columns = ["new", "v17"]
    act = j[(j.new != 0) & (j.v17 != 0)]
    print(f"  v17 일별상관: all={round(j.new.corr(j.v17),3)} active={round(act.new.corr(act.v17),3)} (n={len(act)})")
    if save:
        rdir = Path(__file__).parent.parent / "data" / "results"
        eq.to_frame("eq").to_parquet(rdir / "newedge_f3_eq.parquet")
        tdf.to_csv(rdir / "newedge_f3_trades.csv", index=False)
    return eq


def main():
    dfs = load_full("1d")
    for df in dfs.values():
        df["__atr"] = atr(df, 14)

    greedy = dict(pct_th=0.3, N=10, tp=5.0, sl=2.5, hold=30, win=100, liq_min=10e6)
    conserv = dict(pct_th=0.3, N=20, tp=4.0, sl=2.0, hold=30, win=100, liq_min=0.0)

    sig_g = collect_signals(dfs, greedy)
    sig_c = collect_signals(dfs, conserv)
    print(f"universe={len(dfs)}  greedy={greedy}\nconserv={conserv}")

    report("greedy", sig_g, greedy, 0.0010, save=True)
    report("conserv(L0원형)", sig_c, conserv, 0.0010)
    # 비용 스트레스 (greedy)
    report("greedy", sig_g, greedy, 0.0030)
    report("greedy", sig_g, greedy, 0.0050)

if __name__ == "__main__":
    main()
