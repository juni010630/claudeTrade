"""macross_d 엔진 배선 패리티 체크 — 엔진 vs 격리 시뮬.

1) 엔진: config/newedge_parity_solo.yaml로 run_fill_dump → 체결 진입 + 일별 equity
2) 시뮬: 동일 유니버스/필터로 newedge 격리 시뮬 (신호·포트폴리오)
3) 대조: 신호 일치율(심볼+날짜+방향), 거래수/수익/연도별, equity 통계
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

RES = Path(__file__).parent.parent / "data" / "results"


def run_engine():
    with open("config/newedge_parity_solo.yaml") as f:
        p = yaml.safe_load(f)
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p["primary_timeframe"], cache_dir="data/cache",
        lookback=p["data"]["lookback_bars"],
    )
    engine = build_engine(p, p["backtest"]["initial_capital"], abort_mdd=None)
    print("strategies:", [s.name for s in engine.strategies])
    snaps = loader.iterate(since=pd.Timestamp(p["backtest"]["start"], tz="UTC"),
                           until=pd.Timestamp(p["backtest"]["end"], tz="UTC"))
    report, fills = engine.run_fill_dump(snaps)
    report.print_summary()
    fills.to_parquet(RES / "parity_engine_fills.parquet")
    eq = engine.equity_curve.to_series().resample("1D").last()
    eq.to_frame("eq").to_parquet(RES / "parity_engine_eq.parquet")
    print(f"engine fills={len(fills)} saved")


def run_sim():
    from scripts.newedge_l0 import atr
    from scripts.newedge_l2 import load_full, portfolio_sim
    from scripts.newedge_it2_macross import macross_sig

    with open("config/newedge_parity_solo.yaml") as f:
        p = yaml.safe_load(f)
    alts = set(p["strategies"]["macross_d"]["symbols"])
    dfs = {s: d for s, d in load_full("1d").items() if s in alts}
    for df in dfs.values():
        df["__atr"] = atr(df, 14)
    sigmap = {sym: (df, macross_sig(df, 20, 100, 10e6)) for sym, df in dfs.items()}
    # 신호 이벤트 (슬롯 무관 raw)
    ev = []
    for sym, (df, sig) in sigmap.items():
        ts = df.timestamp.values
        for i in np.nonzero(sig)[0]:
            ev.append((pd.Timestamp(ts[i]), sym, "long" if sig[i] > 0 else "short"))
    pd.DataFrame(ev, columns=["t", "symbol", "direction"]).to_parquet(RES / "parity_sim_signals.parquet")
    eq, tdf, stats = portfolio_sim(sigmap, 6.0, 3.0, 60, risk_f=0.02, max_pos=8, cost_rt=0.0010)
    eq.to_frame("eq").to_parquet(RES / "parity_sim_eq.parquet")
    tdf.to_csv(RES / "parity_sim_trades.csv", index=False)
    print(f"sim signals={len(ev)} trades={stats}")


def compare():
    fills = pd.read_parquet(RES / "parity_engine_fills.parquet")
    sims = pd.read_parquet(RES / "parity_sim_signals.parquet")
    # 엔진 진입 = 00:00 스냅샷 → 시뮬 신호일(1d봉 timestamp)과 동일 날짜
    f = fills.copy()
    f["d"] = pd.to_datetime(f.timestamp).dt.floor("D").dt.tz_localize(None)
    s = sims.copy()
    s["d"] = pd.to_datetime(s.t).dt.floor("D").dt.tz_localize(None)
    fk = set(zip(f.d, f.symbol, f.direction))
    sk = set(zip(s.d, s.symbol, s.direction))
    inter = fk & sk
    print(f"\n== 신호/체결 대조 ==")
    print(f"엔진 체결 진입: {len(fk)} | 시뮬 raw 신호: {len(sk)}")
    print(f"엔진 진입 중 시뮬 신호와 일치: {len(inter)}/{len(fk)} ({len(inter)/max(len(fk),1)*100:.1f}%)")
    only_engine = sorted(fk - sk)[:8]
    print(f"엔진 단독(시뮬에 없음) 예시: {only_engine[:4]}")

    eq_e = pd.read_parquet(RES / "parity_engine_eq.parquet")["eq"]
    eq_s = pd.read_parquet(RES / "parity_sim_eq.parquet")["eq"]
    for name, eq in (("engine", eq_e), ("sim", eq_s)):
        eq.index = pd.DatetimeIndex(eq.index)
        ret = eq.pct_change().dropna()
        sh = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0
        mdd = ((eq - eq.cummax()) / eq.cummax()).min()
        yr = {}
        for y in (2022, 2023, 2024, 2025, 2026):
            sub = eq[(eq.index >= f"{y}-01-01") & (eq.index <= f"{y}-12-31")]
            if len(sub) > 30:
                yr[y] = round((sub.iloc[-1] / sub.iloc[0] - 1) * 100)
        print(f"{name:>7}: x{round(float(eq.iloc[-1])/float(eq.iloc[0]),2)} Sh={round(float(sh),2)} "
              f"MDD={round(float(mdd)*100,1)}% yr={yr}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("engine", "all"):
        run_engine()
    if mode in ("sim", "all"):
        run_sim()
    if mode in ("compare", "all"):
        compare()
