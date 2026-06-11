"""pyramid_ev_analysis.py — 엔진 수정 전 피라미딩 기대값 사전 분석 (데이터만 사용).

각 체결 거래(ml_dataset ⋈ entries)의 1h 경로를 따라가며:
  - +k·R 도달(트리거) 여부 → 도달 시 50% 증액 가정 (동일 TP/SL 유지, 정적 가격만 사용)
  - 추가분의 최종 R 결과를 집계 → 거래당 추가 EV(R)
보수적 처리: 같은 봉에서 트리거와 SL 동시 가능하면 SL 우선(트리거 무효).
look-ahead 없음: 진입 후 경로만 사용, 트리거/청산 모두 정적 가격.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

MAX_HOLD_H = 336

entries = pd.read_parquet("data/entries.parquet")
trades = pd.read_parquet("data/ml_dataset.parquet")
m = trades.merge(entries, on=["timestamp", "symbol", "strategy", "direction"], how="inner",
                 suffixes=("", "_e"))
print(f"체결 {len(trades)}건 중 조인 성공 {len(m)}건")

bars_cache: dict[str, pd.DataFrame] = {}
def get_bars(sym: str) -> pd.DataFrame:
    if sym not in bars_cache:
        df = pd.read_parquet(f"data/cache/ohlcv_{sym}_1h.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        bars_cache[sym] = df.set_index("timestamp").sort_index()
    return bars_cache[sym]


def simulate(row, k: float, add_frac: float, pessimistic: bool = False) -> dict:
    """진입 후 경로 추적. returns: {triggered, base_exit, extra_R}

    pessimistic=True: 같은 봉에 트리거+SL 동시 가능 시 '트리거 → SL' 순서로 가정
    (추가분 최대 손실 인정). False: SL 우선 = 트리거 무효 (낙관 바운드).
    """
    sym, d = row.symbol, row.direction
    e, tp, sl, R = row.entry_price, row.tp_price, row.sl_price, row.sl_dist
    long = d == "long"
    trig = e + k * R if long else e - k * R
    bars = get_bars(sym)
    t0 = pd.Timestamp(row.timestamp)
    path = bars.loc[bars.index > t0].head(MAX_HOLD_H)
    triggered = False
    for ts, b in path.iterrows():
        hi, lo = b["high"], b["low"]
        sl_hit = lo <= sl if long else hi >= sl
        tp_hit = hi >= tp if long else lo <= tp
        trig_hit = hi >= trig if long else lo <= trig
        if sl_hit:
            if pessimistic and trig_hit and not triggered:
                triggered = True  # 비관: 트리거 체결 후 SL
            extra = -add_frac * (k + 1.0) if triggered else 0.0
            return {"triggered": triggered, "exit": "sl", "extra_R": extra}
        if trig_hit and not triggered:
            triggered = True
        if tp_hit:
            tp_r = abs(tp - e) / R
            extra = add_frac * (tp_r - k) if triggered else 0.0
            return {"triggered": triggered, "exit": "tp", "extra_R": extra}
    # timeout
    if len(path) == 0:
        return {"triggered": False, "exit": "none", "extra_R": 0.0}
    last_close = path.iloc[-1]["close"]
    fin_r = (last_close - e) / R if long else (e - last_close) / R
    extra = add_frac * (fin_r - k) if triggered else 0.0
    return {"triggered": triggered, "exit": "timeout", "extra_R": extra}


for k in [0.5, 1.0, 1.5]:
    for pess in [False, True]:
        res = pd.DataFrame([simulate(r, k, 0.5, pess) for r in m.itertuples()])
        res["year"] = pd.to_datetime(m["timestamp"]).dt.year.clip(upper=2025).values
        trig = res[res.triggered]
        tag = "비관(트리거→SL)" if pess else "낙관(SL우선)"
        print(f"\n===== 트리거 +{k}R, 증액 50% [{tag}] =====")
        print(f"트리거 발생: {len(trig)}/{len(res)}건 ({len(trig)/len(res)*100:.0f}%)")
        print(f"트리거 후 결과: {trig['exit'].value_counts().to_dict()}")
        print(f"거래당 추가 EV: {res['extra_R'].mean():+.4f}R  (트리거 거래당 {trig['extra_R'].mean():+.4f}R)")
        print(res.groupby("year")["extra_R"].agg(["sum", "mean"]).round(3).to_string())
