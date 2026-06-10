"""A2 maker-first 진입 타당성 연구 (분석만, 라이브 변경 없음).

가설: 시그널 봉 close에 post-only 지정가 → N분 내 미체결 시 시장가 폴백.
체결 시 절감 = (taker 5 − maker 2 = 3bps) + 실측 슬리피지 S 회피.
미체결 시 추가비용 = N분 동안의 불리한 가격 드리프트(폴백 시장가가 더 비싼 만큼).

방법: v17 전구간 805거래 중 추세 6심볼 거래 전수. 5m 봉으로
  long 체결 = 윈도 내 5m low가 limit 터치(낙관) / strict 미만(보수)
  미체결 드리프트 = 윈도 끝 5m close vs limit (bps, +=불리)
역선택 진단: 체결군 vs 미체결군의 실현 PnL(notional 대비 bps) 비교.

한계(명시): 터치=체결은 큐 위치 미반영 낙관 가정. 보수 변형 병기.
입력: data/results/v17_trades_full.csv (scripts/cost_breakdown.py 산출)

사용: python scripts/maker_entry_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

TRADES = "data/results/v17_trades_full.csv"
SYMS = ["ETHUSDT", "BTCUSDT", "DOGEUSDT", "ARBUSDT", "FILUSDT", "ADAUSDT"]
WINDOWS_MIN = [5, 15, 30]
S_LIVE_BPS = [25.0, 35.0]  # 실측 진입 슬리피지 시나리오 (edge_monitor +23~36bps)
FEE_SAVE_BPS = 3.0         # taker 5 → maker 2


def load_5m(sym: str) -> pd.DataFrame:
    d = pd.read_parquet(f"data/cache/ohlcv_{sym}_5m.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    return d.sort_values("timestamp").reset_index(drop=True)


def main():
    t = pd.read_csv(TRADES, parse_dates=["entry_time"])
    if t["entry_time"].dt.tz is not None:
        t["entry_time"] = t["entry_time"].dt.tz_localize(None)
    t = t[t["strategy"].isin(["ema_cross", "multi_tf_breakout"]) & t["symbol"].isin(SYMS)].copy()
    print(f"추세 거래 {len(t)}건 (6심볼) | notional 합 ${t['size_usd'].sum():,.0f}")

    # 진입 시점 의미 검증: entry_time 봉의 close == entry_price 인지 (= 체결시각 entry_time+1h)
    h1 = pd.read_parquet("data/cache/ohlcv_ETHUSDT_1h.parquet")
    h1["timestamp"] = pd.to_datetime(h1["timestamp"])
    if h1["timestamp"].dt.tz is not None:
        h1["timestamp"] = h1["timestamp"].dt.tz_localize(None)
    h1 = h1.set_index("timestamp")
    sample = t[t["symbol"] == "ETHUSDT"].head(20)
    match = sum(1 for _, r in sample.iterrows()
                if (r["entry_time"] - pd.Timedelta(hours=1)) in h1.index
                and abs(h1.loc[r["entry_time"] - pd.Timedelta(hours=1), "close"] - r["entry_price"])
                / r["entry_price"] < 1e-6)
    assert match >= 18, f"entry_time 의미 검증 실패 ({match}/20) — 윈도 시작점 재확인 필요"
    print(f"[검증] close(entry_time−1h)==entry_price {match}/20 → entry_time = 체결시각 (윈도 시작점)")

    fivem = {s: load_5m(s) for s in SYMS}
    arr = {s: (d["timestamp"].to_numpy(), d["low"].to_numpy(), d["high"].to_numpy(),
               d["close"].to_numpy()) for s, d in fivem.items()}

    rows = []
    for _, r in t.iterrows():
        ts, lo, hi, cl = arr[r["symbol"]]
        t0 = np.datetime64(r["entry_time"])
        i0 = np.searchsorted(ts, t0)  # 첫 5m 봉 = 체결시각부터
        limit = r["entry_price"]
        for n in WINDOWS_MIN:
            i1 = i0 + n // 5
            if i1 > len(ts):
                continue
            w_lo, w_hi = lo[i0:i1], hi[i0:i1]
            if len(w_lo) == 0:
                continue
            if r["direction"] == "long":
                filled_t = bool((w_lo <= limit).any())
                filled_s = bool((w_lo < limit).any())
                drift = (cl[min(i1 - 1, len(cl) - 1)] - limit) / limit * 1e4
            else:
                filled_t = bool((w_hi >= limit).any())
                filled_s = bool((w_hi > limit).any())
                drift = (limit - cl[min(i1 - 1, len(cl) - 1)]) / limit * 1e4
            rows.append({"n": n, "strategy": r["strategy"], "size": r["size_usd"],
                         "ret_bps": r["pnl"] / r["size_usd"] * 1e4,
                         "filled_t": filled_t, "filled_s": filled_s,
                         "drift_bps": drift})
    df = pd.DataFrame(rows)

    for n in WINDOWS_MIN:
        g = df[df["n"] == n]
        for mode, col in (("터치(낙관)", "filled_t"), ("strict(보수)", "filled_s")):
            f, u = g[g[col]], g[~g[col]]
            fill_rate = len(f) / len(g) * 100
            adv = f["ret_bps"].mean() - u["ret_bps"].mean() if len(u) else 0.0
            # 노셔널 가중 절감: 체결 절감 − 미체결 드리프트 (드리프트<0이면 폴백이 오히려 유리)
            line = [f"N={n:>2}m {mode:10} 체결률 {fill_rate:5.1f}%"]
            for S in S_LIVE_BPS:
                save = (f["size"] * (S + FEE_SAVE_BPS)).sum()
                net_bps = (save - (u["size"] * u["drift_bps"]).sum()) / g["size"].sum()
                line.append(f"S={S:.0f}: net {net_bps:+6.1f}bps/노셔널")
            line.append(f"| 역선택(체결−미체결 PnL) {adv:+7.1f}bps")
            print("  " + "  ".join(line))
        f = g[g["filled_t"]]
        u = g[~g["filled_t"]]
        if len(u):
            print(f"        미체결군 평균 드리프트 {u['drift_bps'].mean():+.1f}bps,"
                  f" 미체결군 평균 PnL {u['ret_bps'].mean():+.0f}bps vs 체결군 {f['ret_bps'].mean():+.0f}bps,"
                  f" 전략별 체결률: " + ", ".join(
                      f"{s}: {g[g['strategy'] == s]['filled_t'].mean()*100:.0f}%"
                      for s in g["strategy"].unique()))


if __name__ == "__main__":
    main()
