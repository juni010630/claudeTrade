"""펀딩 극단 역포지셔닝 — Phase 0 진단 (엔진 없음, 사전등록 게이트).

가설: 펀딩 극단 = 포지셔닝 과밀 = 청산 연료 → 군중 반대 방향 양의 기대수익.
  z <= -2 (숏 과밀, 숏이 지불) → 롱 신호 / z >= +2 (롱 과밀) → 숏 신호.

사전등록:
  신호 = 트레일링 90d(270정산) z-score, 주임계 ±2.0 (이벤트 = 첫 진입, |z|<1 복귀 후 재무장)
  주평가 = 방향성 3일 수익 (1h close 기준), 보조 1d/7d. ±1.5/±2.5는 단조성 확인용(선택 금지).
  G0 = IS(~2024) 3d 평균수익 >0 & t>=2  AND  IS 연도별 >=3/4 양수  AND  OOS(2025~) 양수
       AND 바스켓 합산 >=30 이벤트/년. 미달 → 종결.

재탕 구분: 기각된 것은 펀딩-필터(기존 진입 위)·funding3 시간차단. 트리거로서의 standalone은 미시도.
사용: python scripts/funding_extreme_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SYMS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT", "ARBUSDT", "FILUSDT",
        "LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
ZWIN = 270           # 90d × 3정산
Z_MAIN = 2.0
Z_ROBUST = [1.5, 2.5]
HORIZONS_H = {"1d": 24, "3d": 72, "7d": 168}
IS_END = "2024-12-31"


def load_events(sym: str, z_thr: float):
    f = pd.read_parquet(f"data/cache/funding_{sym}_8h.parquet")
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    if f["timestamp"].dt.tz is not None:
        f["timestamp"] = f["timestamp"].dt.tz_localize(None)
    f = f.sort_values("timestamp").set_index("timestamp")["rate"]
    z = (f - f.rolling(ZWIN, min_periods=ZWIN // 2).mean()) / \
        f.rolling(ZWIN, min_periods=ZWIN // 2).std().replace(0, np.nan)

    px = pd.read_parquet(f"data/cache/ohlcv_{sym}_1h.parquet")
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    if px["timestamp"].dt.tz is not None:
        px["timestamp"] = px["timestamp"].dt.tz_localize(None)
    close = px.sort_values("timestamp").set_index("timestamp")["close"]
    # 1h봉 open스탬프 t의 close는 t+1h 확정 → 정산시각 T에 쓸 수 있는 가격 = T-1h 봉 close
    avail = close.copy()
    avail.index = avail.index + pd.Timedelta(hours=1)

    events = []
    armed = True
    for t, zv in z.dropna().items():
        if armed and abs(zv) >= z_thr:
            direction = 1 if zv <= -z_thr else -1   # 숏과밀→롱 / 롱과밀→숏
            p0 = avail.asof(t)
            row = {"sym": sym, "t": t, "z": zv, "dir": direction}
            ok = pd.notna(p0)
            for tag, h in HORIZONS_H.items():
                p1 = avail.asof(t + pd.Timedelta(hours=h))
                row[tag] = direction * (p1 / p0 - 1) * 100 if ok and pd.notna(p1) else np.nan
            events.append(row)
            armed = False
        elif not armed and abs(zv) < 1.0:
            armed = True
    return pd.DataFrame(events)


def report(ev: pd.DataFrame, tag: str):
    if len(ev) == 0:
        print(f"  {tag}: 이벤트 0")
        return None
    r = ev["3d"].dropna()
    t = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
    wr = (r > 0).mean() * 100
    print(f"  {tag}: n={len(r)} | 3d 평균 {r.mean():+.2f}% 중앙 {r.median():+.2f}% "
          f"WR {wr:.0f}% t={t:.2f} | 1d {ev['1d'].mean():+.2f}% 7d {ev['7d'].mean():+.2f}%")
    return r.mean(), t


def main():
    all_ev = pd.concat([load_events(s, Z_MAIN) for s in SYMS], ignore_index=True)
    all_ev = all_ev.set_index("t").sort_index()
    is_ev, oos_ev = all_ev.loc[:IS_END], all_ev.loc["2025-01-01":]
    yrs = (all_ev.index[-1] - all_ev.index[0]).days / 365.25
    print(f"=== 펀딩 극단 이벤트 (|z|>={Z_MAIN}, 90d 트레일링) ===")
    print(f"총 {len(all_ev)}건 / {yrs:.1f}년 = {len(all_ev)/yrs:.0f}건/년 "
          f"(롱신호 {(all_ev['dir']==1).sum()} / 숏신호 {(all_ev['dir']==-1).sum()})")

    print("\n--- 전체/IS/OOS (방향성 수익, 비용 차감 전) ---")
    report(all_ev, "전체")
    is_stat = report(is_ev, "IS(~24)")
    oos_stat = report(oos_ev, "OOS(25~)")

    print("\n--- IS 연도별 3d 평균 ---")
    ywins = 0
    ymeans = is_ev.groupby(is_ev.index.year)["3d"].agg(["mean", "count"])
    for y, row in ymeans.iterrows():
        mark = "+" if row["mean"] > 0 else "−"
        print(f"  {y}: {row['mean']:+.2f}% ({int(row['count'])}건) {mark}")
    ywins = int((ymeans["mean"] > 0).sum())

    print("\n--- 방향별 / 단조성(임계 강화 시 효과 증가?) ---")
    for d, nm in ((1, "롱(숏과밀)"), (-1, "숏(롱과밀)")):
        sub = all_ev[all_ev["dir"] == d]["3d"].dropna()
        print(f"  {nm}: n={len(sub)} 평균 {sub.mean():+.2f}% WR {(sub>0).mean()*100:.0f}%")
    for zt in Z_ROBUST:
        e = pd.concat([load_events(s, zt) for s in SYMS], ignore_index=True)
        r = e["3d"].dropna()
        print(f"  |z|>={zt}: n={len(r)} 평균 {r.mean():+.2f}%")

    g_is = is_stat is not None and is_stat[0] > 0 and is_stat[1] >= 2.0
    g_yr = ywins >= min(3, len(ymeans))
    g_oos = oos_stat is not None and oos_stat[0] > 0
    g_freq = len(all_ev) / yrs >= 30
    print(f"\n[G0] IS 평균>0 & t>=2 → {'OK' if g_is else 'FAIL'} | 연도별 {ywins}/{len(ymeans)} → "
          f"{'OK' if g_yr else 'FAIL'} | OOS>0 → {'OK' if g_oos else 'FAIL'} | "
          f"빈도 {len(all_ev)/yrs:.0f}/년 → {'OK' if g_freq else 'FAIL'}")
    print(f"판정: {'통과 — 전략 구현 단계로' if all([g_is, g_yr, g_oos, g_freq]) else '기각 — 종결'}")
    print("(참고: 왕복 비용 ~15bps(maker 기준)는 평균수익에서 차감해 해석할 것)")


if __name__ == "__main__":
    main()
