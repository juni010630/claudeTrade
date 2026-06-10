"""D축 캐리 수익률 조사 — 현물 롱 + 선물 숏 델타중립 펀딩 수취 (데이터 분석만).

숏 perp 레그의 8h 펀딩 누적: rate>0이면 수취, rate<0이면 지불 (notional 대비, 1x 무레버).
변형 2개 고정(사전선언):
  A 상시: 항상 보유. 비용 = 초기 왕복 0.4% + 유지 0.1%/분기
  B 동적: 직전 정산 rate>0일 때만 보유(인과). 비용 = 전환마다 왕복 0.4%
왕복 0.4% = 현물 taker 10bps×2 + perp taker 5bps×2 + 슬리피지 ~10bps.
판단선: 연 10%+ 안정(연도별 양수)이면 "자본 생기면 구축 가치", 미만 폐기.

사용: python scripts/carry_yield_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT", "ARBUSDT", "FILUSDT",
           "LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
RT_COST = 0.004      # 왕복 (on→off 또는 off→on 한 사이클 = 진입+청산 합산)
HOLD_COST_Q = 0.001  # 상시 변형 분기당 유지비


def load(sym: str) -> pd.Series:
    d = pd.read_parquet(f"data/cache/funding_{sym}_8h.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    s = d.set_index("timestamp").sort_index()["rate"]
    return s[s.index >= "2022-01-01"]


def main():
    rows = []
    for sym in SYMBOLS:
        r = load(sym)
        if len(r) < 300:
            continue
        years = r.groupby(r.index.year)

        # A 상시: 연 gross = rate 합 (숏 수취), net = gross − 유지 0.4%/yr (분기 0.1%)
        a_yr = years.sum() * 100 - 0.4
        # B 동적: 직전 rate>0이면 다음 구간 보유
        on = (r.shift(1) > 0).astype(int)
        b_gross = (r * on).groupby(r.index.year).sum() * 100
        switches = (on.diff().abs().fillna(0)).groupby(r.index.year).sum()
        b_yr = b_gross - switches / 2 * RT_COST * 100  # on/off 두 번 전환 = 1 왕복
        on_pct = on.groupby(r.index.year).mean() * 100

        for y in a_yr.index:
            rows.append({"sym": sym, "year": int(y), "A_net%": round(a_yr[y], 2),
                         "B_net%": round(b_yr.get(y, 0), 2), "B_on%": round(on_pct.get(y, 0), 0),
                         "B_switch": int(switches.get(y, 0))})

    df = pd.DataFrame(rows)
    pv_a = df.pivot(index="sym", columns="year", values="A_net%")
    pv_b = df.pivot(index="sym", columns="year", values="B_net%")
    pd.set_option("display.width", 160)
    print("=== A 상시 캐리 net %/yr (숏 perp 펀딩, 비용 차감) ===")
    print(pv_a.round(1).to_string())
    print(f"\n  심볼 평균(연도별): {pv_a.mean().round(2).to_dict()}")
    print("\n=== B 동적(직전 rate>0만) net %/yr ===")
    print(pv_b.round(1).to_string())
    print(f"\n  심볼 평균(연도별): {pv_b.mean().round(2).to_dict()}")

    majors = pv_a.loc[["BTCUSDT", "ETHUSDT"]].mean()
    print(f"\n=== 판정 보조 ===")
    print(f"  메이저(BTC+ETH) A 상시 연도별: {majors.round(2).to_dict()}")
    print(f"  전심볼 A 평균 of 평균: {pv_a.mean().mean():.2f}%/yr | B: {pv_b.mean().mean():.2f}%/yr")
    print(f"  판단선: 연 10%+ & 전 연도 양수 → 구축 가치 / 미만 → 축 폐기")


if __name__ == "__main__":
    main()
