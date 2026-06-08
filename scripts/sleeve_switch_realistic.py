"""스위치 현실 구현 비교 — 저장된 일별시계열 재사용(백테X).
1) 자본이동(dyn): OFF=v16단독, ON=50:50  ← 이상치(별도계좌 매일 이체 필요, 비현실)
2) 고정가중 게이트: 슬리브계좌가 고ER날 플랫(현금), 가중 고정 ← 단일계좌+게이트로 구현가능
   gap = 이상치 대비 얼마나 살아남나."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

SLEEVE = ["LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]


def er(c, n=20):
    change = (c - c.shift(n)).abs()
    vol = c.diff().abs().rolling(n).sum()
    return change/vol.replace(0, np.nan)


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def stats(daily):
    eq = (1+daily).cumprod()
    mdd = ((eq/eq.cummax()-1)).min()*100
    sh = daily.mean()/daily.std()*np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1+r).prod()-1)*100)
    return eq.iloc[-1]*100, sh, mdd, yr


def main():
    df = pd.read_parquet("data/results/dyn_series.parquet")
    v, s = df["v16"], df["sl"]
    j = pd.concat([v, s], axis=1, keys=["v", "s"]).dropna()
    be = pd.concat([er(load_close(x), 20) for x in SLEEVE], axis=1).mean(axis=1).shift(1)
    if be.index.tz is not None:
        be.index = be.index.tz_localize(None)
    be = be.reindex(j.index, method="ffill")

    def show(tag, r):
        f, sh, mdd, yr = stats(r)
        ys = " ".join(f"{int(k)}:{round(v_):+}" for k, v_ in yr.items())
        print(f"  {tag:34} ${f:>10,.0f} Sh{sh:.2f} MDD{mdd:>4.0f}%  {ys}")

    print("=== baseline ===")
    show("v16 단독", j.v)
    show("항상 50:50", 0.5*j.v + 0.5*j.s)

    for q in [0.60, 0.75]:
        thr = be.quantile(q)
        on = be <= thr
        print(f"\n=== 임계 q{q:.2f} (ER<{thr:.2f}), ON {on.mean()*100:.0f}%일 ===")
        # 1) 자본이동 이상치
        show("[이상]자본이동 OFF=v16/ON=50:50", pd.Series(np.where(on, 0.5*j.v+0.5*j.s, j.v), index=j.index))
        # 2) 고정가중 게이트 (슬리브 OFF날 현금)
        show("[현실]고정50:50 슬리브OFF날현금", 0.5*j.v + 0.5*j.s.where(on, 0.0))
        show("[현실]고정75:25 슬리브OFF날현금", 0.75*j.v + 0.25*j.s.where(on, 0.0))
        # 3) 슬리브 단독 게이트 효과(슬리브계좌만 보기)
        show("  슬리브단독 always-on", j.s)
        show("  슬리브단독 게이트(OFF날현금)", j.s.where(on, 0.0))


if __name__ == "__main__":
    main()
