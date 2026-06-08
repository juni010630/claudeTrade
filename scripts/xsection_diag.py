"""제3그룹 진단 — 크로스섹션 상대강도. 강한 알트가 지속(모멘텀)? 반전?
주간 리밸런스: 최근 N일 수익률 순위 → 상위k 롱 / 하위k 숏 (시장중립) → 다음주 성과."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# 4년 데이터 있는 메이저+중형 풀 (유동성 어느정도)
POOL = ["ETHUSDT", "BTCUSDT", "DOGEUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "AVAXUSDT",
        "LINKUSDT", "LTCUSDT", "DOTUSDT", "ATOMUSDT", "UNIUSDT", "NEARUSDT", "BCHUSDT",
        "FILUSDT", "ETCUSDT", "XLMUSDT", "AAVEUSDT"]


def load_close(s):
    d = pd.read_parquet(f"data/cache/ohlcv_{s}_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()["close"]


def main():
    px = pd.DataFrame({s: load_close(s) for s in POOL})
    px = px[px.index >= "2022-01-01"].resample("1D").last().ffill()
    wk = px.resample("W").last()
    wret = wk.pct_change()

    print("=== 크로스섹션: 최근 N주 수익률 순위 → 상위3 롱/하위3 숏, 다음주 성과 ===")
    print(f"{'룩백':>5} {'롱숏수익/주':>11} {'모멘텀?':>8} | (양수=모멘텀 강한게 지속, 음수=반전)")
    for lb in [1, 2, 4, 8]:
        mom = wk.pct_change(lb)  # 최근 lb주 수익률
        fwd = wret.shift(-1)     # 다음주 수익률
        ls_returns = []
        for t in mom.index:
            if t not in fwd.index: continue
            m = mom.loc[t].dropna(); f = fwd.loc[t]
            if len(m) < 8: continue
            top = m.nlargest(3).index; bot = m.nsmallest(3).index
            # 상위 롱 + 하위 숏 (시장중립)
            r = f[top].mean() - f[bot].mean()
            if not np.isnan(r): ls_returns.append(r)
        arr = np.array(ls_returns)
        sign = "모멘텀" if arr.mean() > 0 else "반전"
        print(f"{lb:>4}주 {arr.mean()*100:>+10.2f}% {sign:>8} | n={len(arr)} 승률{(arr>0).mean()*100:.0f}% "
              f"Sharpe(주){arr.mean()/arr.std()*np.sqrt(52):.2f}")


if __name__ == "__main__":
    main()
