"""FNG(Fear&Greed) 기반 세 책 capital_fraction 동적 틸트 스케줄 빌더.

거래일 D의 가중 = FNG[D-lag_days] (기본 lag=1)에서만 산출 → look-ahead 구조적 차단.
백테(run_backtest)·라이브(live_trade) 양쪽이 동일 CSV·동일 함수를 호출 → 패리티 보장.

틸트(direction=+1, '탐욕→모멘텀'): g = z_lag * delta,  z_lag = clip((FNG-50)/50, -1, 1) 1일 래그.
  모멘텀 전략 frac ×(1+g),  평균회귀 전략 frac ×(1-g),  나머지 불변.  g ∈ [-delta, delta].
(2026-06-17 채택: 백테 스크린상 부호일관 방향. IS/OOS robustness는 미통과 — REGIME_TILT_RESULTS.md.)
"""
from __future__ import annotations

import pandas as pd


def build_fng_tilt_schedule(
    base_fractions: dict[str, float],
    fng_csv: str,
    delta: float,
    direction: int,
    momentum_strategies: list[str],
    meanrev_strategies: list[str],
    lag_days: int = 1,
) -> pd.DataFrame:
    """일별 시변 capital_fraction DataFrame (index=UTC 일자, columns=전략)."""
    fng = pd.read_csv(fng_csv, parse_dates=["date"])
    fng["date"] = fng["date"].dt.tz_localize("UTC")
    s = fng.set_index("date")["fng"].sort_index()
    z = ((s - 50.0) / 50.0).clip(-1.0, 1.0)
    g = direction * z.shift(lag_days) * delta   # 거래일 D는 D-lag 값 사용 (look-ahead 차단)

    mom, mr = set(momentum_strategies), set(meanrev_strategies)
    cols = {}
    for strat, base in base_fractions.items():
        if strat in mom:
            cols[strat] = base * (1.0 + g)
        elif strat in mr:
            cols[strat] = base * (1.0 - g)
        else:
            cols[strat] = pd.Series(base, index=g.index)
    return pd.DataFrame(cols)


def refresh_fng_csv(path: str = "data/regime/fng_daily.csv") -> bool:
    """alternative.me 최신 FNG를 CSV에 증분 병합 (라이브 일일 갱신용).
    실패 시 기존 CSV 유지하고 False 반환(조용히 — 네트워크 일시장애에 봇 중단 금지)."""
    import json
    import urllib.request
    from pathlib import Path

    try:
        with urllib.request.urlopen(
            "https://api.alternative.me/fng/?limit=90&format=json", timeout=15
        ) as resp:
            data = json.load(resp)["data"]
    except Exception:
        return False
    new = pd.DataFrame({
        "date": [pd.Timestamp(int(x["timestamp"]), unit="s", tz="UTC").strftime("%Y-%m-%d")
                 for x in data],
        "fng": [int(x["value"]) for x in data],
    })
    p = Path(path)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["date", "fng"])
    merged = (pd.concat([old, new]).drop_duplicates("date", keep="last")
              .sort_values("date").reset_index(drop=True))
    p.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(p, index=False)
    return True
