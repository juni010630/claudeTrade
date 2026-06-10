"""OI/포지셔닝 일일 수집기 — 바이낸스 30일 한계를 자체 축적으로 우회 (미래 신호 연구용 씨앗).

수집 (퍼블릭, 인증 불필요, 1h 해상도, limit 500 ≈ 20일 → 일일 실행 시 중복 자동 제거):
  openInterestHist            sumOpenInterest/Value
  takerlongshortRatio         buySellRatio, buyVol, sellVol
  globalLongShortAccountRatio longShortRatio, longAccount, shortAccount

저장: data/oi_cache/{metric}_{symbol}_1h.parquet (append + timestamp dedup).
edge-monitor.service의 ExecStartPost로 매일 00:30 UTC 실행. 어떤 오류든 exit 0 (모니터 실패 방지).
연구 사용 가능 시점: 표본 1년+ 축적 후 (project_market_neutral_rejected의 데이터 벽 항목 참조).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fapi.binance.com/futures/data"
SYMS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "ADAUSDT", "ARBUSDT", "FILUSDT",
        "LTCUSDT", "UNIUSDT", "STORJUSDT", "ARPAUSDT", "BANDUSDT"]
METRICS = {
    "oi": "openInterestHist",
    "taker": "takerlongshortRatio",
    "lsacct": "globalLongShortAccountRatio",
}
OUT = Path("data/oi_cache")


def fetch(endpoint: str, sym: str) -> pd.DataFrame | None:
    r = requests.get(f"{BASE}/{endpoint}",
                     params={"symbol": sym, "period": "1h", "limit": 500}, timeout=15)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list) or not rows:
        return None
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    for key, ep in METRICS.items():
        for sym in SYMS:
            try:
                df = fetch(ep, sym)
                if df is None:
                    n_fail += 1
                    continue
                path = OUT / f"{key}_{sym}_1h.parquet"
                if path.exists():
                    old = pd.read_parquet(path)
                    df = pd.concat([old, df], ignore_index=True)
                df = (df.drop_duplicates(subset="timestamp", keep="last")
                        .sort_values("timestamp").reset_index(drop=True))
                df.to_parquet(path, index=False)
                n_ok += 1
                time.sleep(0.2)  # rate limit 여유
            except Exception as e:
                n_fail += 1
                print(f"[oi_collector] {key}/{sym} 실패: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[oi_collector] 완료: {n_ok} 저장 / {n_fail} 실패")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # 어떤 경우에도 edge-monitor 서비스를 실패시키지 않음
        print(f"[oi_collector] 전체 실패: {e}", file=sys.stderr)
    sys.exit(0)
