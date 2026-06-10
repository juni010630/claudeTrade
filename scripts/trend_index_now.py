"""TRI (Trend-Range Index) 현재 수치 — 시장이 지금 횡보장인지 추세장인지 0~100으로.

산출: 시장 TRI(BTC+ETH 평균) + 개별 + 성분 분해 + 최근 추이 스파크라인 + 기존 투표기 병기.
검증: TREND_INDEX_RESULTS.md (OOS AUC 0.75, 플립율 베이스라인 대비 -39%). 정보용 — 봇 배선 없음.

Usage:
    python3 scripts/trend_index_now.py                  # 실시간(실패 시 캐시 폴백)
    python3 scripts/trend_index_now.py --source cache   # 오프라인
    python3 scripts/trend_index_now.py --days 60        # 추이 60일
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from data.cache import ParquetCache
from regime.realtime_switch import classify_regime
from regime.trend_index import DEFAULT_SELECTED, THR_RANGE, THR_TREND, \
    compute_components, normalize, trend_index, label_series

SYMS = ["BTCUSDT", "ETHUSDT"]
TF_MINS = {"1h": 60, "4h": 240, "1d": 1440}
BLOCKS = "▁▂▃▄▅▆▇█"


def _drop_forming(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=TF_MINS[tf])
    if last_close > pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=30):
        return df.iloc[:-1]
    return df


def fetch_live(tf: str, lookback: int) -> dict[str, pd.DataFrame]:
    import ccxt
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    out = {}
    for sym in SYMS:
        raw = ex.fetch_ohlcv(sym, tf, limit=lookback + 1)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        out[sym] = _drop_forming(df, tf)
    return out


def fetch_cache(tf: str, lookback: int) -> dict[str, pd.DataFrame]:
    cache = ParquetCache("data/cache")
    out = {}
    for sym in SYMS:
        df = cache.load(sym, tf)
        if df is not None and not df.empty:
            out[sym] = df.tail(lookback + 1).reset_index(drop=True)
    return out


def _tz_naive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    return df


def spark(s: pd.Series) -> str:
    return "".join(BLOCKS[min(7, max(0, int(v / 12.51)))] if pd.notna(v) else " " for v in s)


def main() -> None:
    ap = argparse.ArgumentParser(description="TRI — 횡보/추세 지수 (0~100)")
    ap.add_argument("--source", choices=["auto", "live", "cache"], default="auto")
    ap.add_argument("--days", type=int, default=30, help="추이 표시 일수")
    args = ap.parse_args()

    data = {}
    used = args.source
    if args.source in ("auto", "live"):
        try:
            data = {"1d": fetch_live("1d", 400), "4h": fetch_live("4h", 700)}
            used = "live(바이낸스)"
        except Exception as e:
            if args.source == "live":
                print(f"실시간 조회 실패: {e}")
                return
            print(f"[알림] 실시간 실패 → 캐시 폴백 ({type(e).__name__})")
    if not data:
        data = {"1d": fetch_cache("1d", 400), "4h": fetch_cache("4h", 700)}
        used = "cache(로컬)"

    tris, comps = {}, {}
    for sym in SYMS:
        d1, d4 = _tz_naive(data["1d"][sym]), _tz_naive(data["4h"][sym])
        tris[sym] = trend_index(d1, d4)["tri"]
        comps[sym] = normalize(compute_components(d1.set_index("timestamp")))[DEFAULT_SELECTED].iloc[-1] * 100

    mkt = (tris[SYMS[0]] + tris[SYMS[1]]) / 2
    label = label_series(mkt).iloc[-1]
    now = pd.Timestamp.now(tz="UTC")

    print(f"\n=== TRI 횡보/추세 지수 | source={used} | {now:%Y-%m-%d %H:%M}Z ===\n")
    cur = mkt.dropna().iloc[-1]
    print(f"  시장 TRI : {cur:5.1f} / 100  →  {label}"
          f"   (≥{THR_TREND:.0f} 추세 / ≤{THR_RANGE:.0f} 횡보 / 사이=직전 유지)")
    for sym in SYMS:
        t = tris[sym].dropna()
        print(f"  {sym:8}: {t.iloc[-1]:5.1f}   ({t.index[-1]:%m-%d} 봉)")
    print(f"\n  최근 {args.days}일 추이 (시장): {spark(mkt.dropna().tail(args.days))}"
          f"  [{mkt.dropna().tail(args.days).min():.0f}~{mkt.dropna().tail(args.days).max():.0f}]")

    print("\n  성분 (0~100, 추세=高):")
    for c in DEFAULT_SELECTED:
        vals = " ".join(f"{s[:3]} {comps[s][c]:5.1f}" for s in SYMS)
        print(f"    {c:8} {vals}")

    print("\n  참고(기존 5지표 투표기):")
    for sym in SYMS:
        v = classify_regime(_tz_naive(data["1d"][sym]).tail(120))
        print(f"    {sym:8} {v.summary()}")
    print()


if __name__ == "__main__":
    main()
