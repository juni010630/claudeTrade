"""지금 이 순간 각 심볼이 추세장인지 횡보장인지 실시간으로 판별해 출력.

백테스트 아님 — 현재 시점의 완성봉으로 국면을 '발견'만 한다.
데이터 소스: 바이낸스 퍼블릭 엔드포인트(인증 불필요) → 실패 시 로컬 캐시 폴백.

Usage:
    python3 scripts/regime_now.py                       # 기본 config 심볼, 1h
    python3 scripts/regime_now.py --tf 4h
    python3 scripts/regime_now.py --source cache         # 오프라인(캐시) 강제
    python3 scripts/regime_now.py --symbols BTCUSDT ETHUSDT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.cache import ParquetCache
from regime.realtime_switch import classify_regime

TF_MINS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _drop_forming(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """미완성(forming) 봉 제외 — look-ahead 방지. 완성봉이면 유지(30s 시계오차 여유)."""
    if df.empty:
        return df
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=TF_MINS.get(tf, 60))
    if last_close > pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=30):
        return df.iloc[:-1]
    return df


def fetch_live(symbols: list[str], tf: str, lookback: int) -> dict[str, pd.DataFrame]:
    """바이낸스 USDM 퍼블릭 OHLCV(실시간). 인증 불필요."""
    import ccxt

    ex = ccxt.binanceusdm({"enableRateLimit": True})
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        raw = ex.fetch_ohlcv(sym, tf, limit=lookback + 1)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        out[sym] = _drop_forming(df, tf)
    return out


def fetch_cache(symbols: list[str], tf: str, lookback: int) -> dict[str, pd.DataFrame]:
    """로컬 캐시 최근 봉(오프라인)."""
    cache = ParquetCache("data/cache")
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = cache.load(sym, tf)
        if df is None or df.empty:
            continue
        out[sym] = df.tail(lookback + 1).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 추세장/횡보장 스위치")
    parser.add_argument("--params", default="config/final_v17.yaml")
    parser.add_argument("--symbols", nargs="*", default=None, help="직접 심볼 지정")
    parser.add_argument("--tf", default="1h", help="판별 타임프레임 (1h/4h/1d)")
    parser.add_argument("--source", choices=["auto", "live", "cache"], default="auto")
    parser.add_argument("--lookback", type=int, default=120)
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    else:
        with open(args.params) as f:
            symbols = yaml.safe_load(f)["symbols"]

    # ── 데이터 확보 (실시간 우선, 실패 시 캐시) ──
    data: dict[str, pd.DataFrame] = {}
    used = args.source
    if args.source in ("auto", "live"):
        try:
            data = fetch_live(symbols, args.tf, args.lookback)
            used = "live(바이낸스 실시간)"
        except Exception as e:
            if args.source == "live":
                print(f"실시간 조회 실패: {e}")
                return
            print(f"[알림] 실시간 조회 실패 → 캐시 폴백 ({type(e).__name__})")
            data = {}
    if not data:
        data = fetch_cache(symbols, args.tf, args.lookback)
        used = "cache(로컬 최근봉)"

    if not data:
        print("데이터 없음 (실시간/캐시 모두 실패)")
        return

    # ── 판별 + 출력 ──
    now = pd.Timestamp.now(tz="UTC")
    print(f"\n=== 실시간 장 국면 스위치 | {args.tf} | source={used} | {now:%Y-%m-%d %H:%M}Z ===\n")
    header = f"{'심볼':<10} {'국면':<9} {'방향':<5} {'신뢰도':>5}  │ {'ADX':>6} {'Chop':>6} {'ER':>5} {'R²':>5} {'BBW%':>5}"
    print(header)
    print("─" * len(header))

    n_trend = n_range = n_neutral = 0
    for sym in symbols:
        df = data.get(sym)
        if df is None or len(df) < 40:
            print(f"{sym:<10} (데이터 부족)")
            continue
        v = classify_regime(df)
        vals = {vote.name: vote for vote in v.votes}

        def cell(name: str, fmt: str) -> str:
            vt = vals.get(name)
            if vt is None or pd.isna(vt.value):
                return f"{'--':>6}"
            mark = {1: "▲", -1: "▽", 0: "·"}[vt.vote]
            return f"{format(vt.value, fmt):>5}{mark}"

        bar = v.timestamp.strftime("%m-%d %H:%M") if v.timestamp is not None else "?"
        print(f"{sym:<10} {v.regime:<9} {v.direction:<5} {v.confidence*100:>4.0f}%  │ "
              f"{cell('ADX', '.1f')} {cell('Choppiness', '.1f')} "
              f"{cell('EfficiencyRatio', '.2f')} {cell('LinReg R²', '.2f')} "
              f"{cell('BB폭 백분위', '.2f')}   (봉 {bar})")

        if v.regime == "추세장":
            n_trend += 1
        elif v.regime == "횡보장":
            n_range += 1
        else:
            n_neutral += 1

    print("─" * len(header))
    print(f"종합: 추세장 {n_trend} · 횡보장 {n_range} · 전환/중립 {n_neutral}")
    print("범례: ▲=추세표 ▽=횡보표 ·=중립 | score≥+1.5→추세장, ≤-1.5→횡보장 (정설 임계값, 튜닝 안 함)\n")


if __name__ == "__main__":
    main()
