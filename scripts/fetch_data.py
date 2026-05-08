"""히스토리 데이터 다운로드 및 캐시.

Usage:
    python scripts/fetch_data.py
    python scripts/fetch_data.py --params config/params.yaml
    python scripts/fetch_data.py --symbols BTCUSDT ETHUSDT --start 2022-01-01
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.cache import ParquetCache
from data.fetcher import FundingRateFetcher, OHLCVFetcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance 선물 데이터 다운로드")
    parser.add_argument("--params", default="config/final_v13_eth.yaml", help="파라미터 파일 경로")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--timeframes", nargs="+", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    data_cfg = p.get("data", {})

    symbols = args.symbols or p["symbols"]
    timeframes = args.timeframes or p["timeframes"]
    start_str = args.start or bt.get("start", "2024-01-01")
    end_str = args.end or bt.get("end")
    cache_dir = data_cfg.get("cache_dir", "data/cache")

    since = datetime.strptime(start_str, "%Y-%m-%d")
    until = datetime.strptime(end_str, "%Y-%m-%d") if end_str else None

    ohlcv_fetcher = OHLCVFetcher()
    funding_fetcher = FundingRateFetcher()
    cache = ParquetCache(cache_dir)

    for sym in symbols:
        for tf in timeframes:
            print(f"[OHLCV] {sym} {tf} 다운로드 중...")
            df = ohlcv_fetcher.fetch(sym, tf, since, until)
            if not df.empty:
                cache.save(df, sym, tf, data_type="ohlcv")
                print(f"  → {len(df)}봉 저장 완료")
            else:
                print(f"  → 데이터 없음")

        print(f"[펀딩비] {sym} 다운로드 중...")
        fd = funding_fetcher.fetch(sym, since, until)
        if not fd.empty:
            cache.save(fd, sym, "8h", data_type="funding")
            print(f"  → {len(fd)}건 저장 완료")

    print("\n완료!")


if __name__ == "__main__":
    main()
