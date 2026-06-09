"""슬리브 라이브 dry-run 검증 (계좌 무접촉 — 공개 피드만).

UTC 0시(슬리브 발동시각)에 실제 라이브 스냅샷을 받아:
 1. 슬리브 1d 데이터 헬스 (마지막 봉이 '전일 완성봉'인가 = finding D)
 2. 라이브 1d vs 캐시 1d 패리티 (겹치는 봉 OHLC 일치?)
 3. mean_reversion.generate_signals 실제 호출 → 지금 무슨 신호를 내는가
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.cache import ParquetCache
from data.live_feed import LiveFeed
from indicators.momentum import rsi as calc_rsi
from regime.detector import RegimeDetector
from strategies.mean_reversion import MeanReversionStrategy

CONFIG = "config/merged_v16_sleeve.yaml"


def main() -> None:
    p = yaml.safe_load(open(CONFIG))
    symbols, tfs = p["symbols"], p["timeframes"]
    print(f"공개 피드 스냅샷 페치 중 ({len(symbols)}심볼 × {tfs})...")
    feed = LiveFeed(symbols=symbols, timeframes=tfs, primary_tf="1h", lookback=300, demo=False)
    snap = feed.snapshot_now()

    print(f"\nsnap_ts = {snap.timestamp}  | hour = {snap.timestamp.hour}")
    fires = snap.timestamp.hour == 0
    print(f"슬리브(1d) 발동조건 hour==0: {'✅ 지금 발동' if fires else '❌ 오늘 미발동'}")

    mr = MeanReversionStrategy(dict(p["strategies"]["mean_reversion"]))
    cache = ParquetCache("data/cache")

    print("\n[1] 슬리브 1d 데이터 헬스 (last_1d_open = 전일이어야 정상)")
    print(f"{'sym':10}{'#bars':>6}{'last_1d_open':>22}{'close':>12}{'RSI14':>8}  신호?")
    for sym in mr.symbols:
        df = snap.bars.get(sym, {}).get("1d")
        if df is None or df.empty:
            print(f"{sym:10}  ⚠️ 1d 데이터 없음"); continue
        last_ts = pd.to_datetime(df["timestamp"].iloc[-1])
        close = float(df["close"].iloc[-1])
        rsi = float(calc_rsi(df, mr.rsi_period).iloc[-1])
        sig = "LONG" if rsi <= mr.rsi_oversold else ("SHORT" if rsi >= mr.rsi_overbought else "-")
        print(f"{sym:10}{len(df):>6}{str(last_ts):>22}{close:>12.4f}{rsi:>8.1f}  {sig}")

    print("\n[2] 라이브 1d vs 캐시 1d 패리티 (겹치는 최신 봉)")
    for sym in mr.symbols:
        df = snap.bars.get(sym, {}).get("1d")
        cdf = cache.load(sym, "1d")
        if df is None or cdf is None:
            print(f"{sym:10} 비교불가"); continue
        cdf = cdf.set_index("timestamp")
        live_ts = pd.to_datetime(df["timestamp"])
        common = [t for t in live_ts if t in cdf.index]
        if not common:
            print(f"{sym:10} 겹치는 ts 없음 (캐시 끝={cdf.index[-1]}, 라이브 끝={live_ts.iloc[-1]})"); continue
        t = common[-1]
        lc = float(df[live_ts == t]["close"].iloc[-1])
        cc = float(cdf.loc[t, "close"])
        d = abs(lc - cc)
        print(f"{sym:10} ts={t}  live={lc:.4f} cache={cc:.4f} 차이={d:.6f} {'✅' if d < 1e-4 else '⚠️ DIFF'}")

    print("\n[3] mean_reversion.generate_signals 실제 호출 (전체 코드경로)")
    try:
        rg = p.get("regime", {})
        rd = RegimeDetector(primary_symbol=symbols[0], primary_tf=rg.get("primary_tf", "1h"))
        regime = rd.classify(snap)
        rstr = regime.regime.value
    except Exception as e:
        from regime.models import MarketRegime, RegimeState
        regime = RegimeState(regime=MarketRegime.RANGING, adx=0.0, bb_width_pct=0.0)
        rstr = f"RANGING(fallback: {e})"
    sigs = mr.generate_signals(snap, regime)
    print(f"  regime={rstr}, use_regime_gate={mr.use_regime_gate}")
    if not sigs:
        why = "hour≠0 게이트" if not fires else "RSI 극단(≤30/≥70) 심볼 없음"
        print(f"  → 신호 0건 ({why})")
    for s in sigs:
        print(f"  → {s.symbol} {s.direction} entry={s.entry_price:.4f} "
              f"tp={s.tp_price:.4f} sl={s.sl_price:.4f}")


if __name__ == "__main__":
    main()
