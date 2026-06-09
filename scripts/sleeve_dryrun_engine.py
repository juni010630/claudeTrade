"""raw 슬리브 신호가 '풀 엔진'(스코어+티어게이트+블록)을 통과해 실제 거래후보가 되는가.
계좌 무접촉(공개 피드 + 백테 브로커, 주문 미전송)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from data.live_feed import LiveFeed
import scripts.run_backtest as rb

CONFIG = "config/merged_v16_sleeve.yaml"
SLEEVE = {"mean_reversion"}


def main() -> None:
    p = yaml.safe_load(open(CONFIG))
    feed = LiveFeed(symbols=p["symbols"], timeframes=p["timeframes"],
                    primary_tf="1h", lookback=300, demo=False)
    snap = feed.snapshot_now()
    print(f"snap_ts={snap.timestamp} hour={snap.timestamp.hour}")

    engine = rb.build_engine(p, 100.0)
    regime = engine.regime_detector.classify(snap)
    prices = engine._get_prices(snap)
    cands = engine._generate_all_candidates(snap, regime, prices)

    sleeve = [c for c in cands if c["strategy"] in SLEEVE]
    trend = [c for c in cands if c["strategy"] not in SLEEVE]
    print(f"\n전체 거래후보(스코어+티어게이트+블록 통과): {len(cands)}건 "
          f"(슬리브 {len(sleeve)} / 추세 {len(trend)})")
    print("\n[슬리브 후보 — raw 신호가 풀 엔진 통과했나]")
    if not sleeve:
        print("  → 0건. raw RSI신호는 있었지만 ConfluenceScorer 점수<tier_a(3)로 NO_TRADE 탈락.")
    for c in sleeve:
        print(f"  {c['symbol']:10} {c['direction']:5} score={c['score']} tier={c['tier']} "
              f"entry={c['entry_price']:.4f}")
    if trend:
        print("\n[추세 후보(참고)]")
        for c in trend:
            print(f"  {c['symbol']:10} {c['direction']:5} {c['strategy']:18} score={c['score']} tier={c['tier']}")
    print("\n※ 이 후보들도 실제 주문 전에 corr_filter·max_positions·사이징을 추가로 거침.")
    print("※ 이 스크립트는 주문 미전송(검사용). 원격 봇과 무관.")


if __name__ == "__main__":
    main()
