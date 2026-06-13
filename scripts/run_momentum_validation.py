"""15m 모멘텀 검증 풀런 + 펀딩 임팩트 정량화.

run_backtest.build_engine를 재사용하되, ledger 전체를 뽑아 per-trade 펀딩/수수료/슬리피지
기여도를 분해한다. 또한 진입을 MAKER 지정가로 강제했을 때의 가정적 결과를 보기 위한
'maker' 모드도 지원(엔진 패치 — 진입 OrderType만 LIMIT, 돌파레벨 지정가).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="config/momentum_15m_validation.yaml")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="ledger CSV 출력 경로")
    args = ap.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100_000)
    since = pd.Timestamp(args.start or bt.get("start"), tz="UTC")
    until_str = args.end or bt.get("end")
    until = pd.Timestamp(until_str, tz="UTC") if until_str else None

    data_cfg = p.get("data", {})
    loader = DataLoader(
        symbols=p["symbols"],
        timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "15m"),
        cache_dir=data_cfg.get("cache_dir", "data/cache"),
        lookback=data_cfg.get("lookback_bars", 60),
    )
    engine = build_engine(p, initial_capital)

    snapshots = loader.iterate(since=since, until=until)
    report = engine.run(snapshots)
    report.print_summary()

    df = engine.ledger.to_dataframe()
    if df.empty:
        print("거래 없음.")
        return

    # ── per-trade 비용 분해 (모두 USD) ──
    # pnl = raw_pnl - commission - slippage - funding  (TradeRecord.pnl은 전부 포함된 net)
    # gross(전비용 전) = pnl + commission + slippage + funding
    df["gross_pnl"] = df["pnl"] + df["commission"] + df["slippage_cost"] + df["funding_paid"]
    df["hold_h"] = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600

    n = len(df)
    notional = df["size_usd"]  # 진입 명목 (USD)
    tot_notional = notional.sum()

    # bps는 명목 기준 (per-trade): cost / notional * 1e4
    def bps(series):
        return (series / notional * 1e4)

    net_bps = bps(df["pnl"])
    gross_bps = bps(df["gross_pnl"])
    fund_bps = bps(df["funding_paid"])
    comm_bps = bps(df["commission"])
    slip_bps = bps(df["slippage_cost"])

    print("\n" + "=" * 60)
    print("  비용 분해 (per-trade, 명목 기준 bps)")
    print("=" * 60)
    print(f"  거래 수:                {n}")
    print(f"  총 진입 명목:           ${tot_notional:,.0f}")
    print(f"  평균 명목/trade:        ${notional.mean():,.0f}")
    print(f"  평균 보유시간:          {df['hold_h'].mean():.2f}h")
    print("-" * 60)
    print(f"  GROSS expectancy:       {gross_bps.mean():+.3f} bps/trade")
    print(f"  - commission:           {-comm_bps.mean():+.3f} bps/trade")
    print(f"  - slippage:             {-slip_bps.mean():+.3f} bps/trade")
    print(f"  - funding:              {-fund_bps.mean():+.3f} bps/trade   <<< Scalp 누락분")
    print(f"  = NET expectancy:       {net_bps.mean():+.3f} bps/trade")
    print("-" * 60)
    print(f"  총 GROSS PnL:           ${df['gross_pnl'].sum():,.2f}")
    print(f"  총 commission:          ${-df['commission'].sum():,.2f}")
    print(f"  총 slippage:            ${-df['slippage_cost'].sum():,.2f}")
    print(f"  총 funding:             ${-df['funding_paid'].sum():,.2f}   <<< Scalp 누락분")
    print(f"  총 NET PnL:             ${df['pnl'].sum():,.2f}")
    print("-" * 60)
    # 펀딩이 net에서 차지하는 비중
    tot_fund = df["funding_paid"].sum()
    print(f"  펀딩 부호:              {'비용(롱이 펀딩 지불 우세)' if tot_fund>0 else '수취'}")
    pos_funded = (df["funding_paid"] > 0).mean() * 100
    print(f"  펀딩 비용 발생 거래%:    {pos_funded:.1f}%")
    # 펀딩을 0으로 했을 때 net (펀딩 제거 시 expectancy)
    net_nofund_bps = bps(df["pnl"] + df["funding_paid"])
    print(f"  펀딩 제외 시 NET:        {net_nofund_bps.mean():+.3f} bps/trade")
    print(f"  펀딩 임팩트:             {net_bps.mean() - net_nofund_bps.mean():+.3f} bps/trade")

    # 방향별 / 심볼별 분해
    print("-" * 60)
    print("  방향별 펀딩 (bps/trade):")
    for d, g in df.groupby("direction"):
        print(f"    {d:5s}: n={len(g):5d}  funding={-(g['funding_paid']/g['size_usd']*1e4).mean():+.3f}  net={(g['pnl']/g['size_usd']*1e4).mean():+.3f}")
    print("  심볼별 (net bps / funding bps):")
    for s, g in df.groupby("symbol"):
        print(f"    {s:9s}: n={len(g):5d}  net={(g['pnl']/g['size_usd']*1e4).mean():+.3f}  funding={-(g['funding_paid']/g['size_usd']*1e4).mean():+.3f}")
    print("  exit_reason 분포:")
    print(df["exit_reason"].value_counts().to_string())
    print("=" * 60)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nledger 저장: {args.out}")


if __name__ == "__main__":
    main()
