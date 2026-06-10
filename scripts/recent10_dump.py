"""최근 10일 v17 백테스트 — 전체 매매 로그 + 보유 포지션 덤프 (프로덕션 무수정)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from scripts.run_backtest import build_engine

CONFIG = "config/final_v17.yaml"
START = "2026-05-30"
END = None  # 캐시 최신까지
CAPITAL = 9450.0  # 라이브 봇 운용자본 근사

p = yaml.safe_load(open(CONFIG))
data_cfg = p.get("data", {})
loader = DataLoader(
    symbols=p["symbols"],
    timeframes=p["timeframes"],
    primary_tf=p.get("primary_timeframe", "1h"),
    cache_dir=data_cfg.get("cache_dir", "data/cache"),
    lookback=data_cfg.get("lookback_bars", 300),
)

engine = build_engine(p, CAPITAL)
since = pd.Timestamp(START, tz="UTC")
until = pd.Timestamp(END, tz="UTC") if END else None
report = engine.run(loader.iterate(since=since, until=until))

print("=" * 70)
print(f"v17 최근 10일 백테스트  {START} ~ {END or '캐시최신'}  시드 ${CAPITAL:,.0f}")
print("=" * 70)
report.print_summary()

# ---- 전체 매매 로그 (청산 완료) ----
df = engine.ledger.to_dataframe()
print("\n" + "=" * 70)
print(f"청산 완료 매매 로그: {len(df)}건")
print("=" * 70)
if len(df):
    cols = ["entry_time", "exit_time", "symbol", "strategy", "direction",
            "entry_price", "exit_price", "leverage", "size_usd", "pnl", "exit_reason"]
    show = df[cols].copy()
    show["entry_time"] = pd.to_datetime(show["entry_time"]).dt.strftime("%m-%d %H:%M")
    show["exit_time"] = pd.to_datetime(show["exit_time"]).dt.strftime("%m-%d %H:%M")
    for c in ("entry_price", "exit_price"):
        show[c] = show[c].map(lambda x: f"{x:,.4f}")
    show["size_usd"] = show["size_usd"].map(lambda x: f"{x:,.0f}")
    show["pnl"] = show["pnl"].map(lambda x: f"{x:+,.2f}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(show.to_string(index=False))
    print(f"\n실현 PnL 합계: ${df['pnl'].sum():+,.2f}")
else:
    print("(청산 완료 매매 없음)")

# ---- 보유중 포지션 (미청산) ----
state = engine.tracker.snapshot()
prices = engine._get_prices(engine._last_snapshot) if hasattr(engine, "_last_snapshot") else {}
print("\n" + "=" * 70)
print(f"보유중(미청산) 포지션: {len(state.positions)}건")
print("=" * 70)
if state.positions:
    for sym, pos in state.positions.items():
        mark = prices.get(sym, pos.entry_price)
        upnl = pos.size_usd * (mark - pos.entry_price) / pos.entry_price
        if pos.direction == "short":
            upnl = -upnl
        print(f"  {sym:10s} {pos.direction:5s} entry={pos.entry_price:,.4f} "
              f"mark={mark:,.4f} lev={pos.leverage}x size=${pos.size_usd:,.0f} "
              f"TP={pos.tp_price:,.4f} SL={pos.sl_price:,.4f} uPnL=${upnl:+,.2f}")
else:
    print("(보유중 포지션 없음)")
