"""counter_entry_bt.py — 반대 방향 1봉 후 원래 방향 진입 백테스트.

전략:
  시그널 T → 반대 방향 1봉 진입 (counter)
  T+1     → counter 청산 + 원래 방향 진입 (main, 원래 ATR 거리 유지)

Counter 청산 방법:
  TP = close_T+1 로 설정 → low_T+1 <= close_T+1 (short) / high_T+1 >= close_T+1 (long) 은
  OHLC 정의상 항상 참 → T+1 close 에서 100% 청산 보장.
  SL = entry ± 5×ATR (극단적 1봉 이상 변동 대비 safety)

비교:
  A. 기준선  — 즉시 진입 (N=0)
  B. N=1 지연  — 1봉 후 진입 (counter 없음)
  C. 이 전략  — counter 1봉 + main T+1 진입
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.cache import ParquetCache
from data.loader import DataLoader
from scripts.run_backtest import build_engine


def load_close_series(cache: ParquetCache, symbols: list[str]) -> dict[str, pd.Series]:
    """심볼별 1h close 시리즈 반환. {sym: Series(index=timestamp UTC)}"""
    result = {}
    for sym in symbols:
        df = cache.load(sym, "1h", data_type="ohlcv")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        result[sym] = df.set_index("timestamp")["close"]
    return result


def build_combined_df(
    entries: pd.DataFrame,
    close_map: dict[str, pd.Series],
    tp_mult: float = 3.5,
    sl_mult: float = 1.8,
    orig_sl_mult: float = 1.8,
    counter_sl_atr_mult: float = 5.0,
) -> pd.DataFrame:
    """
    entries_df → (counter + main) 통합 df 생성.

    counter: 원래 시그널 반대 방향, T+1 close에서 100% 청산
    main:    원래 방향, T+1 entry, ATR 기반 TP/SL
    """
    counter_rows, main_rows = [], []
    skipped = 0

    for _, row in entries.iterrows():
        sym = row["symbol"]
        ts  = pd.Timestamp(row["timestamp"]).tz_convert("UTC")
        ts1 = ts + pd.Timedelta(hours=1)

        closes = close_map.get(sym)
        if closes is None or ts1 not in closes.index:
            skipped += 1
            continue

        close_t1   = float(closes[ts1])
        entry_price = float(row["entry_price"])
        direction   = row["direction"]
        atr = float(row["sl_dist"]) / orig_sl_mult

        # ── counter trade (반대 방향, T에서 진입, T+1 close에서 청산) ──────
        counter_dir = "short" if direction == "long" else "long"
        if counter_dir == "short":
            # short TP: engine checks low_T+1 <= tp → close_T+1 는 항상 low_T+1 이상 → 항상 TP 히트
            counter_tp = close_t1
            counter_sl = entry_price + counter_sl_atr_mult * atr
        else:
            counter_tp = close_t1
            counter_sl = entry_price - counter_sl_atr_mult * atr

        counter_rows.append({
            "timestamp":   ts,
            "symbol":      sym,
            "strategy":    row["strategy"],
            "direction":   counter_dir,
            "entry_price": entry_price,
            "tp_price":    counter_tp,
            "sl_price":    counter_sl,
            "score":       row["score"],
            "tier":        row["tier"],
        })

        # ── main trade (원래 방향, T+1 진입, ATR 거리 유지) ──────────────
        if direction == "long":
            main_tp = close_t1 + tp_mult * atr
            main_sl = close_t1 - sl_mult * atr
        else:
            main_tp = close_t1 - tp_mult * atr
            main_sl = close_t1 + sl_mult * atr

        main_rows.append({
            "timestamp":   ts1,
            "symbol":      sym,
            "strategy":    row["strategy"],
            "direction":   direction,
            "entry_price": close_t1,   # shift = 0 (replay mkt_price = close_t1)
            "tp_price":    main_tp,
            "sl_price":    main_sl,
            "score":       row["score"],
            "tier":        row["tier"],
        })

    if skipped:
        print(f"  (스킵: {skipped}건 — T+1 가격 데이터 없음)")

    df = pd.concat([pd.DataFrame(counter_rows), pd.DataFrame(main_rows)], ignore_index=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def run_and_report(label: str, engine, df_or_none, loader, since, until) -> dict:
    if df_or_none is not None:
        report = engine.run_replay(df_or_none, loader.iterate(since=since, until=until))
    else:
        report = engine.run(loader.iterate(since=since, until=until))

    records = engine.ledger.records
    if records:
        hold_h = [(r.exit_time - r.entry_time).total_seconds() / 3600 for r in records]
        avg_h  = sum(hold_h) / len(hold_h)
    else:
        avg_h = 0.0

    # 분기별 수익률 (손실 분기 수)
    eq = engine.equity_curve.to_series()
    eq.index = pd.to_datetime(eq.index, utc=True)
    daily = eq.resample("D").last().ffill()
    q_ret = daily.resample("QE").last().pct_change().dropna() * 100
    neg_q = int((q_ret < 0).sum())

    print(f"\n[{label}]")
    print(f"  Sharpe {report.sharpe:.2f}  MDD {report.max_drawdown*100:.1f}%  "
          f"PF {getattr(report,'profit_factor',float('nan')):.2f}  "
          f"WR {report.win_rate:.1f}%  "
          f"거래 {report.total_trades}건  "
          f"평균홀딩 {avg_h:.1f}h  손실분기 {neg_q}개  "
          f"최종 ${report.final_equity:,.0f}")

    return {
        "label":        label,
        "sharpe":       round(report.sharpe, 3),
        "mdd_%":        round(report.max_drawdown * 100, 1),
        "pf":           round(getattr(report, "profit_factor", float("nan")), 2),
        "wr_%":         round(report.win_rate, 1),
        "trades":       report.total_trades,
        "avg_hold_h":   round(avg_h, 1),
        "neg_quarters": neg_q,
        "final_equity": round(report.final_equity, 0),
    }


def main() -> None:
    with open("config/final_v13_eth.yaml") as f:
        p = yaml.safe_load(f)

    bt = p.get("backtest", {})
    initial_capital = bt.get("initial_capital", 100)
    since = pd.Timestamp("2022-01-01", tz="UTC")
    until = pd.Timestamp("2026-06-06", tz="UTC")

    orig_sl_mult = p.get("strategies", {}).get("ema_cross", {}).get("atr_sl_mult", 1.8)
    tp_mult      = 3.5
    sl_mult      = 1.8

    data_cfg = p.get("data", {})
    cache_dir = data_cfg.get("cache_dir", "data/cache")

    cache = ParquetCache(cache_dir)
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=cache_dir,
        lookback=data_cfg.get("lookback_bars", 300),
    )

    entries = pd.read_parquet("data/entries.parquet")
    entries["timestamp"] = pd.to_datetime(entries["timestamp"], utc=True)
    print(f"entries 로드: {len(entries)}건")

    # 심볼별 close 시리즈 미리 로드
    close_map = load_close_series(cache, p["symbols"])

    # ── N=1 delay df (counter 없이 1봉 지연) ──────────────────────────────
    df_n1 = entries.copy()
    atr_n1 = df_n1["sl_dist"] / orig_sl_mult
    long_n1 = df_n1["direction"] == "long"
    df_n1.loc[long_n1,  "tp_price"] = df_n1.loc[long_n1,  "entry_price"] + tp_mult * atr_n1[long_n1]
    df_n1.loc[~long_n1, "tp_price"] = df_n1.loc[~long_n1, "entry_price"] - tp_mult * atr_n1[~long_n1]
    df_n1.loc[long_n1,  "sl_price"] = df_n1.loc[long_n1,  "entry_price"] - sl_mult * atr_n1[long_n1]
    df_n1.loc[~long_n1, "sl_price"] = df_n1.loc[~long_n1, "entry_price"] + sl_mult * atr_n1[~long_n1]
    df_n1["timestamp"] = df_n1["timestamp"] + pd.Timedelta(hours=1)

    # ── counter + main df ──────────────────────────────────────────────
    print("counter + main df 생성 중...")
    df_counter = build_combined_df(
        entries, close_map, tp_mult=tp_mult, sl_mult=sl_mult,
        orig_sl_mult=orig_sl_mult,
    )
    print(f"  counter {len(df_counter[df_counter['tp_price'].notna()])//2}건 × 2 = {len(df_counter)}행")

    # ── 세 가지 비교 ──────────────────────────────────────────────────────
    results = []

    # A. 기준선 (N=0, 즉시 진입)
    atr_base = entries["sl_dist"] / orig_sl_mult
    long_base = entries["direction"] == "long"
    df_base = entries.copy()
    df_base.loc[long_base,  "tp_price"] = df_base.loc[long_base,  "entry_price"] + tp_mult * atr_base[long_base]
    df_base.loc[~long_base, "tp_price"] = df_base.loc[~long_base, "entry_price"] - tp_mult * atr_base[~long_base]
    df_base.loc[long_base,  "sl_price"] = df_base.loc[long_base,  "entry_price"] - sl_mult * atr_base[long_base]
    df_base.loc[~long_base, "sl_price"] = df_base.loc[~long_base, "entry_price"] + sl_mult * atr_base[~long_base]
    results.append(run_and_report("A. 기준선 (N=0, 즉시)", build_engine(p, initial_capital),
                                   df_base, loader, since, until))

    # B. N=1 지연 (counter 없음)
    results.append(run_and_report("B. N=1 지연 (counter 없음)", build_engine(p, initial_capital),
                                   df_n1, loader, since, until))

    # C. counter 1봉 + main T+1
    results.append(run_and_report("C. counter 1봉 + main T+1", build_engine(p, initial_capital),
                                   df_counter, loader, since, until))

    # ── 요약 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("비교 요약")
    print("=" * 80)
    df_res = pd.DataFrame(results)
    print(df_res[["label","sharpe","mdd_%","pf","wr_%","trades","avg_hold_h",
                  "neg_quarters","final_equity"]].to_string(index=False))


if __name__ == "__main__":
    main()
