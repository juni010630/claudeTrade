"""거래 통계 — 순수 함수."""
from __future__ import annotations

import pandas as pd


def win_rate(ledger_df: pd.DataFrame) -> float:
    if ledger_df.empty:
        return 0.0
    return float((ledger_df["pnl"] > 0).mean())


def profit_factor(ledger_df: pd.DataFrame) -> float:
    if ledger_df.empty:
        return 0.0
    gross_profit = ledger_df[ledger_df["pnl"] > 0]["pnl"].sum()
    gross_loss = ledger_df[ledger_df["pnl"] < 0]["pnl"].abs().sum()
    return float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")


def avg_rr(ledger_df: pd.DataFrame) -> float:
    """평균 실현 R:R (수익 거래 평균 / 손실 거래 평균 절대값)."""
    if ledger_df.empty:
        return 0.0
    wins = ledger_df[ledger_df["pnl"] > 0]["pnl"]
    losses = ledger_df[ledger_df["pnl"] < 0]["pnl"].abs()
    if wins.empty or losses.empty:
        return 0.0
    return float(wins.mean() / losses.mean())


def consecutive_losses(ledger_df: pd.DataFrame) -> int:
    """전체 기간 최대 연속 손절 횟수."""
    if ledger_df.empty:
        return 0
    is_loss = (ledger_df["pnl"] <= 0).astype(int)
    max_run = 0
    current = 0
    for v in is_loss:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def strategy_breakdown(ledger_df: pd.DataFrame) -> pd.DataFrame:
    """전략별 PnL, 승률, 거래 수 분석."""
    if ledger_df.empty:
        return pd.DataFrame()
    grouped = ledger_df.groupby("strategy")
    result = pd.DataFrame({
        "total_pnl": grouped["pnl"].sum(),
        "trade_count": grouped["pnl"].count(),
        "win_rate": grouped["pnl"].apply(lambda s: (s > 0).mean()),
        "avg_pnl": grouped["pnl"].mean(),
    })
    return result
