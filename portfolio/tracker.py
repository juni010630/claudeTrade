"""포트폴리오 중앙 상태 관리자."""
from __future__ import annotations

import pandas as pd

from execution.models import Fill
from portfolio.ledger import Ledger, TradeRecord
from regime.models import MarketRegime
from risk.models import PortfolioState, Position


class PortfolioTracker:
    def __init__(self, initial_capital: float, ledger: Ledger, notifier=None) -> None:
        self.ledger = ledger
        self.state = PortfolioState(
            equity=initial_capital,
            cash=initial_capital,
            daily_start_equity=initial_capital,
        )
        self.notifier = notifier

    def apply_fill(self, fill: Fill) -> None:
        """진입 Fill → 포지션 오픈."""
        order = fill.order
        pos = Position(
            symbol=order.symbol,
            strategy=order.strategy,
            direction=order.direction,
            entry_price=fill.fill_price,
            size_usd=order.size_usd,
            leverage=order.leverage,
            tp_price=order.tp_price,
            sl_price=order.sl_price,
            opened_at=fill.timestamp,
        )
        self.state.positions[order.symbol] = pos
        self.state.cash -= fill.total_cost  # 수수료+슬리피지 비용만 차감

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: pd.Timestamp,
        exit_reason: str,
        regime: MarketRegime,
        confluence_score: int,
        commission: float = 0.0,
        slippage_cost: float = 0.0,
    ) -> None:
        pos = self.state.positions.pop(symbol, None)
        if pos is None:
            return

        raw_pnl = pos.size_usd * (exit_price - pos.entry_price) / pos.entry_price
        if pos.direction == "short":
            raw_pnl = -raw_pnl

        total_cost = commission + slippage_cost + pos.funding_paid
        realized_pnl = raw_pnl - total_cost

        self.state.equity += realized_pnl
        self.state.cash += realized_pnl

        if self.notifier is not None and getattr(self.notifier, "enabled", False):
            try:
                self.notifier.notify_exit(
                    symbol=symbol,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size_usd=pos.size_usd,
                    leverage=pos.leverage,
                    pnl=realized_pnl,
                    exit_reason=exit_reason,
                    entry_time=pos.opened_at,
                    exit_time=exit_time,
                    equity=self.state.equity,
                )
            except Exception:
                pass

        self.ledger.append(
            TradeRecord(
                trade_id=self.ledger.next_id,
                symbol=symbol,
                strategy=pos.strategy,
                direction=pos.direction,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size_usd=pos.size_usd,
                leverage=pos.leverage,
                pnl=realized_pnl,
                commission=commission,
                slippage_cost=slippage_cost,
                funding_paid=pos.funding_paid,
                entry_time=pos.opened_at,
                exit_time=exit_time,
                exit_reason=exit_reason,
                regime_at_entry=regime,
                confluence_score=confluence_score,
            )
        )

    def apply_funding(self, accruals: dict[str, float]) -> None:
        for sym, cost in accruals.items():
            if sym in self.state.positions:
                self.state.positions[sym].funding_paid += cost
            self.state.equity -= cost
            self.state.cash -= cost

    def mark_to_market(self, prices: dict[str, float]) -> None:
        total_unrealized = 0.0
        for sym, pos in self.state.positions.items():
            price = prices.get(sym, pos.entry_price)
            raw = pos.size_usd * (price - pos.entry_price) / pos.entry_price
            if pos.direction == "short":
                raw = -raw
            pos.unrealized_pnl = raw
            total_unrealized += raw

    def reset_daily(self) -> None:
        self.state.daily_start_equity = self.state.equity

    def snapshot(self) -> PortfolioState:
        return self.state
