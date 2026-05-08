"""포트폴리오 중앙 상태 관리자."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

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
            peak_price=fill.fill_price,
            initial_sl_price=order.sl_price,
            entry_commission=fill.commission,
            entry_slippage=fill.slippage_cost,
            confluence_score=(order.signal_score.total if order.signal_score else 0),
        )
        self.state.positions[order.symbol] = pos
        self.state.cash -= fill.total_cost  # 진입 수수료+슬리피지 즉시 차감
        logger.info(
            "ENTRY %s %s %dx | price=%.4f size=$%.2f | TP=%.4f SL=%.4f | score=%d",
            order.symbol, order.direction, order.leverage,
            fill.fill_price, order.size_usd,
            order.tp_price, order.sl_price,
            order.signal_score.total if order.signal_score else 0,
        )

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

        # cash 업데이트: 진입 비용(entry_commission/slippage)은 apply_fill에서,
        # 펀딩(funding_paid)은 apply_funding에서 이미 차감됨 → 여기선 exit 비용만 반영.
        exit_cost = commission + slippage_cost
        self.state.cash += raw_pnl - exit_cost

        # ledger용 realized_pnl: 전체 비용(진입+청산+펀딩) 모두 포함
        realized_pnl = raw_pnl - pos.entry_commission - pos.entry_slippage - exit_cost - pos.funding_paid

        if self.notifier is not None and getattr(self.notifier, "enabled", False):
            try:
                # cash 기준 equity 추정 (MTM 재계산 전이므로 unrealized 미포함)
                approx_equity = self.state.cash + sum(
                    p.unrealized_pnl for p in self.state.positions.values()
                )
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
                    equity=approx_equity,
                )
            except Exception:
                pass

        record = TradeRecord(
            trade_id=self.ledger.next_id,
            symbol=symbol,
            strategy=pos.strategy,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            leverage=pos.leverage,
            pnl=realized_pnl,
            commission=pos.entry_commission + commission,
            slippage_cost=pos.entry_slippage + slippage_cost,
            funding_paid=pos.funding_paid,
            entry_time=pos.opened_at,
            exit_time=exit_time,
            exit_reason=exit_reason,
            regime_at_entry=regime,
            confluence_score=confluence_score,
        )
        self.ledger.append(record)
        hold_h = (exit_time - pos.opened_at).total_seconds() / 3600
        logger.info(
            "EXIT  %s %s %s | %.4f→%.4f | PnL $%+.2f (%+.1f%%) | %s %.0fh",
            symbol, pos.direction, pos.strategy,
            pos.entry_price, exit_price,
            realized_pnl, realized_pnl / pos.size_usd * 100,
            exit_reason, hold_h,
        )

    def apply_funding(self, accruals: dict[str, float]) -> None:
        for sym, cost in accruals.items():
            if sym not in self.state.positions:
                continue  # 포지션 없는 심볼은 무시 (방어)
            self.state.positions[sym].funding_paid += cost
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
        # equity = 실현기준 cash + 미실현 PnL (진정한 MTM 자산)
        self.state.equity = self.state.cash + total_unrealized

    def reset_daily(self) -> None:
        self.state.daily_start_equity = self.state.equity

    def snapshot(self) -> PortfolioState:
        return self.state
