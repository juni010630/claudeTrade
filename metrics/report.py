"""MetricsReport — 전체 + 전략별 성과 집계."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metrics.drawdown import max_drawdown, recovery_time
from metrics.returns import calmar, cagr, sharpe, sortino
from metrics.trade_stats import (
    avg_rr,
    consecutive_losses,
    profit_factor,
    strategy_breakdown,
    win_rate,
)
from portfolio.equity_curve import EquityCurve
from portfolio.ledger import Ledger


@dataclass
class MetricsReport:
    # 포트폴리오 전체
    initial_equity: float = 0.0
    final_equity: float = 0.0
    total_return_pct: float = 0.0
    cagr: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    recovery_time: str = "N/A"

    # 거래 통계
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_consecutive_losses: int = 0

    # 전략별 분석
    strategy_breakdown: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 파산 플래그: cross margin 청산으로 equity ≤ 0 도달 시 True
    bankrupt: bool = False

    # 조기 중단 플래그: abort_mdd_threshold 초과로 백테가 도중 종료됨
    aborted: bool = False

    @classmethod
    def from_run(cls, equity_curve: EquityCurve, ledger: Ledger) -> "MetricsReport":
        eq = equity_curve.to_series()
        df = ledger.to_dataframe()

        if eq.empty:
            return cls()

        rec = recovery_time(eq)
        rec_str = str(rec) if rec is not None else "미회복"
        initial = float(eq.iloc[0])
        final = float(eq.iloc[-1])

        return cls(
            initial_equity=initial,
            final_equity=final,
            total_return_pct=(final - initial) / initial * 100 if initial > 0 else 0.0,
            cagr=cagr(eq) * 100,
            sharpe=sharpe(eq),
            sortino=sortino(eq),
            calmar=calmar(eq),
            max_drawdown=max_drawdown(eq) * 100,
            recovery_time=rec_str,
            total_trades=len(df) if not df.empty else 0,
            win_rate=win_rate(df) * 100 if not df.empty else 0.0,
            profit_factor=profit_factor(df) if not df.empty else 0.0,
            avg_rr=avg_rr(df) if not df.empty else 0.0,
            max_consecutive_losses=consecutive_losses(df) if not df.empty else 0,
            strategy_breakdown=strategy_breakdown(df) if not df.empty else pd.DataFrame(),
        )

    def print_summary(self) -> None:
        print("=" * 50)
        print("  백테스트 결과")
        print("=" * 50)
        print(f"  초기 자본:         ${self.initial_equity:>12,.2f}")
        print(f"  최종 자산:         ${self.final_equity:>12,.2f}")
        print(f"  총 수익률:         {self.total_return_pct:>+11.2f}%")
        print(f"  CAGR:              {self.cagr:>+11.2f}%")
        print(f"  Sharpe:            {self.sharpe:>12.3f}")
        print(f"  Sortino:           {self.sortino:>12.3f}")
        print(f"  Calmar:            {self.calmar:>12.3f}")
        print(f"  Max Drawdown:      {self.max_drawdown:>+11.2f}%")
        print(f"  회복 시간:         {self.recovery_time}")
        print("-" * 50)
        print(f"  총 거래 수:        {self.total_trades:>12}")
        print(f"  승률:              {self.win_rate:>11.1f}%")
        print(f"  Profit Factor:     {self.profit_factor:>12.3f}")
        print(f"  평균 R:R:          {self.avg_rr:>12.3f}")
        print(f"  최대 연속 손절:    {self.max_consecutive_losses:>12}")
        if not self.strategy_breakdown.empty:
            print("-" * 50)
            print("  전략별 분석:")
            print(self.strategy_breakdown.to_string(float_format="{:.2f}".format))
        print("=" * 50)

    def to_dict(self) -> dict:
        d = {
            "initial_equity": self.initial_equity,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "recovery_time": self.recovery_time,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_rr": self.avg_rr,
            "max_consecutive_losses": self.max_consecutive_losses,
        }
        if not self.strategy_breakdown.empty:
            d["strategy_breakdown"] = self.strategy_breakdown.to_dict()
        return d
