"""워크포워드 검증 — 슬라이딩 윈도우 IS/OOS 분리."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data.loader import DataLoader
from metrics.report import MetricsReport


@dataclass
class FoldResult:
    fold: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp
    is_report: MetricsReport
    oos_report: MetricsReport

    @property
    def degradation_ratio(self) -> float:
        """OOS Sharpe / IS Sharpe. 1에 가까울수록 오버피팅 없음."""
        if self.is_report.sharpe == 0:
            return 0.0
        return self.oos_report.sharpe / self.is_report.sharpe


class WalkForwardValidator:
    def __init__(
        self,
        engine_factory,  # Callable[[], BacktestEngine]
        loader: DataLoader,
        in_sample_months: int = 6,
        out_sample_months: int = 2,
    ) -> None:
        self.engine_factory = engine_factory
        self.loader = loader
        self.is_months = in_sample_months
        self.oos_months = out_sample_months

    def run(
        self,
        total_start: pd.Timestamp,
        total_end: pd.Timestamp,
    ) -> list[FoldResult]:
        results: list[FoldResult] = []
        fold = 0
        cursor = total_start

        is_delta = pd.DateOffset(months=self.is_months)
        oos_delta = pd.DateOffset(months=self.oos_months)

        while cursor + is_delta + oos_delta <= total_end:
            is_start = cursor
            is_end = cursor + is_delta
            oos_start = is_end
            oos_end = oos_start + oos_delta

            # IS 실행
            engine_is = self.engine_factory()
            is_snapshots = self.loader.iterate(since=is_start, until=is_end)
            is_report = engine_is.run(is_snapshots)

            # OOS 실행 (파라미터 동일, 데이터만 다름)
            engine_oos = self.engine_factory()
            oos_snapshots = self.loader.iterate(since=oos_start, until=oos_end)
            oos_report = engine_oos.run(oos_snapshots)

            results.append(
                FoldResult(
                    fold=fold,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    is_report=is_report,
                    oos_report=oos_report,
                )
            )

            fold += 1
            cursor += oos_delta  # 슬라이딩

        return results

    @staticmethod
    def print_results(results: list[FoldResult]) -> None:
        print(f"{'폴드':>4} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'열화비율':>9}")
        print("-" * 40)
        for r in results:
            print(
                f"{r.fold:>4} {r.is_report.sharpe:>10.3f} "
                f"{r.oos_report.sharpe:>11.3f} {r.degradation_ratio:>9.3f}"
            )
