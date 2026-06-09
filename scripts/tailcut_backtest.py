"""계좌단위 컷(A) vs 비대칭 v16-only 컷(B) vs 기준선 백테 비교 실험.

프로덕션 엔진/config 무수정. BacktestEngine을 __class__ 재지정으로 TailCutEngine으로
바꿔 컷 로직(peak 대비 -X% → 대상 전략 청산 + 쿨다운 정지 후 재개)만 주입한다.

- A (계좌단위): 대상=전체 전략 → 전 포지션 청산 + 전체 정지
- B (비대칭):  대상=추세(ema_cross/multi_tf_breakout) → 추세만 청산/정지, 슬리브 유지

연도별은 '경로 슬라이스'(동일 실현경로를 연도로 자른 것 — 독립 $100 실행 아님).
인터벤션 비교 목적이라 같은 경로에서 비교하는 게 맞다.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml

from data.loader import DataLoader
from engine.backtest import BacktestEngine
from metrics.drawdown import max_drawdown
from metrics.returns import sharpe
from metrics.report import MetricsReport
import scripts.run_backtest as rb

CONFIG = "config/merged_v16_sleeve.yaml"
TREND = {"ema_cross", "multi_tf_breakout"}
COOLDOWN_H = 168  # 7일


class TailCutEngine(BacktestEngine):
    """peak 대비 -threshold 도달 시 대상 전략 청산 + 쿨다운 정지(후 재개).

    _cut_targets=None → 전체(A). set → 그 전략만(B).
    추가 속성은 run_config에서 직접 주입(__init__ 미사용).
    """

    def _generate_all_candidates(self, snapshot, regime, prices):
        cands = super()._generate_all_candidates(snapshot, regime, prices)
        if self._cut_pause_until is not None and snapshot.timestamp < self._cut_pause_until:
            if self._cut_targets is None:
                return []
            cands = [c for c in cands if c["strategy"] not in self._cut_targets]
        return cands

    def _process_bar(self, snapshot) -> None:
        super()._process_bar(snapshot)
        if self._bankrupt or self._aborted:
            return
        state = self.tracker.snapshot()
        eq = state.equity
        if eq > self._cut_peak:
            self._cut_peak = eq
        if self._cut_peak <= 0:
            return
        dd = (eq - self._cut_peak) / self._cut_peak
        now = snapshot.timestamp
        # 쿨다운 종료 후(또는 첫 발동) & 임계 하회 → 발동
        if dd <= -self._cut_threshold and (
            self._cut_pause_until is None or now >= self._cut_pause_until
        ):
            prices = self._get_prices(snapshot)
            targets = [
                s for s, p in state.positions.items()
                if self._cut_targets is None or p.strategy in self._cut_targets
            ]
            for sym in targets:
                self._force_close(sym, prices.get(sym, 0.0), now, "tail_cut")
            self._cut_pause_until = now + pd.Timedelta(hours=self._cut_cooldown_h)
            self._cut_events.append((str(now), round(eq, 2), len(targets)))


def _yearly(eq: pd.Series) -> dict:
    """경로 슬라이스 연도별 (수익%, 연내 MDD%)."""
    out = {}
    for y, seg in eq.groupby(eq.index.year):
        if len(seg) < 2:
            continue
        ret = (seg.iloc[-1] / seg.iloc[0] - 1) * 100
        mdd = max_drawdown(seg) * 100
        out[int(y)] = (round(ret, 1), round(mdd, 1))
    return out


def run_config(spec: dict) -> dict:
    name = spec["name"]
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    bt = p.get("backtest", {})
    since = pd.Timestamp(bt.get("start", "2022-01-01"), tz="UTC")
    until = pd.Timestamp(bt.get("end"), tz="UTC") if bt.get("end") else None

    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    if spec["threshold"] is not None:
        engine.__class__ = TailCutEngine
        engine._cut_threshold = spec["threshold"]
        engine._cut_cooldown_h = spec["cooldown_h"]
        engine._cut_targets = spec["targets"]
        engine._cut_peak = 100.0
        engine._cut_pause_until = None
        engine._cut_events = []

    report = engine.run(loader.iterate(since=since, until=until))
    eq = engine.equity_curve.to_series()
    events = getattr(engine, "_cut_events", [])
    return {
        "name": name,
        "final": report.final_equity,
        "mdd": report.max_drawdown,
        "sharpe": report.sharpe,
        "calmar": report.calmar,
        "recovery": report.recovery_time,
        "trades": report.total_trades,
        "cut_events": events,
        "yearly": _yearly(eq),
    }


def main() -> None:
    specs = [
        {"name": "baseline",       "threshold": None, "targets": None,  "cooldown_h": 0},
        {"name": "A(계좌) -40%",   "threshold": 0.40, "targets": None,  "cooldown_h": COOLDOWN_H},
        {"name": "A(계좌) -45%",   "threshold": 0.45, "targets": None,  "cooldown_h": COOLDOWN_H},
        {"name": "A(계좌) -55%",   "threshold": 0.55, "targets": None,  "cooldown_h": COOLDOWN_H},
        {"name": "B(v16만) -40%",  "threshold": 0.40, "targets": TREND, "cooldown_h": COOLDOWN_H},
        {"name": "B(v16만) -45%",  "threshold": 0.45, "targets": TREND, "cooldown_h": COOLDOWN_H},
    ]
    with ProcessPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(run_config, specs))

    order = {s["name"]: i for i, s in enumerate(specs)}
    results.sort(key=lambda r: order[r["name"]])

    print("\n" + "=" * 78)
    print(f"{'config':16} {'최종$':>10} {'MDD%':>8} {'Sharpe':>7} {'Calmar':>7} {'거래':>5} {'컷':>4}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:16} {r['final']:>10,.0f} {r['mdd']:>8.1f} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['trades']:>5} {len(r['cut_events']):>4}")
    print("=" * 78)

    print("\n[연도별 경로슬라이스: 수익% / 연내MDD%]")
    years = sorted({y for r in results for y in r["yearly"]})
    hdr = "config".ljust(16) + "".join(f"{y:>16}" for y in years)
    print(hdr)
    for r in results:
        row = r["name"].ljust(16)
        for y in years:
            if y in r["yearly"]:
                ret, mdd = r["yearly"][y]
                row += f"{ret:>7.0f}/{mdd:>6.0f}  "
            else:
                row += f"{'—':>16}"
        print(row)

    print("\n[컷 발동 내역]")
    for r in results:
        if r["name"] == "baseline":
            continue
        if r["cut_events"]:
            print(f"  {r['name']}: {len(r['cut_events'])}회")
            for ts, eq, n in r["cut_events"][:8]:
                print(f"      {ts}  equity=${eq:,.0f}  청산 {n}건")
        else:
            print(f"  {r['name']}: 0회 (미발동 = baseline과 동일)")


if __name__ == "__main__":
    main()
