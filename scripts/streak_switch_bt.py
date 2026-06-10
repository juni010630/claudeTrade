"""성과 스트릭 기반 추세↔슬리브 메타 스위치 실험 (v17, 프로덕션 무수정).

추세그룹(ema_cross+multi_tf_breakout) vs 슬리브(mean_reversion)를 자기 성과로
켜고 끄는 메타 스위치 5변형. BacktestEngine __class__ 재할당 + ledger 워터마크로
청산을 경로 무관하게 소비(_force_close 포함) → 그룹 연승/연패 카운터 → 모드 전이.

look-ahead 없음: _process_bar에서 TP/SL 청산(:479)이 후보 생성(:522)보다 먼저
처리되므로, 워터마크 소비(후보 생성 직전)는 항상 이미 실현된 청산만 본다.

변형 (K=2 고정, 스윕 금지 — 사전선언):
  A1   포지션 뮤텍스(대칭): 한 그룹 보유 중 상대 그룹 신규진입 차단
  A2   추세 우선 뮤텍스: 추세 보유 중 슬리브만 차단
  B1   핫핸드 독점: 중립 시작, X 2연승→X 독점, 독점 중 X 2연패→상대 독점
  B2   B1과 같되 2연패 시 중립 복귀
  C1   소프트 가중: 핫그룹 capital_fraction 0.7 / 상대 0.3, 핫그룹 2연패 시 0.5 복원
  NOOP 클래스 재할당 + 소비만 (필터/가중 없음) — BASE와 trade-by-trade 일치해야 함
  BASE 순정 엔진 — 하네스 무결성 대조용

사용:
  python scripts/streak_switch_bt.py            # 전구간 BASE+NOOP+5변형 + 무결성 게이트
  python scripts/streak_switch_bt.py isoos A1 C1  # 생존 변형만 IS/OOS + 연도별
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
import scripts.run_backtest as rb

CONFIG = "config/final_v17.yaml"
K = 2  # 스트릭 길이 (사전 고정)
GROUP = {"ema_cross": "trend", "multi_tf_breakout": "trend", "mean_reversion": "sleeve"}
STRATS = list(GROUP)

IS_W = ("IS", "2022-01-01", "2024-12-31")
OOS_W = ("OOS", "2025-01-01", "2026-04-14")
YEAR_WS = [("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025-26", "2025-01-01", "2026-04-14")]

LEDGER_COLS = ["entry_time", "exit_time", "symbol", "strategy", "direction",
               "entry_price", "exit_price", "size_usd", "pnl"]


def _opp(g: str) -> str:
    return "sleeve" if g == "trend" else "trend"


class StreakSwitchEngine(BacktestEngine):
    """그룹 성과 스트릭 기반 진입 필터/가중. 상태 attr은 외부 주입(__init__ 미사용).

    _ss_variant: A1/A2/B1/B2/C1/NOOP
    _ss_mode: "neutral" | "trend" | "sleeve" (B=독점 그룹, C=핫 그룹)
    """

    def _generate_all_candidates(self, snapshot, regime, prices):
        self._ss_consume(snapshot.timestamp)
        cands = super()._generate_all_candidates(snapshot, regime, prices)
        v = self._ss_variant
        blocked: set[str] = set()
        if v in ("A1", "A2"):
            held = {GROUP[p.strategy] for p in self.tracker.snapshot().positions.values()
                    if p.strategy in GROUP}
            if "trend" in held:
                blocked.add("sleeve")
            if v == "A1" and "sleeve" in held:
                blocked.add("trend")
        elif v in ("B1", "B2") and self._ss_mode != "neutral":
            blocked.add(_opp(self._ss_mode))
        if blocked:
            kept = [c for c in cands if GROUP.get(c["strategy"]) not in blocked]
            self._ss_blocked += len(cands) - len(kept)
            cands = kept
        return cands

    def _ss_consume(self, now) -> None:
        """워터마크 이후의 신규 청산 소비 → 스트릭 갱신 → 모드 전이."""
        v = self._ss_variant
        for r in self.ledger._records[self._ss_seen:]:
            g = GROUP.get(r.strategy)
            if g is None:
                continue
            if r.pnl > 0:
                self._ss_wins[g] += 1
                self._ss_losses[g] = 0
            else:
                self._ss_losses[g] += 1
                self._ss_wins[g] = 0
            old, new = self._ss_mode, self._ss_mode
            if v in ("B1", "B2"):
                if old == "neutral":
                    if self._ss_wins[g] >= K:
                        new = g
                elif old == g and self._ss_losses[g] >= K:
                    new = _opp(g) if v == "B1" else "neutral"
            elif v == "C1":
                if old != g and self._ss_wins[g] >= K:
                    new = g
                elif old == g and self._ss_losses[g] >= K:
                    new = "neutral"
            if new != old:
                self._ss_mode = new
                # 전이 시 양쪽 카운터 리셋 — 동결된 과거 스트릭이 즉시 재전이 일으키는 것 방지
                self._ss_wins = {"trend": 0, "sleeve": 0}
                self._ss_losses = {"trend": 0, "sleeve": 0}
                self._ss_switch_history.append(
                    (str(now), old, new, f"{g}:{'win' if r.pnl > 0 else 'loss'}"))
                if v == "C1":
                    for s in STRATS:
                        self._strategy_capital_fraction[s] = (
                            0.5 if new == "neutral" else (0.7 if GROUP[s] == new else 0.3))
        self._ss_seen = len(self.ledger._records)


def run_config(spec: dict) -> dict:
    name, variant, start, end = spec["name"], spec["variant"], spec["start"], spec["end"]
    with open(CONFIG) as f:
        p = yaml.safe_load(f)
    bt = p.get("backtest", {})
    since = pd.Timestamp(start or bt.get("start", "2022-01-01"), tz="UTC")
    until_s = end or bt.get("end")
    until = pd.Timestamp(until_s, tz="UTC") if until_s else None

    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=p.get("data", {}).get("cache_dir", "data/cache"),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, 100.0)
    if variant != "BASE":
        engine.__class__ = StreakSwitchEngine
        engine._ss_variant = variant
        engine._ss_seen = 0
        engine._ss_mode = "neutral"
        engine._ss_wins = {"trend": 0, "sleeve": 0}
        engine._ss_losses = {"trend": 0, "sleeve": 0}
        engine._ss_switch_history = []
        engine._ss_blocked = 0

    report = engine.run(loader.iterate(since=since, until=until))
    df = engine.ledger.to_dataframe()
    ledger = df[LEDGER_COLS].copy() if not df.empty else pd.DataFrame(columns=LEDGER_COLS)
    return {
        "name": name,
        "final": report.final_equity,
        "ret": report.final_equity - 100.0,
        "mdd": report.max_drawdown,
        "sharpe": report.sharpe,
        "calmar": report.calmar,
        "trades": report.total_trades,
        "switches": getattr(engine, "_ss_switch_history", []),
        "blocked": getattr(engine, "_ss_blocked", 0),
        "ledger": ledger,
    }


def _print_table(results: list[dict]) -> None:
    print(f"\n{'config':14} {'최종$':>10} {'수익%':>9} {'MDD%':>7} {'Sharpe':>7} "
          f"{'Calmar':>7} {'거래':>5} {'스위치':>6} {'차단후보':>8}")
    print("-" * 84)
    for r in results:
        print(f"{r['name']:14} {r['final']:>10,.0f} {r['ret']:>+9.1f} {r['mdd']:>7.1f} "
              f"{r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['trades']:>5} "
              f"{len(r['switches']):>6} {r['blocked']:>8}")


def main_full() -> None:
    variants = ["BASE", "NOOP", "A1", "A2", "B1", "B2", "C1"]
    specs = [{"name": v, "variant": v, "start": None, "end": None} for v in variants]
    print(f"전구간(config 기재 기간) {len(specs)}백테 병렬 실행…")
    with ProcessPoolExecutor(max_workers=len(specs)) as ex:
        results = list(ex.map(run_config, specs))
    by = {r["name"]: r for r in results}

    # 무결성 게이트: NOOP vs BASE trade-by-trade
    lb, ln = by["BASE"]["ledger"].reset_index(drop=True), by["NOOP"]["ledger"].reset_index(drop=True)
    identical = len(lb) == len(ln) and lb.equals(ln)
    print(f"\n[무결성 게이트] BASE {len(lb)}건 vs NOOP {len(ln)}건 → "
          f"{'trade-by-trade 일치 ✓' if identical else '불일치 ✗ — 변형 결과 해석 금지'}")
    if not identical and len(lb) == len(ln):
        diff_cols = [c for c in LEDGER_COLS if not lb[c].equals(ln[c])]
        print(f"  불일치 컬럼: {diff_cols}")

    _print_table(results)

    print("\n[스위치 내역 (B/C변형)]")
    for v in ("B1", "B2", "C1"):
        sw = by[v]["switches"]
        print(f"  {v}: {len(sw)}회")
        for ts, old, new, trig in sw[:12]:
            print(f"      {ts[:19]}  {old:>7} → {new:<7}  ({trig})")
        if len(sw) > 12:
            print(f"      … 외 {len(sw) - 12}회")


def main_isoos(variants: list[str]) -> None:
    windows = [IS_W, OOS_W] + YEAR_WS
    specs = [{"name": f"{v}|{w}", "variant": v, "start": s, "end": e}
             for v in variants for (w, s, e) in windows]
    print(f"IS/OOS+연도별 — {len(variants)}변형 × {len(windows)}구간 = {len(specs)}백테 병렬 실행…")
    with ProcessPoolExecutor(max_workers=min(12, len(specs))) as ex:
        results = list(ex.map(run_config, specs))
    by = {r["name"]: r for r in results}

    print(f"\n{'변형':6} {'구간':8} {'수익%':>9} {'MDD%':>7} {'Sharpe':>7} {'거래':>5} {'스위치':>6}")
    print("-" * 56)
    for v in variants:
        for (w, _, _) in windows:
            r = by[f"{v}|{w}"]
            print(f"{v:6} {w:8} {r['ret']:>+9.1f} {r['mdd']:>7.1f} {r['sharpe']:>7.2f} "
                  f"{r['trades']:>5} {len(r['switches']):>6}")
        print("-" * 56)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "isoos":
        main_isoos(sys.argv[2:])
    else:
        main_full()
