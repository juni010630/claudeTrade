"""주간 국면 스위치 Phase 0/1 — 진단 + 2계좌 주간 리밸 합성 (프로덕션 무수정).

Phase 0: ETHUSDT 1d 5지표 투표(regime/realtime_switch)를 매주 월요일 00:00 UTC
경계에서 인과 판정(닫힌 봉만) → 다음 주 추세그룹 vs 슬리브그룹 수익 스프레드 예측력 진단.
  게이트 G0 (사전등록): 추세주 스프레드 평균>0 AND 횡보주<0(전구간),
  각 측은 해당 판정 주 4개 이상인 연도 중 >=3년 방향 일치. 불통과 = 실험 종결.

Phase 1 (G0 통과 시에만): 주간 자본 이동 2계좌 합성.
  HARD(추세 100:0 / 횡보 0:100 / 중립 50:50), TILT(70:30 / 30:70 / 50:50)
  vs 정적 50:50, 40:60. IS(2022~2024) pick → OOS(2025~) 파레토 게이트 G1.

솔로 계좌 = final_v17.yaml에서 반대 그룹 enabled:false, 활성 그룹 capital_fraction 1.0
(병합 0.5×총자본 = 솔로 1.0×반쪽계좌, 사이징 동일), deep_floor 해제(계좌 분리라 별도 관리).

사용: python scripts/weekly_regime_diag.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yaml

CFG = "config/final_v17.yaml"
SER_PATH = "data/results/weekly_switch_solo_daily.parquet"
START, END = "2022-01-01", "2026-04-14"
TREND = ["ema_cross", "multi_tf_breakout"]
SLEEVE = ["mean_reversion"]


# ── 솔로 계좌 백테 ────────────────────────────────────────────────────
def run_solo(group: str) -> pd.Series:
    from data.loader import DataLoader
    from scripts.run_backtest import build_engine

    p = yaml.safe_load(open(CFG))
    active = TREND if group == "trend" else SLEEVE
    for name in p["strategies"]:
        p["strategies"][name]["enabled"] = name in active
    p["strategy_capital_fraction"] = {n: 1.0 for n in active}
    p["risk"]["deep_floor_dd"] = None

    loader = DataLoader(symbols=p["symbols"], timeframes=p["timeframes"], primary_tf="1h",
                        cache_dir="data/cache", lookback=300)
    eng = build_engine(p, 100)
    eng.run(loader.iterate(since=pd.Timestamp(START, tz="UTC"),
                           until=pd.Timestamp(END, tz="UTC")))
    return eng.equity_curve.to_series()


def get_series() -> pd.DataFrame:
    if Path(SER_PATH).exists():
        return pd.read_parquet(SER_PATH)
    with ProcessPoolExecutor(max_workers=2) as ex:
        eq_t, eq_s = list(ex.map(run_solo, ["trend", "sleeve"]))
    out = {}
    for k, eq in (("trend", eq_t), ("sleeve", eq_s)):
        d = eq.resample("1D").last().pct_change().fillna(0)
        d.index = d.index.tz_localize(None)
        out[k] = d
    df = pd.concat(out, axis=1).dropna()
    Path("data/results").mkdir(exist_ok=True)
    df.to_parquet(SER_PATH)
    return df


# ── 주간 국면 캘린더 (인과: 경계 시점에 닫힌 1d 봉만) ─────────────────
def build_calendar() -> pd.DataFrame:
    from regime.realtime_switch import RealtimeRegimeSwitch

    d = pd.read_parquet("data/cache/ohlcv_ETHUSDT_1d.parquet")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    if d["timestamp"].dt.tz is not None:
        d["timestamp"] = d["timestamp"].dt.tz_localize(None)
    d = d.sort_values("timestamp").reset_index(drop=True)

    mondays = pd.date_range(START, END, freq="W-MON")
    sw = RealtimeRegimeSwitch()
    rows = []
    for wk in mondays:
        # 1d 봉 timestamp=open. open < wk 인 봉은 wk 시점에 종가 확정(마지막 = 일요일 봉, 월 00:00 마감)
        hist = d[d["timestamp"] < wk].tail(120)
        state = sw.update(hist)
        v = sw.last_verdict
        rows.append({"week": wk, "state": state, "score": v.score, "raw": v.regime})
    cal = pd.DataFrame(rows).set_index("week")
    n_switch = (cal["state"] != cal["state"].shift()).sum() - 1
    print(f"[캘린더] {len(cal)}주 | 분포: {cal['state'].value_counts().to_dict()} | 상태 전환 {n_switch}회")
    return cal


# ── 지표 ──────────────────────────────────────────────────────────────
def stats(daily: pd.Series):
    eq = (1 + daily).cumprod()
    mdd = (eq / eq.cummax() - 1).min() * 100
    sh = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
    yr = daily.groupby(daily.index.year).apply(lambda r: ((1 + r).prod() - 1) * 100)
    return eq.iloc[-1] * 100, sh, mdd, yr


def show(tag: str, daily: pd.Series):
    f, sh, mdd, yr = stats(daily)
    ys = " ".join(f"{int(k)}:{round(v):+}" for k, v in yr.items())
    print(f"  {tag:24} ${f:>9,.0f} Sh{sh:5.2f} MDD{mdd:6.1f}%  {ys}")


# ── Phase 0: 예측력 진단 ─────────────────────────────────────────────
def diagnose(df: pd.DataFrame, cal: pd.DataFrame) -> bool:
    wk_start = df.index - pd.to_timedelta((df.index.weekday) % 7, unit="D")
    wret = (1 + df).groupby(wk_start).prod() - 1  # 주간 수익률 (월요일 시작 키)
    j = wret.join(cal[["state"]], how="inner")
    j["spread"] = j["trend"] - j["sleeve"]

    print("\n=== Phase 0: 이번 주 판정 → 같은 주 그룹 스프레드(trend−sleeve) ===")
    ok_sides = {}
    for side, want_pos in (("추세장", True), ("횡보장", False)):
        sub = j[j["state"] == side]
        m = sub["spread"].mean() * 100
        full_ok = (m > 0) == want_pos
        yr_means = sub.groupby(sub.index.year)["spread"].agg(["mean", "count"])
        yr_q = yr_means[yr_means["count"] >= 4]
        yr_ok = int(((yr_q["mean"] > 0) == want_pos).sum())
        ok_sides[side] = full_ok and yr_ok >= min(3, len(yr_q))
        detail = " ".join(f"{y}:{r['mean']*100:+.2f}%({int(r['count'])}주)" for y, r in yr_means.iterrows())
        print(f"  {side} {len(sub)}주: 평균 스프레드 {m:+.3f}%/주 "
              f"[전구간 {'OK' if full_ok else 'FAIL'}] 연도일치 {yr_ok}/{len(yr_q)}  | {detail}")
    sub_n = j[j["state"] == "전환/중립"]
    if len(sub_n):
        print(f"  중립 {len(sub_n)}주: 스프레드 {sub_n['spread'].mean()*100:+.3f}%/주 (게이트 미적용)")

    passed = all(ok_sides.values())
    print(f"\n[G0] {'통과' if passed else '기각'} — 추세측 {ok_sides['추세장']}, 횡보측 {ok_sides['횡보장']}")
    return passed


# ── Phase 1: 주간 자본 이동 2계좌 합성 ────────────────────────────────
W = {
    "HARD": {"추세장": (1.0, 0.0), "횡보장": (0.0, 1.0), "전환/중립": (0.5, 0.5)},
    "TILT": {"추세장": (0.7, 0.3), "횡보장": (0.3, 0.7), "전환/중립": (0.5, 0.5)},
    "정적50:50": {k: (0.5, 0.5) for k in ("추세장", "횡보장", "전환/중립")},
    "정적40:60": {k: (0.4, 0.6) for k in ("추세장", "횡보장", "전환/중립")},
}


def synth(df: pd.DataFrame, cal: pd.DataFrame, wmap: dict) -> pd.Series:
    """주 경계에서 총자본을 (w_t, w_s)로 재분배, 주중엔 각 계좌 독립 복리."""
    wk_start = df.index - pd.to_timedelta(df.index.weekday % 7, unit="D")
    states = cal["state"].reindex(pd.Index(wk_start).unique()).ffill().fillna("전환/중립")
    out = []
    total = 1.0
    for wk, g in df.groupby(wk_start):
        wt, ws = wmap[states.loc[wk]]
        ct = (1 + g["trend"]).cumprod()
        cs = (1 + g["sleeve"]).cumprod()
        eq = total * (wt * ct + ws * cs)
        out.append(eq)
        total = eq.iloc[-1]
    eq = pd.concat(out)
    return eq.pct_change().fillna(eq.iloc[0] - 1.0 if len(eq) else 0)


def phase1(df: pd.DataFrame, cal: pd.DataFrame):
    print("\n=== Phase 1: 주간 리밸 합성 (전구간) ===")
    daily = {tag: synth(df, cal, wmap) for tag, wmap in W.items()}
    for tag, r in daily.items():
        show(tag, r)

    print("\n=== IS(2022~2024) / OOS(2025~) ===")
    is_stats, oos_stats = {}, {}
    for tag, r in daily.items():
        ris, roos = r[r.index < "2025-01-01"], r[r.index >= "2025-01-01"]
        is_stats[tag] = stats(ris)[:3]
        oos_stats[tag] = stats(roos)[:3]
        print(f"  {tag:24} IS  ${is_stats[tag][0]:>8,.0f} Sh{is_stats[tag][1]:5.2f} MDD{is_stats[tag][2]:6.1f}%"
              f" | OOS ${oos_stats[tag][0]:>8,.0f} Sh{oos_stats[tag][1]:5.2f} MDD{oos_stats[tag][2]:6.1f}%")

    # G1: IS에서 HARD/TILT 중 Sharpe 우세 변형 → OOS에서 50:50 파레토 + 40:60 비열위
    pick = max(("HARD", "TILT"), key=lambda t: is_stats[t][1])
    _, sh_p, mdd_p = oos_stats[pick]
    _, sh_b, mdd_b = oos_stats["정적50:50"]
    _, sh_f, mdd_f = oos_stats["정적40:60"]
    pareto = sh_p > sh_b and mdd_p > mdd_b
    not_worse = sh_p >= sh_f or mdd_p >= mdd_f
    print(f"\n[G1] IS pick={pick} → OOS: vs50:50 파레토 {'OK' if pareto else 'FAIL'}"
          f" (Sh {sh_p:.2f} vs {sh_b:.2f}, MDD {mdd_p:.1f} vs {mdd_b:.1f}),"
          f" vs40:60 비열위 {'OK' if not_worse else 'FAIL'}"
          f" → {'통과' if pareto and not_worse else '기각'}")

    # 경계 ±1주 시프트 강건성
    print("\n=== 경계 시프트 (캘린더 ±1주) ===")
    for shift in (-1, 1):
        cal_s = cal.copy()
        cal_s.index = cal_s.index + pd.Timedelta(weeks=shift)
        r = synth(df, cal_s, W[pick])
        show(f"{pick} shift{shift:+d}w", r)


def main():
    cal = build_calendar()
    df = get_series()
    print(f"[솔로 시리즈] {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}일)")
    show("trend 솔로", df["trend"])
    show("sleeve 솔로", df["sleeve"])
    if not diagnose(df, cal):
        print("\nG0 기각 → Phase 1 생략, 실험 종결 (결과 MD 기록 후 종료)")
        return
    phase1(df, cal)


if __name__ == "__main__":
    main()
