"""라이브 엣지부패 모니터 — 매일 1회 (systemd timer), 프로덕션 무수정.

① 신호 패리티: 설정 버전별 anchor부터 현재까지 리플레이 vs 라이브 실체결(trades.csv +
   state.json 보유 포지션) 진입 이벤트 대조. 설명 불가능한 불일치 = WARN.
   (경로의존 때문에 윈도우 분할 금지 — 항상 해당 배포 anchor부터 전체 리플레이)
② 체결 품질: 매칭된 진입의 라이브 vs 리플레이 가격 괴리(bps, +=불리).
   중앙값 > 15bps 또는 단건 > 50bps = WARN. (백테 가정 5bps)
③ 엣지부패: 일별 equity 로그의 30d/90d 수익률을 백테 분포
   (config/edge_baseline_v21d.json) 백분위로 환산. p5 미만 = ALERT.
   로그 31일 누적 전엔 침묵. 수익률만 비교(일별 표본 vs 1h 분포의 Sharpe 왜곡 방지).

자동 행동 없음 — 경보만. 성과 기반 자동 감속은 tail-cut/스트릭 기각 교훈상 금지,
파국 경로는 딥플로어(live_trade)가 담당.

알림: 새 WARN/ALERT 또는 해소 후 재발 시 즉시, 월요일엔 주간 요약.
동일한 과거 이상을 매일 재전송하지 않는다. 실행 실패는 항상 텔레그램.
재배포(config 변경) 시 config hash로 anchor를 자동 갱신하고, live_trade가 시작 상태를
리플레이 시드로 저장한다. 베이스라인 hash가 다르면 즉시 운영 경보를 낸다.

사용:
  python scripts/edge_monitor.py                # 정기 실행 (timer)
  python scripts/edge_monitor.py --force-notify # 요약 강제 발송 (설치 검증용)
  python scripts/edge_monitor.py --no-balance   # 잔고 조회 생략 (로컬 테스트)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import numpy as np
import pandas as pd
import yaml

from data.loader import DataLoader
from portfolio import state_store
from risk.circuit_breaker import BreakerStatus, CircuitBreaker
import scripts.run_backtest as rb

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("edge_monitor")

# ── 기준점 폴백 (최초 마이그레이션에만 사용; 이후 hash별 상태파일 자동관리) ──
ANCHOR = pd.Timestamp("2026-07-03 18:00", tz="UTC")
CONFIG = "config/final_v21d_eexit.yaml"   # 2026-06-19 v21d 배포 (v21c + 반대신호 조기청산)
BASELINE = "config/edge_baseline_v21d.json"  # 생성 전엔 ③ 자동 스킵 (31일 누적 후 발동)
# replay 시작 자본: 신호/슬리피지 패리티엔 무관(포지션 사이즈만 스케일하고, ③ 롤링은
# 실잔고 로그를 씀) → 정밀값 불필요. main에서 라이브 잔고를 API로 긁어 쓰고,
# 못 긁으면(--no-balance/조회 실패) 이 폴백 사용.
ANCHOR_CAPITAL_FALLBACK = 10000.0

# ── 파일 경로 ─────────────────────────────────────────────────────
CACHE_DIR = Path("data/edge_cache")
EQUITY_LOG = Path("data/edge_equity_log.csv")
TRADES_CSV = Path("trades.csv")
STATE_JSON = Path("data/state.json")
ALERT_STATE_JSON = Path("data/edge_alert_state.json")
ANCHOR_STATE_JSON = Path("data/edge_replay_anchor.json")
REPLAY_SEED_JSON = Path("data/live_replay_seed.json")

# ── 사전선언 임계 (사후 완화 금지) ────────────────────────────────
SLIP_MED_WARN_BPS = 15.0   # 매칭 진입 슬리피지 중앙값
SLIP_MAX_WARN_BPS = 50.0   # 단건 최대
PCTL_ALERT = 5.0           # 롤링 수익률 백테 백분위
MATCH_TOL = pd.Timedelta(hours=1)

_TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
_PREFETCH_BARS = 320  # 라이브 lookback 300 + 여유


def _load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bar_boundary(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    ts = _ts(ts)
    return ts.ceil(timeframe)


def resolve_anchor(
    config_path: Path,
    *,
    timeframe: str,
    state_path: Path = ANCHOR_STATE_JSON,
    seed_path: Path = REPLAY_SEED_JSON,
    fallback: pd.Timestamp = ANCHOR,
    explicit: str | None = None,
) -> tuple[pd.Timestamp, bool]:
    """설정 버전별 리플레이 기준점을 보존한다.

    같은 설정이면 저장된 기준점을 재사용하고, 설정 내용이 바뀌면 파일 배포시각의
    다음 완성 봉부터 새 리플레이로 전환한다. 구 설정 라이브 이력을 현 설정으로
    덮어 재생하던 가장 큰 패리티 오염 원인을 제거한다.
    """
    digest = _config_sha256(config_path)
    previous = None
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text())
        except (OSError, ValueError, TypeError) as e:
            logger.warning("리플레이 기준점 상태 손상 — 재생성: %s", e)

    seed_anchor = None
    if seed_path.exists():
        try:
            seed_data = json.loads(seed_path.read_text())
            if seed_data.get("config_sha256") == digest:
                seed_anchor = _ts(seed_data["anchor"])
        except (OSError, ValueError, TypeError, KeyError) as e:
            logger.warning("리플레이 시드 기준점 읽기 실패: %s", e)

    if explicit:
        anchor = _bar_boundary(pd.Timestamp(explicit), timeframe)
    elif seed_anchor is not None and (
        not previous or seed_anchor > _ts(previous.get("anchor", fallback))
    ):
        # config가 같아도 라이브 프로세스 재시작 시 새 시작상태를 기준으로 재앵커한다.
        anchor = seed_anchor
    elif previous and previous.get("config_sha256") == digest:
        anchor = _ts(previous["anchor"])
    else:
        deployed_at = pd.Timestamp(config_path.stat().st_mtime, unit="s", tz="UTC")
        anchor = max(_ts(fallback), _bar_boundary(deployed_at, timeframe))

    changed = (
        not previous
        or previous.get("config_sha256") != digest
        or _ts(previous.get("anchor", anchor)) != anchor
        or explicit is not None
    )
    payload = {
        "anchor": anchor.isoformat(),
        "config": str(config_path),
        "config_sha256": digest,
        "config_mtime": pd.Timestamp(config_path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp_path.replace(state_path)
    if changed:
        logger.info("리플레이 기준점 갱신: %s (config %s)", anchor, digest[:12])
    return anchor, changed


# ── ① 데이터 갱신 (증분) ──────────────────────────────────────────
def refresh_cache(symbols: list[str], timeframes: list[str], anchor: pd.Timestamp) -> None:
    ex = ccxt.binanceusdm({"enableRateLimit": True})
    now_ms = ex.milliseconds()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        for tf in timeframes:
            path = CACHE_DIR / f"ohlcv_{sym}_{tf}.parquet"
            tf_ms = _TF_MS[tf]
            old = pd.read_parquet(path) if path.exists() else None
            since_ms = (int(old["timestamp"].max().timestamp() * 1000) + tf_ms
                        if old is not None and len(old)
                        else int(anchor.timestamp() * 1000) - _PREFETCH_BARS * tf_ms)
            rows = []
            while since_ms < now_ms:
                batch = ex.fetch_ohlcv(sym, tf, since=since_ms, limit=1000)
                if not batch:
                    break
                rows.extend(batch)
                since_ms = batch[-1][0] + tf_ms
                if len(batch) < 1000:
                    break
            if rows:
                new = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True)
                # 미완성 봉 제거 (봉시작 + tf가 현재 이후 = 아직 진행 중)
                new = new[new["timestamp"] + pd.Timedelta(milliseconds=tf_ms)
                          <= pd.Timestamp.now(tz="UTC")]
                df = (pd.concat([old, new]) if old is not None else new)
                df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                df.to_parquet(path)
        # 펀딩 (scorer confluence 점수 + 펀딩 시뮬에 필요)
        fpath = CACHE_DIR / f"funding_{sym}_8h.parquet"
        fold = pd.read_parquet(fpath) if fpath.exists() else None
        fsince = (int(fold["timestamp"].max().timestamp() * 1000) + 1
                  if fold is not None and len(fold)
                  else int(anchor.timestamp() * 1000) - 3 * 86_400_000)
        frows = []
        while True:
            batch = ex.fetch_funding_rate_history(sym, since=fsince, limit=1000)
            if not batch:
                break
            frows.extend(batch)
            fsince = batch[-1]["timestamp"] + 1
            if len(batch) < 1000:
                break
        if frows:
            fnew = pd.DataFrame([{
                "timestamp": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                "symbol": r.get("symbol", sym),
                "rate": float(r.get("fundingRate") or 0.0),
            } for r in frows])
            fdf = (pd.concat([fold, fnew]) if fold is not None else fnew)
            fdf = fdf.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            fdf.to_parquet(fpath)
    logger.info("캐시 갱신 완료 (%d심볼 × %s)", len(symbols), timeframes)


# ── ② 리플레이 ────────────────────────────────────────────────────
def load_replay_seed(
    config_path: Path,
    anchor: pd.Timestamp,
    seed_path: Path = REPLAY_SEED_JSON,
) -> tuple[Path | None, float | None]:
    if not seed_path.exists():
        return None, None
    try:
        data = json.loads(seed_path.read_text())
        if data.get("config_sha256") != _config_sha256(config_path):
            logger.warning("리플레이 시드 설정 버전 불일치 — flat 시작")
            return None, None
        if _ts(data["anchor"]) != _ts(anchor):
            logger.warning("리플레이 시드 anchor 불일치 — flat 시작")
            return None, None
        return seed_path, float(data["equity"])
    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.warning("리플레이 시드 손상 — flat 시작: %s", e)
        return None, None


def run_replay(
    p: dict,
    capital: float,
    anchor: pd.Timestamp,
    seed_path: Path | None = None,
):
    loader = DataLoader(
        symbols=p["symbols"], timeframes=p["timeframes"],
        primary_tf=p.get("primary_timeframe", "1h"),
        cache_dir=str(CACHE_DIR),
        lookback=p.get("data", {}).get("lookback_bars", 300),
    )
    engine = rb.build_engine(p, capital)
    if seed_path is not None:
        seed_meta = json.loads(seed_path.read_text())
        seed = state_store.load(seed_path)
        if seed is None:
            raise RuntimeError(f"검증된 리플레이 시드를 읽을 수 없음: {seed_path}")
        engine.tracker.state = seed
        state_store.restore_runtime(engine, path=seed_path)
        if seed_meta.get("last_day"):
            engine._last_day = _ts(seed_meta["last_day"])
        if seed_meta.get("session_started_at"):
            engine.funding_sim.sync_to(_ts(seed_meta["session_started_at"]))
        logger.info("리플레이 라이브 시작상태 복원: equity=%.2f, positions=%d",
                    seed.equity, len(seed.positions))
    # loader.iterate의 since는 봉 open 기준, 스냅샷 timestamp는 close 기준(open+1h).
    # 라이브가 ANCHOR 정각에 처리한 스냅샷(= open ANCHOR-1h 봉)을 포함하려면 1h 앞당김.
    primary_delta = pd.Timedelta(p.get("primary_timeframe", "1h"))
    engine.run(loader.iterate(since=anchor - primary_delta))
    return engine


# ── ③ 진입 이벤트 수집 ────────────────────────────────────────────
_EVENT_COLS = ["symbol", "strategy", "direction", "entry_time", "entry_price",
               "status", "exit_time", "exit_price", "exit_reason"]


def live_entries(anchor: pd.Timestamp) -> pd.DataFrame:
    rows = []
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        for _, r in df.iterrows():
            et = pd.Timestamp(r["entry_time"])
            et = et.tz_localize("UTC") if et.tzinfo is None else et.tz_convert("UTC")
            if et < anchor:
                continue
            xt = pd.Timestamp(r["exit_time"])
            xt = xt.tz_localize("UTC") if xt.tzinfo is None else xt.tz_convert("UTC")
            rows.append(dict(symbol=r["symbol"], strategy=r["strategy"],
                             direction=r["direction"], entry_time=et,
                             entry_price=float(r["entry_price"]), status="closed",
                             exit_time=xt, exit_price=float(r["exit_price"]),
                             exit_reason=r.get("exit_reason", "")))
    if STATE_JSON.exists():
        data = json.loads(STATE_JSON.read_text())
        for sym, pos in data.get("positions", {}).items():
            ot = pd.Timestamp(pos["opened_at"])
            ot = ot.tz_localize("UTC") if ot.tzinfo is None else ot.tz_convert("UTC")
            if ot < anchor:
                continue
            rows.append(dict(symbol=sym, strategy=pos["strategy"],
                             direction=pos["direction"], entry_time=ot,
                             entry_price=float(pos["entry_price"]), status="open",
                             exit_time=None, exit_price=None, exit_reason=None))
    return pd.DataFrame(rows, columns=_EVENT_COLS)


def replay_entries(engine, anchor: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for r in engine.ledger.records:
        if _ts(r.entry_time) < anchor:
            continue  # 시드로 이월된 포지션은 경로에는 반영하되 신규 진입 비교에서 제외
        rows.append(dict(symbol=r.symbol, strategy=r.strategy, direction=r.direction,
                         entry_time=r.entry_time, entry_price=r.entry_price,
                         status="closed", exit_time=r.exit_time,
                         exit_price=r.exit_price, exit_reason=r.exit_reason))
    for sym, pos in engine.tracker.snapshot().positions.items():
        if _ts(pos.opened_at) < anchor:
            continue
        rows.append(dict(symbol=sym, strategy=pos.strategy, direction=pos.direction,
                         entry_time=pos.opened_at, entry_price=pos.entry_price,
                         status="open", exit_time=None, exit_price=None, exit_reason=None))
    return pd.DataFrame(rows, columns=_EVENT_COLS)


# ── ④ 대조 ────────────────────────────────────────────────────────
def match_events(live: pd.DataFrame, rep: pd.DataFrame):
    """동일 이벤트 후보 전체에서 시간차가 가장 작은 쌍부터 매칭한다.

    CSV 행 순서에 따라 첫 후보를 집던 기존 방식은 같은 전략이 연속 진입하면 더 먼
    이벤트끼리 붙여 가짜 live-only/replay-only를 만들 수 있었다.
    """
    live_rows = [r for _, r in live.iterrows()]
    rep_rows = [r for _, r in rep.iterrows()]
    candidates = []
    for li, lr in enumerate(live_rows):
        for ri, rr in enumerate(rep_rows):
            delta = abs(lr["entry_time"] - rr["entry_time"])
            if (lr["symbol"] == rr["symbol"] and lr["strategy"] == rr["strategy"]
                    and lr["direction"] == rr["direction"] and delta <= MATCH_TOL):
                candidates.append((delta, li, ri))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    live_used, rep_used = set(), set()
    matched = []
    for _, li, ri in candidates:
        if li in live_used or ri in rep_used:
            continue
        live_used.add(li)
        rep_used.add(ri)
        matched.append((live_rows[li], rep_rows[ri]))
    live_only = [r for i, r in enumerate(live_rows) if i not in live_used]
    rep_only = [r for i, r in enumerate(rep_rows) if i not in rep_used]
    return matched, live_only, rep_only


def slippage_bps(live_row, rep_row) -> float:
    """진입가 괴리 bps. + = 라이브가 불리한 방향."""
    raw = (live_row["entry_price"] - rep_row["entry_price"]) / rep_row["entry_price"] * 1e4
    return raw if live_row["direction"] == "long" else -raw


# ── ④-b 미체결 사유 분류 (경로분기 vs 미상) ───────────────────────
# rep_only(라이브 미체결) 각 건을, 그 entry_time 시점 라이브 북을 복원해
# 엔진(_process_bar)과 동일한 게이트 순서로 차단 사유를 추정한다. 전부 설명되면
# 경로분기(리플레이=앵커 flat 시작 vs 라이브=이월 포지션·실체결 경로) → 양성이라
# WARN 억제. 하나라도 '미상'이면 진짜 누락 가능성 → WARN. (자동 행동 없음, 진단만)
def _ts(v) -> pd.Timestamp:
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _gate_cfg(p: dict) -> dict:
    """엔진 진입 게이트 임계 (config 단일 출처 — 하드코딩 금지)."""
    r = p.get("risk", {})
    return dict(
        max_positions=int(r.get("max_positions", 10)),
        max_same_direction=int(r.get("max_same_direction", 4)),
        corr_thr=float(r.get("correlation_block_threshold", 0.8)),
        corr_lookback=int(r.get("correlation_lookback", 100)),
        cooldown_h=float(r.get("tp_cooldown_hours", 0.0)),
        pause_losses=int(r.get("circuit_breaker_pause_losses", 5)),
        stop_losses=int(r.get("circuit_breaker_stop_losses", 10)),
        pause_hours=int(r.get("circuit_breaker_pause_hours", 48)),
        iso=p.get("strategy_guard_isolated", {}) or {},
    )


def live_book_history() -> list[dict]:
    """라이브 전체 포지션 이력 (앵커 무관) — 북 복원 + CB 재구성용.
    trades.csv 청산분(pnl·exit_reason 포함) + state.json 미청산분. 앵커 이전
    이월 포지션도 포함해야 그 시각 상관·심볼점유 판정이 라이브와 일치한다."""
    rows = []
    if TRADES_CSV.exists():
        df = pd.read_csv(TRADES_CSV)
        for _, r in df.iterrows():
            rows.append(dict(
                symbol=r["symbol"], strategy=r["strategy"], direction=r["direction"],
                entry_time=_ts(r["entry_time"]), exit_time=_ts(r["exit_time"]),
                exit_reason=str(r.get("exit_reason", "") or ""),
                pnl=float(r.get("pnl", 0) or 0)))
    if STATE_JSON.exists():
        data = json.loads(STATE_JSON.read_text())
        for sym, pos in data.get("positions", {}).items():
            rows.append(dict(
                symbol=sym, strategy=pos["strategy"], direction=pos["direction"],
                entry_time=_ts(pos["opened_at"]), exit_time=None,
                exit_reason="", pnl=None))
    return rows


def _open_at(history: list[dict], t: pd.Timestamp) -> list[dict]:
    """t 시점 라이브 보유 (진입 < t, 미청산 또는 청산 > t). 엔진은 청산(2·4단계)을
    진입(6단계)보다 먼저 처리하므로 동봉 청산분(exit==t)은 제외한다."""
    return [h for h in history
            if h["entry_time"] < t and (h["exit_time"] is None or h["exit_time"] > t)]


class _CorrCache:
    """edge_cache 1h 종가로 임의 시점 상관 — 엔진 corr_filter와 동일(1h 수익률 lookback개)."""
    def __init__(self, lookback: int) -> None:
        self.lookback = lookback
        self._closes: dict = {}

    def _close(self, sym: str):
        if sym not in self._closes:
            path = CACHE_DIR / f"ohlcv_{sym}_1h.parquet"
            if path.exists():
                s = pd.read_parquet(path).set_index("timestamp")["close"]
                s.index = pd.to_datetime(s.index, utc=True)
                self._closes[sym] = s
            else:
                self._closes[sym] = None
        return self._closes[sym]

    def corr(self, a: str, b: str, t: pd.Timestamp):
        sa, sb = self._close(a), self._close(b)
        if sa is None or sb is None:
            return None
        ra = sa[sa.index < t].pct_change().dropna().iloc[-self.lookback:]
        rb_ = sb[sb.index < t].pct_change().dropna().iloc[-self.lookback:]
        m = pd.DataFrame({"a": ra, "b": rb_}).dropna()
        if len(m) < 10:
            return None
        return abs(float(m["a"].corr(m["b"])))


def _cb_status_at(history, strategy, t, iso_set, g):
    """closed trades(앵커 무관·격리 제외)를 시간순 record_result로 재생 → t 시점 CB 상태.
    임계 도달 직후 get_status(청산시각)를 호출해 paused_until 설정을 모사한다
    (승 1건이 48h 정지창을 못 지우는 엔진 동작과 일치). 승=실현 pnl>0."""
    cb = CircuitBreaker(strategy_pause_losses=g["pause_losses"],
                        global_stop_losses=g["stop_losses"],
                        pause_duration_hours=g["pause_hours"])
    closed = sorted((h for h in history
                     if h["exit_time"] is not None and h["strategy"] not in iso_set),
                    key=lambda h: h["exit_time"])
    for h in closed:
        if h["exit_time"] >= t:
            break
        cb.record_result(h["strategy"], (h["pnl"] or 0) > 0, h["exit_time"])
        cb.get_status(h["strategy"], h["exit_time"])
    return cb.get_status(strategy, t)


def classify_miss(rr, history, corr_cache, g) -> tuple[str, bool]:
    """라이브 미체결 1건의 차단 사유 → (사유, 양성여부). 엔진 게이트 순서 준수."""
    t, sym, strat, dirn = rr["entry_time"], rr["symbol"], rr["strategy"], rr["direction"]
    iso_set = set(g["iso"].keys())
    book = _open_at(history, t)
    # 1) 심볼 점유 (격리 포함 — 엔진은 전체 state.positions로 막는다)
    occ = next((h for h in book if h["symbol"] == sym), None)
    if occ is not None:
        return f"심볼점유({occ['direction']} {occ['strategy']} 보유중)", True
    if strat in iso_set:  # 격리 북: 자체 슬롯/방향 한도만 (corr·글로벌가드 면제)
        ic = g["iso"].get(strat, {})
        own = [h for h in book if h["strategy"] == strat]
        if len(own) >= int(ic.get("max_positions", 9999)):
            return f"격리북 슬롯풀({len(own)}/{ic.get('max_positions')})", True
        if sum(1 for h in own if h["direction"] == dirn) >= int(ic.get("max_same_direction", 9999)):
            return "격리북 방향한도", True
        return "미상(점검필요)", False
    non_iso = [h for h in book if h["strategy"] not in iso_set]
    # 2) CB (전략 연속손절 정지 / 전체 중단)
    cb = _cb_status_at(history, strat, t, iso_set, g)
    if cb == BreakerStatus.STOPPED:
        return "CB 전체중단", True
    if cb == BreakerStatus.PAUSED:
        return f"CB 전략정지({strat} 연속손절)", True
    # 3) 슬롯 / 방향 한도 (비격리 가드뷰 기준)
    if len(non_iso) >= g["max_positions"]:
        return f"슬롯풀({len(non_iso)}/{g['max_positions']})", True
    same = sum(1 for h in non_iso if h["direction"] == dirn)
    if same >= g["max_same_direction"]:
        return f"방향한도({same}/{g['max_same_direction']} {dirn})", True
    # 4) TP 쿨다운
    if g["cooldown_h"] > 0:
        for h in history:
            if (h["symbol"] == sym and h["strategy"] == strat and h["exit_reason"] == "tp"
                    and h["exit_time"] is not None and h["exit_time"] < t
                    and (t - h["exit_time"]) <= pd.Timedelta(hours=g["cooldown_h"])):
                return "TP쿨다운", True
    # 5) 상관차단 (동방향 비격리 보유분 중 |corr|≥임계)
    best = None
    for h in non_iso:
        if h["direction"] != dirn:
            continue
        c = corr_cache.corr(sym, h["symbol"], t)
        if c is not None and c >= g["corr_thr"] and (best is None or c > best[1]):
            best = (h["symbol"], c)
    if best is not None:
        return f"상관차단({best[0]} {best[1]:.2f}≥{g['corr_thr']:.2f})", True
    return "미상(점검필요)", False


def _timing_pairs(rep_only, live_only) -> dict:
    """rep_only↔live_only 동일(심볼·전략·방향) 근접진입(1h<Δ≤36h)=타이밍시프트 페어.
    id(row)→상대 진입시각 문자열."""
    out, rep_used, live_used = {}, set(), set()
    candidates = []
    for ri, rr in enumerate(rep_only):
        for li, lr in enumerate(live_only):
            delta = abs(lr["entry_time"] - rr["entry_time"])
            if (lr["symbol"] == rr["symbol"] and lr["strategy"] == rr["strategy"]
                    and lr["direction"] == rr["direction"]
                    and MATCH_TOL < delta <= pd.Timedelta(hours=36)):
                candidates.append((delta, ri, li))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, ri, li in candidates:
        if ri in rep_used or li in live_used:
            continue
        rep_used.add(ri)
        live_used.add(li)
        rr, lr = rep_only[ri], live_only[li]
        out[id(rr)] = f"{lr['entry_time']:%m-%d %H:%M}"
        out[id(lr)] = f"{rr['entry_time']:%m-%d %H:%M}"
    return out


# ── ⑤ 일별 equity 로그 + 롤링 백분위 ──────────────────────────────
def fetch_usdt_balance() -> float | None:
    """라이브 USDT 총잔고. 조회 실패 시 None."""
    try:
        ex = ccxt.binanceusdm({
            "apiKey": os.environ.get("BINANCE_API_KEY", ""),
            "secret": os.environ.get("BINANCE_SECRET", ""),
            "enableRateLimit": True,
        })
        return float(ex.fetch_balance().get("USDT", {}).get("total", 0) or 0) or None
    except Exception:
        logger.warning("잔고 조회 실패 — replay 폴백 자본 사용")
        return None


def append_equity_log(usdt: float | None) -> None:
    if usdt is None:
        return  # --no-balance or 조회 실패
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    if EQUITY_LOG.exists():
        with open(EQUITY_LOG) as f:
            if any(row.startswith(today) for row in f):
                return  # 오늘 이미 기록
    EQUITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    new_file = not EQUITY_LOG.exists()
    with open(EQUITY_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "equity"])
        w.writerow([today, f"{usdt:.2f}"])
    logger.info("equity 로그: %s $%.2f", today, usdt)


def rolling_percentiles(
    *,
    config_path: Path = Path(CONFIG),
    baseline_path: Path = Path(BASELINE),
) -> tuple[list[str], set[str]]:
    """30d/90d 백분위와 운영 결함을 함께 반환한다."""
    if not baseline_path.exists():
        return [f"③ 엣지부패: 베이스라인 없음 ({baseline_path}) ⚠️"], {"baseline:missing"}
    try:
        baseline = json.loads(baseline_path.read_text())
    except (OSError, ValueError, TypeError) as e:
        return [f"③ 엣지부패: 베이스라인 손상 ({type(e).__name__}) ⚠️"], {"baseline:invalid"}
    expected_hash = _config_sha256(config_path)
    if baseline.get("config_sha256") != expected_hash:
        return ["③ 엣지부패: 현재 설정과 베이스라인 버전 불일치 ⚠️"], {"baseline:stale"}
    if not EQUITY_LOG.exists():
        return ["③ 롤링: equity 로그 0/31일 — 누적 중"], set()
    eq = pd.read_csv(EQUITY_LOG, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(eq) < 31:
        return [f"③ 롤링: 표본 {len(eq)}/31일 — 누적 중"], set()
    grids = baseline["grids"]
    last_date = eq["date"].iloc[-1]
    last_eq = eq["equity"].iloc[-1]
    out = []
    for name, days in (("30d", 30), ("90d", 90)):
        # 행 위치가 아닌 '날짜 기준' lookback — 로그에 결손일(잔고조회 실패/서버 다운/
        # --no-balance)이 섞이면 positional iloc은 30/90일보다 더 과거를 참조해
        # 수익률·백분위가 왜곡된다(baseline은 연속 1h봉 기준이라 호라이즌 불일치).
        target = last_date - pd.Timedelta(days=days)
        prior = eq[eq["date"] <= target]
        if prior.empty:
            continue  # 아직 days일 치 히스토리 없음
        base_eq = prior["equity"].iloc[-1]
        ret = last_eq / base_eq - 1
        g = grids[f"ret_{name}"]
        pctl = float(np.interp(ret, g["values"], g["percentiles"]))
        flag = " 🚨ALERT" if pctl < PCTL_ALERT else ""
        out.append(f"③ {name} 수익률 {ret:+.1%} = 백테 p{pctl:.0f}{flag}")
    return out, set()


# ── ⑥ 경보 상태 전이/중복 억제 ───────────────────────────────────
def should_notify_alert_transition(
    active_alerts: set[str],
    *,
    is_monday: bool = False,
    force: bool = False,
    state_path: Path = ALERT_STATE_JSON,
    persist: bool = True,
) -> bool:
    """신규·해소를 포함한 경보 상태 변화 여부를 판정한다.

    edge replay는 ANCHOR부터 전구간을 매일 다시 계산하므로, 과거 불일치 한 건도
    해소될 때까지 매일 WARN으로 재검출된다. 경보의 존재가 아니라 상태 전이
    (신규/해소 후 재발)를 알림 조건으로 삼아 경보 피로를 막는다.

    초기 실행에서 발견된 경보와 해소도 알린다. ``persist=False``는 실제 Telegram
    전달 성공 뒤 상태를 확정해야 하는 단발성 모니터가 사용한다.
    """
    state_exists = state_path.exists()
    previous: set[str] = set()
    if state_exists:
        try:
            raw = json.loads(state_path.read_text())
            previous = set(raw.get("active_alerts", []))
        except (OSError, ValueError, TypeError) as e:
            # 손상/읽기 실패는 신규 경보를 침묵시키지 않도록 빈 상태로 간주한다.
            logger.warning("edge 경보 상태 읽기 실패 — 신규 상태로 복구: %s", e)

    changed = active_alerts != previous
    if persist:
        save_alert_state(active_alerts, state_path=state_path)

    if force or is_monday:
        return True
    if not state_exists:
        logger.info("edge 경보 상태 생성: 활성 %d건", len(active_alerts))
    return changed


def save_alert_state(active_alerts: set[str], *, state_path: Path = ALERT_STATE_JSON) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    payload = {
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "active_alerts": sorted(active_alerts),
    }
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    tmp_path.replace(state_path)


# ── 메인 ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-notify", action="store_true")
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--anchor", help="리플레이 기준 UTC 시각 강제 지정(다음 봉 경계로 정렬)")
    args = parser.parse_args()
    _load_env()

    from execution.notifier import TelegramNotifier
    notifier = TelegramNotifier.from_env()

    try:
        with open(CONFIG) as f:
            p = yaml.safe_load(f)
        config_path = Path(CONFIG)
        primary_tf = p.get("primary_timeframe", "1h")
        anchor, anchor_changed = resolve_anchor(
            config_path, timeframe=primary_tf, explicit=args.anchor,
        )
        usdt = None if args.no_balance else fetch_usdt_balance()
        seed_path, seed_capital = load_replay_seed(config_path, anchor)
        capital = seed_capital or usdt or ANCHOR_CAPITAL_FALLBACK
        refresh_cache(p["symbols"], p["timeframes"], anchor)
        engine = run_replay(p, capital, anchor, seed_path=seed_path)
        live = live_entries(anchor)
        rep = replay_entries(engine, anchor)
        matched, live_only, rep_only = match_events(live, rep)
        append_equity_log(usdt)

        lines, severity = [], "OK"
        active_alerts: set[str] = set()
        days = max(0, (pd.Timestamp.now(tz="UTC") - anchor).days)
        seed_label = "live-state" if seed_path else "flat"
        lines.append(f"기간 {anchor:%Y-%m-%d %H:%M}~ ({days}일) | "
                     f"라이브 {len(live)}건 / 리플레이 {len(rep)}건 | 시드 {seed_label}")
        if anchor_changed:
            lines.append("ℹ️ 설정 변경 감지: 리플레이 기준점 자동 갱신")

        # ① 신호 패리티 — 불일치 각 건을 그 시각 라이브 북 복원 + 엔진 게이트 체인으로
        #    사유 분류. 전부 경로분기(양성)면 WARN 억제, '미상'이 하나라도 있으면 WARN.
        if live_only or rep_only:
            history = live_book_history()
            gcfg = _gate_cfg(p)
            corr_cache = _CorrCache(gcfg["corr_lookback"])
            timing = _timing_pairs(rep_only, live_only)
            ev_lines, rep_unexplained, live_unexplained = [], 0, 0
            for rr in rep_only:
                pair = timing.get(id(rr))
                if pair is not None:
                    reason, benign = f"타이밍시프트(라이브 {pair} 진입)", True
                else:
                    reason, benign = classify_miss(rr, history, corr_cache, gcfg)
                rep_unexplained += 0 if benign else 1
                if not benign:
                    active_alerts.add(
                        f"parity:{rr['symbol']}:{rr['direction']}:{rr['strategy']}:"
                        f"{rr['entry_time'].isoformat()}"
                    )
                ev_lines.append(f"① 라이브 미체결: {rr['symbol']} {rr['direction']} "
                                f"{rr['strategy']} @{rr['entry_time']:%m-%d %H:%M} → {reason}")
            for lr in live_only:
                pair = timing.get(id(lr))
                benign = pair is not None
                reason = (f"타이밍시프트(리플레이 {pair} 진입)" if benign
                          else "미상(리플레이 신호 미발생)")
                if not benign:
                    live_unexplained += 1
                    active_alerts.add(
                        f"live_only:{lr['symbol']}:{lr['direction']}:{lr['strategy']}:"
                        f"{lr['entry_time'].isoformat()}"
                    )
                ev_lines.append(f"① 라이브 전용: {lr['symbol']} {lr['direction']} "
                                f"{lr['strategy']} @{lr['entry_time']:%m-%d %H:%M} → {reason}")
            unexplained = rep_unexplained + live_unexplained
            if unexplained:
                severity = "WARN"
            benign_cnt = len(rep_only) - rep_unexplained
            lines.append(f"① 패리티: 미체결 {len(rep_only)}건(경로분기 {benign_cnt}/"
                         f"미상 {rep_unexplained}) · 라이브전용 {len(live_only)}건"
                         f"(미상 {live_unexplained})"
                         + (" ⚠️미상 점검필요" if unexplained else " ✓경로분기"))
            max_details = 30
            lines.extend(ev_lines[:max_details])
            if len(ev_lines) > max_details:
                lines.append(f"① 상세 {len(ev_lines) - max_details}건 생략 (콘솔 로그 확인)")
        else:
            lines.append(f"① 신호 패리티: {len(matched)}/{len(matched)} 일치 ✓")

        # ② 체결 품질
        if matched:
            slips = [slippage_bps(l, r) for l, r in matched]
            med, mx = float(np.median(slips)), float(max(slips, key=abs))
            note = ""
            # 중앙값 기준은 표본 5건부터 (n=2로 매일 같은 WARN 반복 = 경보 피로 방지).
            # 단건 50bps 초과는 즉시 발동.
            if ((len(slips) >= 5 and abs(med) > SLIP_MED_WARN_BPS)
                    or abs(mx) > SLIP_MAX_WARN_BPS):
                severity = "WARN" if severity == "OK" else severity
                note = " ⚠️"
                if len(slips) >= 5 and abs(med) > SLIP_MED_WARN_BPS:
                    active_alerts.add("slippage:median")
                if abs(mx) > SLIP_MAX_WARN_BPS:
                    active_alerts.add("slippage:max")
            lines.append(f"② 진입 슬리피지(n={len(slips)}): 중앙값 {med:+.1f}bps / "
                         f"최대 {mx:+.1f}bps (가정 5bps){note}")

        # ③ 엣지부패
        roll, baseline_alerts = rolling_percentiles(config_path=config_path)
        active_alerts.update(baseline_alerts)
        if baseline_alerts:
            severity = "WARN" if severity == "OK" else severity
        if any("ALERT" in s for s in roll):
            severity = "ALERT"
            for s in roll:
                if "ALERT" in s:
                    active_alerts.add(f"edge_decay:{s.split()[1]}")
        lines.extend(roll)

        head = {"OK": "🩺", "WARN": "⚠️", "ALERT": "🚨"}[severity]
        body = f"{head} <b>엣지 모니터 [{severity}]</b>\n" + "\n".join(lines)
        print(body)
        is_monday = pd.Timestamp.now(tz="UTC").weekday() == 0
        should_notify = should_notify_alert_transition(
            active_alerts, is_monday=is_monday, force=args.force_notify, persist=False,
        )
        if should_notify:
            if notifier.enabled and notifier.send_and_wait(body, timeout=30):
                save_alert_state(active_alerts)
            elif notifier.enabled:
                logger.error("edge 경보 Telegram 전달 실패 — 상태 미확정, 다음 실행에서 재시도")
            else:
                logger.warning("edge 경보 전송 필요하지만 Telegram 비활성 — 상태 미확정")
        else:
            save_alert_state(active_alerts)
    except Exception as e:
        logger.exception("edge_monitor 실패")
        if notifier.enabled:
            notifier.send_and_wait(
                f"🔧 edge_monitor 실행 실패: {type(e).__name__}: {str(e)[:200]}",
                timeout=30,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
