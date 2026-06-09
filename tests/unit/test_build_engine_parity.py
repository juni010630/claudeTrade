"""백테(run_backtest)와 라이브(live_trade)의 build_engine 파라미터 패리티 강제.

CLAUDE.md 최우선 규칙 '백테=라이브, config 새 키 추가 시 양쪽 build_engine 모두 반영'을
자동 검증한다. 한쪽 build_engine에만 키를 추가/누락하면 이 테스트가 실패한다.
verify_replay는 백테 내부 dump==replay 결정론만 보므로 이 패리티 갭을 못 잡았다(F13).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from scripts.run_backtest import build_engine as build_bt
from scripts.live_trade import build_engine as build_live
from execution.live_broker import LiveBroker

CONFIG = Path(__file__).resolve().parents[2] / "config" / "final_v17.yaml"


def _params(e) -> dict:
    """엔진에서 config 유래 트레이딩 파라미터만 추출 (notifier/trade_log/broker 등 의도된 차이 제외)."""
    d = {}
    for a in [
        "price_tf", "max_hold_hours", "breakeven_trigger_r", "trailing_r_mult",
        "_strategy_min_score", "_strategy_block_hours", "_strategy_block_symbols",
        "_tier_block_symbols", "_symbol_block_directions", "_strategy_block_tiers",
        "_block_weekdays", "_direction_size_mult", "_strategy_size_penalty",
        "_strategy_size_bonus", "_strategy_size_bonus_mult", "_strategy_capital_fraction",
        "_pyramid_trigger_r", "_pyramid_add_fraction", "_pyramid_max_adds",
        "_pyramid_strategies", "_pyramid_min_score", "_rsi_mom_gate", "_rsi_mom_weight",
        "_rsi_mom_period", "_vol_target", "_vol_scale_min", "_vol_scale_max", "_vol_lookback",
        "_btc_mom_gate", "_btc_mom_opp_w", "_btc_mom_lookback", "_equity_curve_trading",
        "_adx_scaling",
    ]:
        d[a] = getattr(e, a)

    s = e.sizer
    d["sizer"] = (s.risk_per_trade, s._cfg, s.max_notional_usd, s.max_notional_equity_mult)
    d["strategy_sizers"] = {
        k: (sz.risk_per_trade, sz._cfg) for k, sz in e._strategy_sizers.items()
    }

    g = e.guards
    d["guards"] = (g.max_positions, g.max_same_direction, g.daily_pause, g.daily_stop,
                   g.tp_cooldown_hours)

    cb = e.circuit_breaker
    d["cb"] = (cb.strategy_pause_losses, cb.global_stop_losses, cb.pause_duration_hours)

    cf = e.corr_filter
    d["corr"] = (cf.block_threshold, cf.lookback)

    sc = e.scorer
    d["scorer"] = (
        sc.tier_sss_min_score, sc.tier_ss_min_score, sc.tier_s_min_score,
        sc.tier_a_min_score, sc.tier_b_min_score, sc.tier_c_min_score,
        sc.volume_ratio_threshold, sc.rsi_long_max, sc.rsi_short_min,
        sc.funding_long_max, sc.funding_short_min, sc.daily_ema_period,
        sc._regime_strong_adx, sc._regime_high_adx_cutoff, sc._rsi_neutral_penalty,
        sc._ml_mode, sc._ml_cut_threshold, sc._ml_bonus_t1, sc._ml_bonus_t2,
    )

    rd = e.regime_detector
    d["regime"] = (rd.adx_period, rd.adx_trending, rd.adx_ranging, rd.bb_period, rd.bb_std,
                   rd.bb_width_lookback, rd.bb_width_squeeze_pct, rd.primary_symbol, rd.primary_tf)

    d["funding"] = e.funding_sim.interval_hours
    d["strategies"] = sorted(
        (st.name, getattr(st, "signal_tf", None), tuple(getattr(st, "symbols", ())))
        for st in e.strategies
    )
    return d


def test_build_engine_parity():
    raw = yaml.safe_load(CONFIG.read_text())
    # build_engine은 config dict를 in-place 변형(cfg.setdefault 등) → 각자 deepcopy 사용
    bt = build_bt(copy.deepcopy(raw), initial_capital=100.0)
    broker = LiveBroker(exchange=object(), dry_run=True)  # __init__은 exchange 미사용
    live = build_live(copy.deepcopy(raw), broker, initial_capital=100.0)

    p_bt, p_live = _params(bt), _params(live)
    diffs = {k: (p_bt[k], p_live[k]) for k in p_bt if p_bt[k] != p_live[k]}
    assert not diffs, f"백테↔라이브 build_engine 파라미터 불일치: {diffs}"
