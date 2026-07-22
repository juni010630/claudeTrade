import os
import hashlib
import json

import pandas as pd

from scripts.edge_monitor import (
    _EVENT_COLS,
    match_events,
    resolve_anchor,
    rolling_percentiles,
)


def _events(times, prices):
    rows = []
    for timestamp, price in zip(times, prices):
        rows.append({
            "symbol": "BTC/USDT", "strategy": "ema_cross", "direction": "long",
            "entry_time": pd.Timestamp(timestamp, tz="UTC"), "entry_price": price,
            "status": "open", "exit_time": None, "exit_price": None,
            "exit_reason": None,
        })
    return pd.DataFrame(rows, columns=_EVENT_COLS)


def test_match_events_uses_nearest_pair_not_csv_order():
    live = _events(["2026-01-01 00:00", "2026-01-01 01:00"], [100, 200])
    replay = _events(["2026-01-01 00:50", "2026-01-01 01:00"], [101, 201])

    matched, live_only, replay_only = match_events(live, replay)

    assert not live_only and not replay_only
    assert {(l.entry_price, r.entry_price) for l, r in matched} == {(100, 101), (200, 201)}


def test_config_change_advances_anchor_and_then_stays_stable(tmp_path):
    config = tmp_path / "config.yaml"
    state = tmp_path / "anchor.json"
    config.write_text("version: 1\n")
    deployed = pd.Timestamp("2026-07-22 04:37:53", tz="UTC")
    os.utime(config, (deployed.timestamp(), deployed.timestamp()))

    anchor, changed = resolve_anchor(
        config, timeframe="1h", state_path=state,
        fallback=pd.Timestamp("2026-07-03 18:00", tz="UTC"),
    )
    stable_anchor, stable_changed = resolve_anchor(
        config, timeframe="1h", state_path=state,
        fallback=pd.Timestamp("2026-07-03 18:00", tz="UTC"),
    )

    assert anchor == pd.Timestamp("2026-07-22 05:00", tz="UTC")
    assert changed
    assert stable_anchor == anchor
    assert not stable_changed


def test_new_live_session_seed_advances_anchor_even_with_same_config(tmp_path):
    config = tmp_path / "config.yaml"
    state = tmp_path / "anchor.json"
    seed = tmp_path / "seed.json"
    config.write_text("version: 1\n")
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    state.write_text(json.dumps({
        "anchor": "2026-07-22T05:00:00+00:00", "config_sha256": digest,
    }))
    seed.write_text(json.dumps({
        "anchor": "2026-07-23T04:00:00+00:00", "config_sha256": digest,
    }))

    anchor, changed = resolve_anchor(
        config, timeframe="1h", state_path=state, seed_path=seed,
    )

    assert anchor == pd.Timestamp("2026-07-23 04:00", tz="UTC")
    assert changed


def test_missing_baseline_is_an_operational_alert(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("version: 1\n")

    lines, alerts = rolling_percentiles(
        config_path=config, baseline_path=tmp_path / "missing.json"
    )

    assert alerts == {"baseline:missing"}
    assert "베이스라인 없음" in lines[0]
