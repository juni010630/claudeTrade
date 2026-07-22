import json

from scripts.edge_monitor import should_notify_alert_transition


def test_first_run_notifies_existing_issue(tmp_path):
    path = tmp_path / "edge_alert_state.json"

    assert should_notify_alert_transition({"parity:old"}, state_path=path)
    assert json.loads(path.read_text())["active_alerts"] == ["parity:old"]


def test_unchanged_alert_is_suppressed_but_new_alert_notifies(tmp_path):
    path = tmp_path / "edge_alert_state.json"
    should_notify_alert_transition({"parity:old"}, state_path=path)

    assert not should_notify_alert_transition({"parity:old"}, state_path=path)
    assert should_notify_alert_transition(
        {"parity:old", "slippage:max"}, state_path=path
    )


def test_resolved_alert_notifies_and_can_notify_on_recurrence(tmp_path):
    path = tmp_path / "edge_alert_state.json"
    should_notify_alert_transition({"parity:old"}, state_path=path)
    assert should_notify_alert_transition(set(), state_path=path)

    assert should_notify_alert_transition({"parity:old"}, state_path=path)


def test_weekly_summary_and_force_notify_even_without_transition(tmp_path):
    path = tmp_path / "edge_alert_state.json"
    should_notify_alert_transition(set(), state_path=path)

    assert should_notify_alert_transition(set(), is_monday=True, state_path=path)
    assert should_notify_alert_transition(set(), force=True, state_path=path)


def test_corrupt_existing_state_fails_open_for_active_alert(tmp_path):
    path = tmp_path / "edge_alert_state.json"
    path.write_text("not-json")

    assert should_notify_alert_transition({"parity:new"}, state_path=path)


def test_non_persistent_check_does_not_hide_failed_delivery(tmp_path):
    path = tmp_path / "edge_alert_state.json"

    assert should_notify_alert_transition(
        {"parity:new"}, state_path=path, persist=False
    )
    assert not path.exists()
