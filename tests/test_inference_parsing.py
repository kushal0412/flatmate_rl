from __future__ import annotations

import pytest

from flatmate_rl.inference import malformed_action_observation, parse_action
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment


def test_strict_parse_rejects_tool_name_in_action_type() -> None:
    parsed = parse_action('{"action_type":"store_user_details","tool_arguments":{}}', strict=True)

    assert parsed.action is None
    assert parsed.error is not None
    assert "schema_validation_failed" in parsed.error
    assert "action_type must be" in parsed.error


def test_legacy_parse_can_coerce_tool_name_in_action_type() -> None:
    parsed = parse_action('{"action_type":"store_user_details","tool_arguments":{}}', strict=False)

    assert parsed.action is not None
    assert parsed.action.action_type == "tool_call"
    assert parsed.action.tool_name == "store_user_details"
    assert parsed.warning is not None
    assert "coerced invalid action_type" in parsed.warning


def test_strict_parse_reports_json_error() -> None:
    parsed = parse_action('{"action_type":"tool_call"', strict=True)

    assert parsed.action is None
    assert parsed.error is not None
    assert parsed.error.startswith("json_parse_failed")


def test_malformed_action_feedback_is_recoverable() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single")

    feedback_obs = malformed_action_observation(obs, "schema_validation_failed: bad action")

    assert feedback_obs.done is False
    assert feedback_obs.step_reward == pytest.approx(-0.05)
    assert feedback_obs.total_reward == pytest.approx(-0.05)
    assert feedback_obs.last_tool_result["error"] == "schema_validation_failed"
    assert "expected_schema" in feedback_obs.last_tool_result

