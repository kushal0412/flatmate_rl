from __future__ import annotations

from types import SimpleNamespace

import pytest

from flatmate_rl.inference import (
    ModelConfigurationError,
    build_user_prompt,
    get_model_action,
    malformed_action_observation,
    parse_action,
)
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
from flatmate_rl.models import FlatmateRlAction


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


def test_user_prompt_renders_prerequisites_and_recent_tools() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single")
    obs = env.step(
        FlatmateRlAction(
            action_type="assistant_message",
            assistant_message="Please share your dietary preference and visit availability.",
        )
    )
    obs = env.step(
        FlatmateRlAction(
            action_type="tool_call",
            tool_name="store_user_details",
            tool_arguments={},
        )
    )

    prompt = build_user_prompt(step=2, observation=obs)

    assert "Prerequisites satisfied:" in prompt
    assert '"details_stored": true' in prompt
    assert "Recent tool calls:" in prompt
    assert "store_user_details" in prompt


def test_model_call_error_does_not_fallback_to_heuristic() -> None:
    class FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("requested model is not supported")

    client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single")

    with pytest.raises(ModelConfigurationError, match="MODEL_NAME is invalid or unsupported"):
        get_model_action(
            client=client,
            task_id="task_visit_single",
            step=1,
            observation=obs,
            explain=False,
            strict_parsing=True,
        )
