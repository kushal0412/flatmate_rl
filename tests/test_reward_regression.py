from __future__ import annotations

import pytest

from flatmate_rl.models import FlatmateRlAction
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
from flatmate_rl.server.heuristic_policy import expected_policy_action


HEURISTIC_BASELINES = {
    "task_visit_single": 0.70,
    "task_visit_single_hidden_flex": 0.90,
    "task_visit_multi": 1.10,
    "task_visit_single_seller_followup": 0.90,
}


def _run_heuristic_scenario(scenario_id: str, max_steps: int = 30):
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id=scenario_id)
    steps = 0
    for _ in range(max_steps):
        payload = expected_policy_action(scenario_id, obs.model_dump())
        if payload is None:
            break
        obs = env.step(FlatmateRlAction.model_validate(payload))
        steps += 1
        if obs.done:
            break
    return obs, steps


def test_heuristic_reward_regression() -> None:
    for scenario_id, expected_total in HEURISTIC_BASELINES.items():
        obs, _steps = _run_heuristic_scenario(scenario_id)

        assert obs.total_reward == pytest.approx(expected_total, abs=0.3)
