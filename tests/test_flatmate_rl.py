from __future__ import annotations

import importlib

from flatmate_rl.models import FlatmateRlAction
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
from flatmate_rl.server.heuristic_policy import autopolicy_next_request, expected_policy_action
from flatmate_rl.server.scenarios import POSTS, SCENARIOS


def _tool(env: FlatmateRlEnvironment, name: str, **kwargs):
    scenario_id = env.state.scenario_id or getattr(getattr(env, "_episode", None), "_scenario", {}).get("task_id", "")
    if not kwargs and name == "store_user_details":
        kwargs = dict(SCENARIOS[scenario_id]["scenario_creation_config"]["expected_answers"])
    if not kwargs and name == "store_seller_details":
        kwargs = dict(SCENARIOS[scenario_id]["scenario_creation_config"]["followup_seller_expected_answers"])
    return env.step(
        FlatmateRlAction(
            action_type="tool_call",
            tool_name=name,
            tool_arguments=kwargs,
        )
    )


def _msg(env: FlatmateRlEnvironment, text: str):
    return env.step(
        FlatmateRlAction(
            action_type="assistant_message",
            assistant_message=text,
        )
    )


def test_scenarios_are_self_consistent() -> None:
    for scenario_id, scenario in SCENARIOS.items():
        assert scenario["task_id"] == scenario_id
        assert scenario["label"]
        assert scenario["difficulty"] in {"medium", "hard"}
        assert scenario["initial_user_message"]

        assert scenario["task_post_ids"]
        assert len(scenario["task_post_ids"]) == len(set(scenario["task_post_ids"]))
        assert all(post_id in POSTS for post_id in scenario["task_post_ids"])

        ground_truth = scenario["ground_truth"]
        expected_answers = scenario["scenario_creation_config"]["expected_answers"]

        assert ground_truth["required_bookings"] >= 1
        assert ground_truth["required_tool_calls"]
        assert ground_truth["required_info"]
        assert ground_truth["optimal_posts"]
        assert set(ground_truth["optimal_posts"]).issubset(set(scenario["task_post_ids"]) | {"post_dynamic_followup_1"})
        assert set(ground_truth["acceptable_posts"]).issubset(set(scenario["task_post_ids"]))
        assert set(ground_truth["dealbreaker_posts"]).issubset(set(scenario["task_post_ids"]))

        assert expected_answers["user_type"] == "buyer"
        assert expected_answers["user_sub_type"] == "flat"
        assert expected_answers["budget_max"] == scenario["buyer_profile"]["budget_max"]
        assert expected_answers["dietary"] == scenario["buyer_profile"]["dietary"]
        assert expected_answers["areas"] == scenario["buyer_profile"]["areas"]
        assert expected_answers["occupation"] == scenario["buyer_profile"]["occupation"]
        assert expected_answers["visit_availability"] == scenario["buyer_profile"]["visit_availability"]

        if scenario_id == "task_visit_single_seller_followup":
            assert scenario["seller_profile"] is not None
            assert (
                scenario["scenario_creation_config"]["followup_seller_expected_answers"]["calendar_slots"]
                == scenario["seller_profile"]["calendar_slots"]
            )
            assert scenario["scenario_creation_config"]["followup_seller_expected_answers"]["area"] == scenario["seller_profile"]["area"]
        else:
            assert scenario["seller_profile"] is None


def test_reset_exposes_initial_buyer_message() -> None:
    env = FlatmateRlEnvironment()
    observation = env.reset(scenario_id="task_visit_single")

    assert observation.status == "ready"
    assert observation.scenario_id == "task_visit_single"
    assert observation.phase == "buyer"
    assert "budget is up to Rs. 20,000" in observation.last_user_message
    assert observation.remaining_required_fields == ["diet", "visit_availability"]


def test_search_before_store_user_details_fails() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single")

    result = _tool(env, "search_posts")

    assert result.last_tool_result["success"] is False
    assert "store_user_details must be called before search_posts" in result.last_tool_result["message"]


def test_store_user_details_does_not_return_expected_answers_payload() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single")

    _msg(env, "Please share your dietary preference and visit availability.")
    result = _tool(env, "store_user_details")

    assert result.last_tool_result == {
        "tool": "store_user_details",
        "success": True,
        "message": "Buyer profile stored.",
    }


def test_strict_eval_mode_hides_scenario_metadata_and_reward(monkeypatch) -> None:
    monkeypatch.setenv("STRICT_EVAL_MODE", "1")

    environment_module = importlib.import_module("flatmate_rl.server.flatmate_rl_environment")
    environment_module = importlib.reload(environment_module)
    env = environment_module.FlatmateRlEnvironment()

    observation = env.reset(scenario_id="task_visit_single")

    assert observation.scenario_id == ""
    assert observation.scenario_label == ""
    assert observation.difficulty == ""
    assert observation.gathered_fields == []
    assert observation.remaining_required_fields == []
    assert observation.violations == []
    assert observation.tool_trace == []
    assert observation.total_reward == 0.0
    assert "diet" in observation.feedback_summary
    assert "visit_availability" in observation.feedback_summary

    _msg(env, "Please share your dietary preference and visit availability.")
    result = _tool(env, "store_user_details")

    assert result.last_tool_result == {
        "tool": "store_user_details",
        "success": True,
        "message": "Buyer profile stored.",
    }
    assert result.total_reward == 0.0
    assert result.tool_trace == []

    monkeypatch.delenv("STRICT_EVAL_MODE", raising=False)
    importlib.reload(environment_module)


def test_single_visit_scenario_books_one_visit() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single")

    _msg(env, "Please share your dietary preference and visit availability.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    _tool(env, "match_location_preference", post_ids=["post_023", "post_031"])
    _tool(env, "get_commute_time", post_ids=["post_023", "post_031"])
    _tool(env, "check_calendar_slots", post_ids=["post_023"])
    _msg(env, "post_023 is available Saturday 11am. Please confirm Saturday 11am if that works.")
    _tool(env, "contact_poster", post_id="post_023", time_text="Saturday 11am")
    final_obs = _tool(env, "book_viewing", post_id="post_023", time_text="Saturday 11am")

    assert final_obs.done is True
    assert final_obs.booked_visits == [{"post_id": "post_023", "time": "Saturday 11am"}]
    assert len(final_obs.seller_conversation_history) >= 2
    assert "Can we visit at Saturday 11am" in final_obs.seller_conversation_history[0]["content"]
    assert "Saturday 11am works for the visit" in final_obs.seller_conversation_history[1]["content"]


def test_buyer_answers_diet_and_availability_when_broker_asks_for_both() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single")

    obs = _msg(env, "Please share your dietary preference and visit availability.")

    assert "non-vegetarian" in obs.last_user_message
    assert "visit availability" in obs.last_user_message
    assert "diet" in obs.gathered_fields
    assert "visit_availability" in obs.gathered_fields
    assert obs.remaining_required_fields == []


def test_heuristic_policy_progresses_after_confirmation_in_single_visit() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single")

    for _ in range(12):
        payload = expected_policy_action("task_visit_single", obs.model_dump())
        assert payload is not None
        obs = env.step(FlatmateRlAction.model_validate(payload))
        if obs.done:
            break

    assert obs.done is True
    assert obs.booked_visits == [{"post_id": "post_023", "time": "Saturday 11am"}]


def test_expected_flow_violation_gets_large_penalty() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single")

    _msg(env, "Please share your dietary preference and visit availability.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    wrong_next_step = _tool(env, "search_posts")

    assert wrong_next_step.done is True
    assert wrong_next_step.status == "failed"
    assert wrong_next_step.step_reward == -9.9
    assert "expected flow violation" in wrong_next_step.message.lower()
    assert "expected_flow_violation" in wrong_next_step.violations


def test_seller_followup_wrong_step_gets_large_penalty() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single_seller_followup")

    _msg(env, "Please share your dietary preference.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    wrong_transition = _tool(env, "match_location_preference", post_ids=["post_131"])

    assert wrong_transition.done is True
    assert wrong_transition.status == "failed"
    assert wrong_transition.step_reward == -9.9
    assert "expected next step" in wrong_transition.message.lower()
    assert "expected_flow_violation" in wrong_transition.violations


def test_seller_followup_accepts_paraphrased_assistant_message() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single_seller_followup")

    obs = _msg(
        env,
        "Could you please let me know about your dietary preferences? This will help me find the best match for you.",
    )

    assert obs.status == "user_response"
    assert obs.done is False
    assert obs.violations == []
    assert "diet" in obs.gathered_fields


def test_seller_followup_accepts_expected_tool_with_different_arguments() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single_seller_followup")

    _msg(
        env,
        "Could you please let me know about your dietary preferences? This will help me find the best match for you.",
    )
    obs = _tool(env, "store_user_details", diet="non-vegetarian")

    assert obs.status == "tool_result"
    assert obs.done is False
    assert obs.violations == []
    assert obs.last_tool_result["success"] is True


def test_heuristic_policy_recovers_from_strict_eval_feedback() -> None:
    sanitized_observation = {
        "done": False,
        "phase": "buyer",
        "buyer_profile_stored": False,
        "seller_profile_stored": False,
        "remaining_required_fields": [],
        "feedback_summary": "Ask the buyer for these missing fields before storing details: diet, visit_availability.",
        "message": "Missing buyer fields: diet, visit_availability.",
        "last_tool_result": {
            "tool": "store_user_details",
            "success": False,
            "message": "Missing buyer fields: diet, visit_availability.",
        },
        "booked_visits": [],
        "selected_posts": [],
        "tool_trace": [],
        "buyer_conversation_history": [
            {
                "role": "user",
                "content": "Hi, I'm looking for a flatmate-share near Goregaon East.",
            }
        ],
        "status": "tool_result",
    }

    action = autopolicy_next_request("task_visit_single", sanitized_observation)

    assert action == {
        "action_type": "assistant_message",
        "assistant_message": "Please share your dietary preference and visit availability.",
    }


def test_hidden_flex_requires_alternative_slot_to_unlock_backup_availability() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single_hidden_flex")
    assert "Tuesday after 6pm" in obs.last_user_message

    obs = _msg(env, "Please share your dietary preference.")
    assert obs.last_user_message == "I’m non-vegetarian."
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    _tool(env, "match_location_preference", post_ids=["post_023", "post_052"])
    _tool(env, "get_commute_time", post_ids=["post_023", "post_052"])
    _tool(env, "check_calendar_slots", post_ids=["post_023", "post_052"])
    obs = _msg(env, "No Tuesday slot matches. I can offer Saturday 1pm or Sunday 5pm instead.")
    assert "confirm" in obs.last_user_message.lower()
    assert "Sunday 5pm" in obs.last_user_message or "Saturday 1pm" in obs.last_user_message


def test_multi_visit_scenario_books_two_visits() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_multi")

    for _ in range(20):
        payload = expected_policy_action("task_visit_multi", obs.model_dump())
        assert payload is not None
        obs = env.step(FlatmateRlAction.model_validate(payload))
        if obs.done:
            break

    assert obs.done is True
    assert len(obs.booked_visits) == 2


def test_seller_followup_scenario_schedules_dynamic_visit() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single_seller_followup")

    _msg(env, "Please share your dietary preference.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    transition = _tool(env, "close_buyer_conversation")
    assert transition.phase == "seller"
    assert "I will follow up if a suitable listing comes in" in transition.buyer_conversation_history[-1]["content"]
    assert "listing a new flatmate-share opening" in transition.seller_conversation_history[-1]["content"]
    _msg(env, "Please share the household dietary setup and who the flat is for.")
    _tool(env, "store_seller_details")
    _tool(env, "match_location_preference", post_ids=["post_dynamic_followup_1"])
    _tool(env, "check_table_slot_matches", post_ids=["post_dynamic_followup_1"])
    _tool(env, "confirm_seller_match", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")
    _tool(env, "offer_matched_listing_to_buyer", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")
    final_obs = _tool(env, "schedule_table_visit", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")

    assert final_obs.done is True
    assert final_obs.booked_visits == [{"post_id": "post_dynamic_followup_1", "time": "Sunday 5pm"}]
