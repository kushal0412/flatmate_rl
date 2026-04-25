from __future__ import annotations

from broker_app.data import TASKS as BROKER_TASKS

from flatmate_rl.models import FlatmateRlAction
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
from flatmate_rl.server.heuristic_policy import autopolicy_next_request
from flatmate_rl.server.scenarios import SCENARIOS


def _tool(env: FlatmateRlEnvironment, name: str, **kwargs):
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


def test_scenarios_keep_broker_app_task_metadata() -> None:
    for scenario_id, scenario in SCENARIOS.items():
        broker = BROKER_TASKS[scenario_id]
        assert scenario["label"] == broker["label"]
        assert scenario["difficulty"] == broker["difficulty"]
        assert scenario["task_post_ids"] == broker["task_post_ids"]
        assert scenario["ground_truth"]["required_bookings"] == broker["ground_truth"]["required_bookings"]
        assert scenario["scenario_creation_config"]["expected_answers"] == broker["scenario_creation_config"]["expected_answers"]
        if scenario_id == "task_visit_single_seller_followup":
            assert (
                scenario["scenario_creation_config"]["followup_seller_expected_answers"]
                == broker["scenario_creation_config"]["followup_seller_expected_answers"]
            )


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
        payload = autopolicy_next_request("task_visit_single", obs.model_dump())
        assert payload is not None
        obs = env.step(FlatmateRlAction.model_validate(payload))
        if obs.done:
            break

    assert obs.done is True
    assert obs.booked_visits == [{"post_id": "post_023", "time": "Saturday 11am"}]


def test_hidden_flex_requires_alternative_slot_to_unlock_backup_availability() -> None:
    env = FlatmateRlEnvironment()
    obs = env.reset(scenario_id="task_visit_single_hidden_flex")
    assert "Tuesday after 6pm" in obs.last_user_message

    obs = _msg(env, "Can you confirm your dietary preference?")
    assert obs.last_user_message == "I’m non-vegetarian."

    obs = _msg(env, "No Tuesday slot matches. I can offer Saturday 1pm or Sunday 5pm instead.")
    assert "confirm" in obs.last_user_message.lower()
    assert "Sunday 5pm" in obs.last_user_message or "Saturday 1pm" in obs.last_user_message


def test_multi_visit_scenario_books_two_visits() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_multi")

    _msg(env, "Please share your dietary preference and visit availability.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    _tool(env, "match_location_preference", post_ids=["post_031", "post_052"])
    _tool(env, "get_commute_time", post_ids=["post_031", "post_052"])
    _tool(env, "check_calendar_slots", post_ids=["post_031", "post_052"])
    obs = _msg(env, "I shortlisted post_031 at tomorrow 7pm and post_052 at Sunday 4pm. Which listings do you want to pursue?")
    assert obs.selected_posts == ["post_031", "post_052"]
    _msg(env, "Please confirm tomorrow 7pm for post_031.")
    _tool(env, "contact_poster", post_id="post_031", time_text="tomorrow 7pm")
    _tool(env, "book_viewing", post_id="post_031", time_text="tomorrow 7pm")
    _msg(env, "Please confirm Sunday 4pm for post_052.")
    _tool(env, "contact_poster", post_id="post_052", time_text="Sunday 4pm")
    final_obs = _tool(env, "book_viewing", post_id="post_052", time_text="Sunday 4pm")

    assert final_obs.done is True
    assert len(final_obs.booked_visits) == 2


def test_seller_followup_scenario_schedules_dynamic_visit() -> None:
    env = FlatmateRlEnvironment()
    env.reset(scenario_id="task_visit_single_seller_followup")

    _msg(env, "Please share your dietary preference.")
    _tool(env, "store_user_details")
    _tool(env, "search_posts")
    transition = _msg(env, "None of the current listings fit your weekend availability.")
    assert transition.phase == "seller"
    _msg(env, "Please share the household dietary setup and who the flat is for.")
    _tool(env, "store_seller_details")
    _tool(env, "match_location_preference", post_ids=["post_dynamic_followup_1"])
    _tool(env, "check_table_slot_matches", post_ids=["post_dynamic_followup_1"])
    _tool(env, "confirm_seller_match", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")
    _tool(env, "offer_matched_listing_to_buyer", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")
    final_obs = _tool(env, "schedule_table_visit", post_id="post_dynamic_followup_1", time_text="Sunday 5pm")

    assert final_obs.done is True
    assert final_obs.booked_visits == [{"post_id": "post_dynamic_followup_1", "time": "Sunday 5pm"}]
