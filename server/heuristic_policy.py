"""Shared heuristic policy for Flatmate RL scenario execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

try:
    from .scenarios import SCENARIOS
except ImportError:
    from server.scenarios import SCENARIOS


def active_scenario(task_id: str) -> dict[str, Any]:
    return SCENARIOS[task_id]


def _enrich_expected_tool_arguments(task_id: str, observation: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(payload)
    if enriched.get("action_type") != "tool_call":
        return enriched

    scenario = active_scenario(task_id)
    tool_name = enriched.get("tool_name", "")
    phase = observation.get("phase", "buyer")

    if tool_name == "store_user_details" and phase == "buyer":
        enriched["tool_arguments"] = deepcopy(scenario["scenario_creation_config"].get("expected_answers", {}))
    elif tool_name == "store_seller_details" and phase == "seller":
        enriched["tool_arguments"] = deepcopy(scenario["scenario_creation_config"].get("followup_seller_expected_answers", {}))
    else:
        enriched.setdefault("tool_arguments", {})

    return enriched


def expected_policy_action(task_id: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    payload = autopolicy_next_request(task_id, observation)
    if payload is None:
        return None
    return _enrich_expected_tool_arguments(task_id, observation, payload)


def _missing_fields_from_feedback(observation: dict[str, Any]) -> list[str]:
    feedback = " ".join(
        [
            str(observation.get("feedback_summary", "")),
            str(observation.get("message", "")),
            str(observation.get("last_tool_result", {}).get("message", "")),
        ]
    ).lower()
    fields = []
    patterns = {
        "diet": ["diet"],
        "visit_availability": ["visit_availability", "visit availability"],
        "occupation": ["occupation"],
        "budget": ["budget"],
        "areas": ["areas", "area"],
        "listing_choices": ["listing_choices", "listing choices"],
        "dietary": ["dietary"],
        "occupation_requirement": ["occupation requirement", "who the flat is for"],
    }
    for field, phrases in patterns.items():
        if any(phrase in feedback for phrase in phrases):
            fields.append(field)
    return fields


def _ask_for_missing_fields(missing: list[str], phase: str, task_id: str) -> dict[str, Any] | None:
    if phase == "seller":
        if "dietary" in missing and "occupation_requirement" in missing:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup and who the flat is for."}
        if "dietary" in missing:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup."}
        if "occupation_requirement" in missing:
            return {"action_type": "assistant_message", "assistant_message": "Please share who the flat is for."}
        return None

    if "diet" in missing and "visit_availability" in missing:
        return {"action_type": "assistant_message", "assistant_message": "Please share your dietary preference and visit availability."}
    if "diet" in missing:
        return {"action_type": "assistant_message", "assistant_message": "Please share your dietary preference."}
    if "visit_availability" in missing:
        return {"action_type": "assistant_message", "assistant_message": "Please share your visit availability."}
    if "listing_choices" in missing and task_id == "task_visit_multi":
        return {"action_type": "assistant_message", "assistant_message": "I shortlisted post_031 at tomorrow 7pm and post_052 at Sunday 4pm. Which listings do you want to pursue?"}
    return None


def autopolicy_next_request(task_id: str, observation: dict[str, Any]) -> dict[str, Any] | None:
    trace = observation.get("tool_trace", [])
    tool_names = [item.get("tool", "") for item in trace]
    phase = observation.get("phase", "buyer")
    remaining = set(observation.get("remaining_required_fields", []))
    selected_posts = list(observation.get("selected_posts", []))
    booked = [item["post_id"] for item in observation.get("booked_visits", [])]
    buyer_history = observation.get("buyer_conversation_history", [])
    last_buyer_role = str(buyer_history[-1].get("role", "")) if buyer_history else ""
    last_buyer_message = str(observation.get("last_user_message", ""))
    user_has_replied = observation.get("status") == "user_response" and last_buyer_role == "user"
    feedback_missing = _missing_fields_from_feedback(observation)
    if feedback_missing:
        remaining.update(feedback_missing)

    def has_tool(name: str) -> bool:
        return name in tool_names

    if observation.get("done"):
        return None

    if phase == "buyer":
        if task_id == "task_visit_single_seller_followup" and not observation.get("seller_profile_stored"):
            if not observation.get("buyer_profile_stored"):
                missing_prompt = _ask_for_missing_fields(sorted(remaining), phase=phase, task_id=task_id)
                if missing_prompt is not None:
                    return missing_prompt
                return {"action_type": "tool_call", "tool_name": "store_user_details", "tool_arguments": {}}
            if not has_tool("search_posts"):
                return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}
            return {"action_type": "tool_call", "tool_name": "close_buyer_conversation", "tool_arguments": {}}

        if not observation.get("buyer_profile_stored"):
            missing_prompt = _ask_for_missing_fields(sorted(remaining), phase=phase, task_id=task_id)
            if missing_prompt is not None:
                return missing_prompt
            return {"action_type": "tool_call", "tool_name": "store_user_details", "tool_arguments": {}}

        if not has_tool("search_posts"):
            return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}

        if task_id == "task_visit_single":
            if not has_tool("match_location_preference"):
                return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_023", "post_031"]}}
            if not has_tool("get_commute_time"):
                return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_023", "post_031"]}}
            if not has_tool("check_calendar_slots"):
                return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_023"]}}
            if "contact_poster" not in tool_names and "book_viewing" not in tool_names and not user_has_replied:
                return {"action_type": "assistant_message", "assistant_message": "post_023 is available Saturday 11am. Please confirm Saturday 11am if that works."}
            if not has_tool("contact_poster"):
                return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_023", "time_text": "Saturday 11am"}}
            return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_023", "time_text": "Saturday 11am"}}

        if task_id == "task_visit_single_hidden_flex":
            if not has_tool("match_location_preference"):
                return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_023", "post_052"]}}
            if not has_tool("get_commute_time"):
                return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_023", "post_052"]}}
            if not has_tool("check_calendar_slots"):
                return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_023", "post_052"]}}
            if not has_tool("contact_poster") and not has_tool("book_viewing") and not user_has_replied:
                return {"action_type": "assistant_message", "assistant_message": "No Tuesday slot matches. I can offer Saturday 1pm or Sunday 5pm instead."}
            if not has_tool("contact_poster"):
                return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_023", "time_text": "Sunday 5pm"}}
            return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_023", "time_text": "Sunday 5pm"}}

        if task_id == "task_visit_multi":
            if not has_tool("match_location_preference"):
                return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_031", "post_052"]}}
            if not has_tool("get_commute_time"):
                return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_031", "post_052"]}}
            if not has_tool("check_calendar_slots"):
                return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_031", "post_052"]}}
            if not selected_posts and not user_has_replied:
                return {"action_type": "assistant_message", "assistant_message": "I shortlisted post_031 at tomorrow 7pm and post_052 at Sunday 4pm. Which listings do you want to pursue?"}
            if "post_031" not in booked and tool_names.count("contact_poster") == 0 and "confirm tomorrow 7pm" not in last_buyer_message.lower():
                return {"action_type": "assistant_message", "assistant_message": "Please confirm tomorrow 7pm for post_031."}
            if "post_031" not in booked and tool_names.count("contact_poster") == 0:
                return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_031", "time_text": "tomorrow 7pm"}}
            if "post_031" not in booked:
                return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_031", "time_text": "tomorrow 7pm"}}
            if "post_052" not in booked and tool_names.count("contact_poster") == 1 and "confirm sunday 4pm" not in last_buyer_message.lower():
                return {"action_type": "assistant_message", "assistant_message": "Please confirm Sunday 4pm for post_052."}
            if "post_052" not in booked and tool_names.count("contact_poster") == 1 and tool_names.count("book_viewing") == 1:
                return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_052", "time_text": "Sunday 4pm"}}
            if "post_052" not in booked:
                return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_052", "time_text": "Sunday 4pm"}}

    if phase == "seller":
        if not observation.get("seller_profile_stored"):
            missing_prompt = _ask_for_missing_fields(sorted(remaining), phase=phase, task_id=task_id)
            if missing_prompt is not None:
                return missing_prompt
            return {"action_type": "tool_call", "tool_name": "store_seller_details", "tool_arguments": {}}
        if not has_tool("match_location_preference"):
            return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_dynamic_followup_1"]}}
        if not has_tool("check_table_slot_matches"):
            return {"action_type": "tool_call", "tool_name": "check_table_slot_matches", "tool_arguments": {"post_ids": ["post_dynamic_followup_1"]}}
        if not has_tool("confirm_seller_match"):
            return {"action_type": "tool_call", "tool_name": "confirm_seller_match", "tool_arguments": {"post_id": "post_dynamic_followup_1", "time_text": "Sunday 5pm"}}
        if not has_tool("offer_matched_listing_to_buyer"):
            return {"action_type": "tool_call", "tool_name": "offer_matched_listing_to_buyer", "tool_arguments": {"post_id": "post_dynamic_followup_1", "time_text": "Sunday 5pm"}}
        return {"action_type": "tool_call", "tool_name": "schedule_table_visit", "tool_arguments": {"post_id": "post_dynamic_followup_1", "time_text": "Sunday 5pm"}}

    return None
