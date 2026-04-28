"""Shared heuristic policy for Flatmate RL scenario execution."""

from __future__ import annotations

from copy import deepcopy
import re
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
        need_dietary = "dietary" in missing
        need_occupation = "occupation_requirement" in missing
        need_slots = "calendar_slots" in missing
        if need_dietary and need_occupation and need_slots:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup, who the flat is for, and available visit time slots."}
        if need_dietary and need_occupation:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup and who the flat is for."}
        if need_dietary and need_slots:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup and available visit time slots."}
        if need_occupation and need_slots:
            return {"action_type": "assistant_message", "assistant_message": "Please share who the flat is for and available visit time slots."}
        if need_dietary:
            return {"action_type": "assistant_message", "assistant_message": "Please share the household dietary setup."}
        if need_occupation:
            return {"action_type": "assistant_message", "assistant_message": "Please share who the flat is for."}
        if need_slots:
            return {"action_type": "assistant_message", "assistant_message": "Please share available visit time slots."}
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


def _store_or_ask(remaining: set[str], task_id: str, phase: str) -> dict[str, Any]:
    """Return ask-for-fields message or store_user_details tool call."""
    missing_prompt = _ask_for_missing_fields(sorted(remaining), phase=phase, task_id=task_id)
    if missing_prompt is not None:
        return missing_prompt
    return {"action_type": "tool_call", "tool_name": "store_user_details", "tool_arguments": {}}


def _extract_stated_budget(observation: dict[str, Any], fallback: int = 20000) -> int:
    text_parts = [
        str(observation.get("current_user_request", "")),
        str(observation.get("last_user_message", "")),
    ]
    for entry in observation.get("buyer_conversation_history", []):
        if entry.get("role") == "user":
            text_parts.append(str(entry.get("content", "")))
    text = " ".join(text_parts)
    matches = re.findall(r"Rs\.\s*([0-9][0-9,]*)", text)
    if not matches:
        return fallback
    return int(matches[0].replace(",", ""))


def _autopolicy_negotiation(
    observation: dict[str, Any],
    tool_names: list[str],
    user_has_replied: bool,
    remaining: set[str],
) -> dict[str, Any] | None:
    """Heuristic for task_negotiation_hidden_budget."""
    def has_tool(name: str) -> bool:
        return name in tool_names

    stated_budget = _extract_stated_budget(observation)
    buyer_probe_high = stated_budget + 3000
    overlap_price = stated_budget + 1000

    if not observation.get("buyer_profile_stored"):
        return _store_or_ask(remaining, "task_negotiation_hidden_budget", "buyer")
    if not has_tool("search_posts"):
        return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}
    if not has_tool("match_location_preference"):
        return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_155"]}}
    if not has_tool("get_commute_time"):
        return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_155"]}}
    if not has_tool("check_calendar_slots"):
        return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_155"]}}
    if not has_tool("shortlist"):
        return {"action_type": "tool_call", "tool_name": "shortlist", "tool_arguments": {"post_ids": ["post_155"]}}
    # Probe buyer: first at a value above their ceiling (expect reject), then below
    buyer_probes = [t for t in observation.get("tool_trace", []) if t.get("tool") == "propose_price_to_buyer"]
    if len(buyer_probes) == 0:
        return {"action_type": "tool_call", "tool_name": "propose_price_to_buyer", "tool_arguments": {"post_id": "post_155", "proposed_rent": buyer_probe_high}}
    if len(buyer_probes) == 1:
        return {"action_type": "tool_call", "tool_name": "propose_price_to_buyer", "tool_arguments": {"post_id": "post_155", "proposed_rent": overlap_price}}
    buyer_accepted = {
        int(t.get("args", {}).get("proposed_rent", 0))
        for t in buyer_probes
        if "accepted" in str(t.get("message", "")).lower()
    }
    seller_probes = [t for t in observation.get("tool_trace", []) if t.get("tool") == "propose_price_to_seller"]
    if not seller_probes:
        return {"action_type": "tool_call", "tool_name": "propose_price_to_seller", "tool_arguments": {"post_id": "post_155", "proposed_rent": overlap_price}}
    seller_accepted = {
        int(t.get("args", {}).get("proposed_rent", 0))
        for t in seller_probes
        if "accepted" in str(t.get("message", "")).lower()
    }
    agreed_prices = sorted(buyer_accepted & seller_accepted)
    if agreed_prices:
        return {"action_type": "tool_call", "tool_name": "confirm_negotiated_deal", "tool_arguments": {"post_id": "post_155", "agreed_rent": agreed_prices[0]}}
    if buyer_accepted:
        return {"action_type": "tool_call", "tool_name": "propose_price_to_seller", "tool_arguments": {"post_id": "post_155", "proposed_rent": max(buyer_accepted)}}
    return {"action_type": "tool_call", "tool_name": "propose_price_to_buyer", "tool_arguments": {"post_id": "post_155", "proposed_rent": overlap_price - 500}}


def _autopolicy_waitlist(
    observation: dict[str, Any],
    tool_names: list[str],
    user_has_replied: bool,
    remaining: set[str],
) -> dict[str, Any] | None:
    """Heuristic for task_slot_cancellation_waitlist."""
    def has_tool(name: str) -> bool:
        return name in tool_names

    if not observation.get("buyer_profile_stored"):
        return _store_or_ask(remaining, "task_slot_cancellation_waitlist", "buyer")
    if not has_tool("search_posts"):
        return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}
    if not has_tool("match_location_preference"):
        return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_162"]}}
    if not has_tool("get_commute_time"):
        return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_162"]}}
    if not has_tool("check_calendar_slots"):
        return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_162"]}}
    if not has_tool("add_to_waitlist") and not user_has_replied:
        return {"action_type": "assistant_message", "assistant_message": "All slots for post_162 are currently fully booked. Let me add you to the waitlist."}
    if not has_tool("add_to_waitlist"):
        return {"action_type": "tool_call", "tool_name": "add_to_waitlist", "tool_arguments": {"post_id": "post_162"}}
    if not has_tool("notify_buyer_slot_freed") and not user_has_replied:
        return {"action_type": "assistant_message", "assistant_message": "You're on the waitlist for post_162. I'll reach out as soon as a slot opens up."}
    if not has_tool("notify_buyer_slot_freed"):
        return {"action_type": "tool_call", "tool_name": "notify_buyer_slot_freed", "tool_arguments": {"post_id": "post_162", "slot": "Saturday 10am"}}
    if not has_tool("contact_poster"):
        return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_162", "time_text": "Saturday 10am"}}
    return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_162", "time_text": "Saturday 10am"}}


def _autopolicy_multi_visit(
    observation: dict[str, Any],
    tool_names: list[str],
    user_has_replied: bool,
    remaining: set[str],
    booked: list[str],
    last_buyer_message: str,
) -> dict[str, Any] | None:
    """Heuristic for task_multi_visit_preference_evolution."""
    def has_tool(name: str) -> bool:
        return name in tool_names

    def count_tool(name: str) -> int:
        return tool_names.count(name)

    trace = observation.get("tool_trace", [])
    debrief_posts = [t.get("args", {}).get("post_id") for t in trace if t.get("tool") == "debrief_visit"]
    debrief1_done = "post_023" in debrief_posts
    debrief2_done = "post_052" in debrief_posts
    visit1_booked = "post_023" in booked
    visit2_booked = "post_052" in booked
    visit3_booked = "post_067" in booked

    if not observation.get("buyer_profile_stored"):
        return _store_or_ask(remaining, "task_multi_visit_preference_evolution", "buyer")

    # ---- Initial search and visit 1 setup ----
    if count_tool("search_posts") == 0:
        return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}
    if not has_tool("match_location_preference"):
        return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_023", "post_052"]}}
    if not has_tool("get_commute_time"):
        return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_023"]}}
    if count_tool("check_calendar_slots") == 0:
        return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_023"]}}

    # Propose and confirm visit 1
    if not visit1_booked and count_tool("contact_poster") == 0 and "confirm saturday 11am" not in last_buyer_message.lower():
        return {"action_type": "assistant_message", "assistant_message": "post_023 has Saturday 11am available. Please confirm Saturday 11am if that works."}
    if not visit1_booked and count_tool("contact_poster") == 0:
        return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_023", "time_text": "Saturday 11am"}}
    if not visit1_booked:
        return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_023", "time_text": "Saturday 11am"}}

    # ---- Debrief visit 1 and update preferences ----
    if not debrief1_done and not user_has_replied:
        return {"action_type": "assistant_message", "assistant_message": "How was your visit to post_023? What did you think of the flat?"}
    if not debrief1_done:
        return {"action_type": "tool_call", "tool_name": "debrief_visit", "tool_arguments": {"post_id": "post_023", "user_feedback": "buyer found the area too noisy, needs a quiet location"}}
    if count_tool("store_user_details") < 2:
        return {"action_type": "tool_call", "tool_name": "store_user_details", "tool_arguments": {}}
    if count_tool("filter_new_arrivals") == 0:
        return {"action_type": "tool_call", "tool_name": "filter_new_arrivals", "tool_arguments": {"post_ids": ["post_n01", "post_n02", "post_q01"]}}
    if count_tool("search_posts") == 1:
        return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}

    # ---- Visit 2 setup ----
    if count_tool("check_calendar_slots") == 1:
        return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_052"]}}
    if not visit2_booked and count_tool("contact_poster") == 1 and "confirm sunday 4pm" not in last_buyer_message.lower():
        return {"action_type": "assistant_message", "assistant_message": "post_052 has Sunday 4pm available. Please confirm Sunday 4pm if that works."}
    if not visit2_booked and count_tool("contact_poster") == 1:
        return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_052", "time_text": "Sunday 4pm"}}
    if not visit2_booked:
        return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_052", "time_text": "Sunday 4pm"}}

    # ---- Debrief visit 2 and update preferences ----
    if not debrief2_done and not user_has_replied:
        return {"action_type": "assistant_message", "assistant_message": "How was your visit to post_052? Did it meet your needs?"}
    if not debrief2_done:
        return {"action_type": "tool_call", "tool_name": "debrief_visit", "tool_arguments": {"post_id": "post_052", "user_feedback": "flat was quiet but no gym nearby, needs gym access"}}
    if count_tool("store_user_details") < 3:
        return {"action_type": "tool_call", "tool_name": "store_user_details", "tool_arguments": {}}
    if count_tool("filter_new_arrivals") == 1:
        return {"action_type": "tool_call", "tool_name": "filter_new_arrivals", "tool_arguments": {"post_ids": ["post_g01", "post_i01", "post_067"]}}
    if count_tool("search_posts") == 2:
        return {"action_type": "tool_call", "tool_name": "search_posts", "tool_arguments": {}}

    # ---- Visit 3 setup ----
    if count_tool("check_calendar_slots") == 2:
        return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_067"]}}
    if not visit3_booked and count_tool("contact_poster") == 2 and "confirm saturday 3pm" not in last_buyer_message.lower():
        return {"action_type": "assistant_message", "assistant_message": "post_067 has Saturday 3pm available — quiet area with a gym nearby. Please confirm Saturday 3pm."}
    if not visit3_booked and count_tool("contact_poster") == 2:
        return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_067", "time_text": "Saturday 3pm"}}
    if not visit3_booked:
        return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_067", "time_text": "Saturday 3pm"}}

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
        if task_id == "task_negotiation_hidden_budget":
            return _autopolicy_negotiation(observation, tool_names, user_has_replied, remaining)

        if task_id == "task_slot_cancellation_waitlist":
            return _autopolicy_waitlist(observation, tool_names, user_has_replied, remaining)

        if task_id == "task_multi_visit_preference_evolution":
            return _autopolicy_multi_visit(observation, tool_names, user_has_replied, remaining, booked, last_buyer_message)

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

        if task_id == "task_visit_conflict_check":
            if not has_tool("match_location_preference"):
                return {"action_type": "tool_call", "tool_name": "match_location_preference", "tool_arguments": {"post_ids": ["post_142"]}}
            if not has_tool("get_commute_time"):
                return {"action_type": "tool_call", "tool_name": "get_commute_time", "tool_arguments": {"post_ids": ["post_142"]}}
            if not has_tool("check_calendar_slots"):
                return {"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_142"]}}
            if "contact_poster" not in tool_names and "book_viewing" not in tool_names and not user_has_replied:
                return {"action_type": "assistant_message", "assistant_message": "post_142 has Sunday 5pm available (Saturday slots are already booked by other buyers). Please confirm Sunday 5pm if that works for you."}
            if not has_tool("contact_poster"):
                return {"action_type": "tool_call", "tool_name": "contact_poster", "tool_arguments": {"post_id": "post_142", "time_text": "Sunday 5pm"}}
            return {"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_142", "time_text": "Sunday 5pm"}}

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
