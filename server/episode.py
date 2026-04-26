"""Deterministic episode logic for Flatmate RL."""

from __future__ import annotations

import os
from copy import deepcopy
import json
import re
from typing import Any

try:
    from ..models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState
    from .heuristic_policy import expected_policy_action
    from .scenario_variants import apply_seed_variant
    from .scenarios import POSTS, SCENARIOS
except ImportError:
    from models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState
    from server.heuristic_policy import expected_policy_action
    from server.scenario_variants import apply_seed_variant
    from server.scenarios import POSTS, SCENARIOS


BUYER_TOOLS = [
    "store_user_details",
    "search_posts",
    "close_buyer_conversation",
    "match_location_preference",
    "get_commute_time",
    "check_calendar_slots",
    "shortlist",
    "contact_poster",
    "book_viewing",
    # Scenario 1: hidden-budget negotiation
    "propose_price_to_buyer",
    "propose_price_to_seller",
    "confirm_negotiated_deal",
    # Scenario 2: slot cancellation waitlist
    "add_to_waitlist",
    "notify_buyer_slot_freed",
    # Scenario 3: multi-visit preference evolution
    "debrief_visit",
    "filter_new_arrivals",
]
SELLER_TOOLS = [
    "store_seller_details",
    "match_location_preference",
    "check_table_slot_matches",
    "confirm_seller_match",
    "offer_matched_listing_to_buyer",
    "schedule_table_visit",
]
ALL_TOOLS = set(BUYER_TOOLS + SELLER_TOOLS)
BUYER_FIELD_KEYWORDS = {
    "budget": ("budget", "rs.", "20,000"),
    "diet": ("diet", "non-veg", "vegetarian"),
    "areas": ("area", "andheri", "jogeshwari"),
    "occupation": ("work", "occupation", "engineer", "job"),
    "visit_availability": ("visit", "availability", "slot", "time"),
}
SELLER_FIELD_KEYWORDS = {
    "area": ("area", "jogeshwari", "andheri"),
    "rent": ("rent", "19,500", "19500"),
    "dietary": ("diet", "non-veg", "veg"),
    "listing_type": ("2bhk", "1bhk", "room", "share", "household"),
    "occupation_requirement": ("working professional", "professionals", "occupation", "fit", "flat is for", "who the flat is for"),
    "calendar_slots": ("slot", "saturday", "sunday", "time"),
}
FIELD_TO_PROFILE_KEY = {
    "budget": "budget_max",
    "diet": "dietary",
    "areas": "areas",
    "occupation": "occupation",
    "visit_availability": "visit_availability",
}


class FlatmateEpisode:
    """Stateful deterministic simulator for broker-style visit scheduling."""

    def __init__(self, strict_eval_mode: bool | None = None) -> None:
        if strict_eval_mode is None:
            strict_eval_mode = os.getenv("STRICT_EVAL_MODE", "").lower() in {"1", "true", "yes", "on"}
        self._strict_eval_mode = strict_eval_mode
        self._state = FlatmateRlState()
        self._scenario: dict[str, Any] = {}
        self._posts: dict[str, dict[str, Any]] = {}
        self._tool_results: list[dict[str, Any]] = []
        self._tool_trace: list[dict[str, Any]] = []
        self._history: list[dict[str, str]] = []
        self._buyer_history: list[dict[str, str]] = []
        self._seller_history: list[dict[str, str]] = []
        self._violations: list[str] = []
        self._matched_posts: dict[str, bool] = {}
        self._slots_checked: dict[str, list[str]] = {}
        self._commutes_checked: dict[str, int] = {}
        self._poster_confirmations: dict[str, str] = {}
        self._client_confirmations: dict[str, str] = {}
        self._seller_profile_fit_confirmations: dict[str, bool] = {}
        self._seller_confirmations: dict[str, str] = {}
        self._buyer_offer_confirmations: dict[str, str] = {}
        self._dynamic_post_id: str | None = None
        self._searched = False
        self._done = False
        self._last_user_message = ""
        self._total_reward = 0.0
        self._last_action_signature = ""
        self._repeated_action_streak = 0
        self._last_observation: FlatmateRlObservation | None = None
        # Scenario 1: hidden-budget negotiation state
        self._negotiation_rounds_buyer: int = 0
        self._negotiation_rounds_seller: int = 0
        self._buyer_price_accepted: int | None = None
        self._seller_price_accepted: int | None = None
        self._negotiated_deal_closed: bool = False
        # Scenario 2: slot cancellation waitlist state
        self._waitlist_active: bool = False
        self._waitlist_post_id: str = ""
        self._waitlist_slot: str = ""
        self._cancellation_fired: bool = False
        # Scenario 3: multi-visit preference evolution state
        self._post_arrivals_fired: set[int] = set()
        self._available_post_ids: list[str] = []

    def reset(self, scenario_id: str | None = None, seed: int | None = None) -> FlatmateRlObservation:
        selected = scenario_id or "task_visit_single"
        base_scenario = deepcopy(SCENARIOS[selected])
        base_posts = {post_id: deepcopy(POSTS[post_id]) for post_id in base_scenario["task_post_ids"]}
        self._scenario, self._posts = apply_seed_variant(base_scenario, base_posts, seed)
        self._tool_results = []
        self._tool_trace = []
        self._history = []
        self._buyer_history = []
        self._seller_history = []
        self._violations = []
        self._matched_posts = {}
        self._slots_checked = {}
        self._commutes_checked = {}
        self._poster_confirmations = {}
        self._client_confirmations = {}
        self._seller_profile_fit_confirmations = {}
        self._seller_confirmations = {}
        self._buyer_offer_confirmations = {}
        self._dynamic_post_id = None
        self._searched = False
        self._done = False
        self._total_reward = 0.0
        self._last_action_signature = ""
        self._repeated_action_streak = 0
        self._last_observation = None
        # Reset scenario-specific state
        self._negotiation_rounds_buyer = 0
        self._negotiation_rounds_seller = 0
        self._buyer_price_accepted = None
        self._seller_price_accepted = None
        self._negotiated_deal_closed = False
        self._waitlist_active = False
        self._waitlist_post_id = ""
        self._waitlist_slot = ""
        self._cancellation_fired = False
        self._post_arrivals_fired = set()
        # Set available post IDs (may be a subset for multi-visit scenario)
        initial_ids = self._scenario.get("scenario_creation_config", {}).get("initial_post_ids")
        if initial_ids is not None:
            self._available_post_ids = list(initial_ids)
        else:
            self._available_post_ids = list(self._scenario["task_post_ids"])

        gathered_fields = self._initial_buyer_fields()
        initial_message = self._scenario["initial_user_message"]
        self._last_user_message = initial_message
        self._history.append({"role": "user", "content": initial_message})
        self._buyer_history.append({"role": "user", "content": initial_message})
        self._state = FlatmateRlState(
            scenario_id=selected,
            phase="buyer",
            status="ready",
            gathered_fields=gathered_fields,
            selected_posts=[],
            booked_visits=[],
            buyer_profile_stored=False,
            seller_profile_stored=False,
            tool_trace=[],
            total_reward=0.0,
            done=False,
        )
        return self._observation(
            status="ready",
            message="Scenario ready.",
            current_user_request=initial_message,
            last_tool_result={},
            reward=0.0,
            done=False,
        )

    def step(self, action: FlatmateRlAction) -> FlatmateRlObservation:
        if self._done:
            return self._observation(
                status="completed",
                message="Episode is finished. Call reset() to start a new scenario.",
                current_user_request="",
                last_tool_result={},
                reward=0.0,
                done=True,
            )
        self._state.step_count += 1
        expected_action = self._expected_flow_action()
        if action.action_type == "assistant_message":
            observation = self._handle_assistant_message(action.assistant_message)
        else:
            observation = self._handle_tool_call(action.tool_name, action.tool_arguments)
        return self._apply_flow_adjustment(observation, action, expected_action)

    def state(self) -> FlatmateRlState:
        return self._state

    def _initial_buyer_fields(self) -> list[str]:
        return list(self._scenario["buyer_profile"]["initial_disclosure_fields"])

    def _phase_tools(self) -> list[str]:
        tools = SELLER_TOOLS if self._state.phase == "seller" else BUYER_TOOLS
        tools = list(tools)
        if self._state.phase == "seller" and self._state.seller_profile_stored:
            tools.remove("store_seller_details")
        if self._state.phase == "buyer" and self._state.buyer_profile_stored:
            tools.remove("store_user_details")
        return tools

    def _required_fields(self) -> list[str]:
        if self._state.phase == "seller":
            return ["area", "rent", "dietary", "listing_type", "occupation_requirement", "calendar_slots"]
        required = list(self._scenario["ground_truth"]["required_info"])
        if self._state.phase == "buyer":
            return [field for field in required if field != "listing_choices" or self._scenario["task_id"] == "task_visit_multi"]
        return required

    def _remaining_fields(self) -> list[str]:
        gathered = set(self._state.gathered_fields)
        remaining = []
        for field in self._required_fields():
            if field == "listing_choices" and not self._searched:
                continue
            if field not in gathered:
                remaining.append(field)
        return remaining

    def _matches_any_slot(self, candidate: str, slots: list[str]) -> bool:
        normalized = candidate.strip().lower()
        for slot in slots:
            slot_normalized = slot.strip().lower()
            if normalized == slot_normalized:
                return True
            if normalized.endswith("7pm") and slot_normalized in {"today 7pm", "tomorrow 7pm"}:
                return True
        return False

    def _all_buyer_slots(self) -> list[str]:
        profile = self._scenario["buyer_profile"]
        slots = list(profile["visit_availability"])
        if self._scenario["task_id"] == "task_visit_single_hidden_flex":
            if self._state.gathered_fields.count("hidden_flex_revealed"):
                slots.extend(profile["hidden_additional_availability"])
        return slots

    def _record_client_confirmation_for_slot(self, slot: str) -> None:
        for post_id, checked_slots in self._slots_checked.items():
            if slot in checked_slots:
                self._client_confirmations[post_id] = slot
                return

    def _record_violation(self, text: str) -> None:
        if text not in self._violations:
            self._violations.append(text)

    def _expected_flow_action(self) -> FlatmateRlAction | None:
        if self._last_observation is None:
            return None
        payload = expected_policy_action(self._scenario["task_id"], self._last_observation.model_dump())
        if payload is None:
            return None
        return FlatmateRlAction.model_validate(payload)

    def _actions_match_expected_flow(self, actual: FlatmateRlAction, expected: FlatmateRlAction | None) -> bool:
        if expected is None:
            return True
        if actual.action_type != expected.action_type:
            return False
        if actual.action_type == "assistant_message":
            return bool(actual.assistant_message.strip())
        return actual.tool_name == expected.tool_name

    def _describe_action(self, action: FlatmateRlAction | None) -> str:
        if action is None:
            return "null"
        if action.action_type == "assistant_message":
            return "assistant_message"
        return action.tool_name

    def _missing_required_args(self, action: FlatmateRlAction) -> list[str]:
        if action.action_type != "tool_call":
            return []
        args = action.tool_arguments
        tool_name = action.tool_name
        if tool_name in {"contact_poster", "book_viewing"}:
            return [field for field in ["post_id", "time_text"] if not args.get(field)]
        if tool_name in {"match_location_preference", "get_commute_time", "check_calendar_slots"} and self._state.phase == "buyer":
            return ["post_ids"] if not args.get("post_ids") else []
        return []

    def _is_redundant_successful_tool_call(self, action: FlatmateRlAction) -> bool:
        if action.action_type != "tool_call":
            return False
        current_args = json.dumps(action.tool_arguments or {}, ensure_ascii=False, sort_keys=True)
        for trace in self._tool_trace[-6:-1]:
            if not trace.get("success"):
                continue
            previous_args = json.dumps(trace.get("args") or {}, ensure_ascii=False, sort_keys=True)
            if trace.get("tool") == action.tool_name and previous_args == current_args:
                return True
        return False

    def _book_viewing_violation_category(self, action: FlatmateRlAction) -> tuple[str, str] | None:
        if action.action_type != "tool_call" or action.tool_name != "book_viewing":
            return None
        post_id = str(action.tool_arguments.get("post_id", ""))
        time_text = str(action.tool_arguments.get("time_text", ""))
        checked_slots = self._slots_checked.get(post_id, [])
        if not checked_slots:
            return "missing_prerequisite", "book_viewing requires a successful check_calendar_slots for that post first"
        if time_text not in checked_slots:
            return "calendar_mismatch", f"book_viewing slot {time_text or '<missing>'} was not returned by check_calendar_slots for {post_id or '<missing>'}"
        if self._poster_confirmations.get(post_id) != time_text or self._client_confirmations.get(post_id) != time_text:
            return "consent_violation", "book_viewing requires both buyer and poster confirmation for the same slot"
        return None

    def _classify_flow_adjustment(
        self,
        observation: FlatmateRlObservation,
        actual_action: FlatmateRlAction,
        expected_action: FlatmateRlAction | None,
    ) -> tuple[str, float | None, bool, str] | None:
        if actual_action.action_type == "tool_call":
            if actual_action.tool_name not in ALL_TOOLS:
                return "hallucination", -1.0, True, f"unknown tool {actual_action.tool_name}"
            missing_args = self._missing_required_args(actual_action)
            if missing_args:
                return "hallucination", -1.0, True, f"{actual_action.tool_name} missing required args: {', '.join(missing_args)}"

            if not (observation.done and "action_loop_detected" in self._violations):
                booking_violation = self._book_viewing_violation_category(actual_action)
                if booking_violation is not None:
                    category, detail = booking_violation
                    return category, -0.5, False, detail

            last_message = str(observation.last_tool_result.get("message", "")).lower()
            if "must be called before" in last_message or "before closing" in last_message:
                return "missing_prerequisite", -0.5, False, observation.last_tool_result.get("message", "")

            if self._is_redundant_successful_tool_call(actual_action):
                return "redundant_tool_call", -0.05, False, f"repeated successful {actual_action.tool_name} call within last 5 steps"

        if self._actions_match_expected_flow(actual_action, expected_action):
            if float(observation.step_reward) >= 0.0:
                return "on_canonical_path", 0.1, False, "matched expected action"
            return "on_canonical_path", None, False, "matched expected action"

        expected = self._describe_action(expected_action)
        got = self._describe_action(actual_action)
        return "non_canonical_order", -0.1, False, f"expected {expected}, got {got}"

    def _apply_flow_adjustment(
        self,
        observation: FlatmateRlObservation,
        actual_action: FlatmateRlAction,
        expected_action: FlatmateRlAction | None,
    ) -> FlatmateRlObservation:
        adjustment = self._classify_flow_adjustment(observation, actual_action, expected_action)
        if adjustment is None:
            return observation

        category, replacement_reward, terminate, detail = adjustment
        if category == "on_canonical_path" and replacement_reward is None:
            return observation

        if category != "on_canonical_path":
            self._record_violation(category)

        payload = observation.model_dump()
        previous_reward = float(payload.get("step_reward", 0.0))
        if replacement_reward is not None:
            reward_delta = replacement_reward - previous_reward
            self._total_reward += reward_delta
            payload["step_reward"] = replacement_reward
            payload["reward"] = replacement_reward
        else:
            reward_delta = 0.0

        if terminate:
            self._done = True
            self._state.done = True
            self._state.status = "failed"
            payload["status"] = "failed"
            payload["done"] = True

        self._state.total_reward = self._total_reward
        self._state.tool_trace = deepcopy(self._tool_trace)
        payload["total_reward"] = self._total_reward
        payload["violations"] = list(self._violations)
        if reward_delta:
            payload["message"] = f"{observation.message} {category}: {detail}.".strip()
        else:
            payload["message"] = observation.message
        adjusted = FlatmateRlObservation.model_validate(payload)
        self._last_observation = adjusted
        if self._strict_eval_mode:
            return self._strict_eval_observation(adjusted)
        return adjusted

    def _action_signature(self, action_type: str, content: str = "", tool_name: str = "", arguments: dict[str, Any] | None = None) -> str:
        if action_type == "assistant_message":
            normalized_message = re.sub(r"\s+", " ", content.strip().lower())
            return f"assistant:{normalized_message}"
        normalized_args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
        return f"tool:{tool_name}:{normalized_args}"

    def _apply_loop_penalty(self, signature: str, reward: float, message: str, status: str, done: bool) -> tuple[float, str, str, bool]:
        if signature == self._last_action_signature:
            self._repeated_action_streak += 1
        else:
            self._last_action_signature = signature
            self._repeated_action_streak = 1

        if self._repeated_action_streak < 3:
            return reward, message, status, done

        penalty = -0.5 * (self._repeated_action_streak - 2)
        self._record_violation("action_loop_detected")
        reward += penalty
        message = f"{message} Loop penalty applied for repeating the same action {self._repeated_action_streak} times."

        if self._repeated_action_streak >= 4:
            self._done = True
            self._state.done = True
            self._state.status = "failed"
            return reward, "Episode terminated due to repeated identical actions.", "failed", True

        return reward, message, status, done

    def _handle_assistant_message(self, message: str) -> FlatmateRlObservation:
        phase_before_message = self._state.phase
        self._history.append({"role": "assistant", "content": message})
        if phase_before_message == "seller":
            self._seller_history.append({"role": "assistant", "content": message})
        else:
            self._buyer_history.append({"role": "assistant", "content": message})
        lowered = message.lower()
        response = ""
        reward = 0.0

        if self._state.phase == "buyer":
            if self._scenario["task_id"] == "task_visit_multi" and "post_" in lowered and ("which" in lowered or "choose" in lowered):
                response = "Let’s pursue post_031 and post_052 first."
                if "listing_choices" not in self._state.gathered_fields:
                    self._state.gathered_fields.append("listing_choices")
                self._state.selected_posts = ["post_031", "post_052"]
                reward = 0.2
            else:
                response = self._buyer_response(message)
        else:
            response = self._seller_response(message)

        self._last_user_message = response
        self._history.append({"role": "user", "content": response})
        if self._state.phase == "seller":
            self._seller_history.append({"role": "user", "content": response})
        else:
            self._buyer_history.append({"role": "user", "content": response})
        done = self._maybe_finish_from_message()
        status = "completed" if done else "user_response"
        reward, response_message, status, done = self._apply_loop_penalty(
            signature=self._action_signature("assistant_message", content=message),
            reward=reward,
            message="User responded.",
            status=status,
            done=done,
        )
        self._total_reward += reward
        return self._observation(
            status=status,
            message=response_message,
            current_user_request=response,
            last_tool_result={},
            reward=reward,
            done=done,
        )

    def _buyer_response(self, message: str) -> str:
        lowered = message.lower()
        profile = self._scenario["buyer_profile"]
        task_id = self._scenario["task_id"]

        if task_id == "task_visit_single_hidden_flex":
            alternatives_offered = any(slot.lower() in lowered for slot in ["saturday", "sunday"])
            if alternatives_offered and "hidden_flex_revealed" not in self._state.gathered_fields:
                self._state.gathered_fields.append("hidden_flex_revealed")
            if alternatives_offered:
                if "sunday 5pm" in lowered:
                    self._record_client_confirmation_for_slot("Sunday 5pm")
                    return "I can make Sunday 5pm work, so I confirm Sunday 5pm."
                if "saturday 1pm" in lowered:
                    self._record_client_confirmation_for_slot("Saturday 1pm")
                    return "Saturday 1pm works for me too, so I confirm Saturday 1pm."

        # Scenario 2: waitlist — fire cancellation notification on first message after add_to_waitlist
        if task_id == "task_slot_cancellation_waitlist":
            if self._waitlist_active and not self._cancellation_fired:
                self._cancellation_fired = True
                freed_slot = self._waitlist_slot
                wl_post = self._waitlist_post_id
                # Make freed slot bookable in subsequent calls
                self._slots_checked[wl_post] = [freed_slot]
                post = self._posts.get(wl_post)
                if post and freed_slot in post.get("pre_booked_slots", []):
                    post["pre_booked_slots"].remove(freed_slot)
                return (
                    f"Thanks for adding me to the waitlist! "
                    f"Oh — I just got a notification that {freed_slot} for {wl_post} has opened up due to a cancellation. "
                    f"Can you please book that slot for me?"
                )

        # Scenario 3: multi-visit — return scripted post-visit feedback when agent asks
        if task_id == "task_multi_visit_preference_evolution":
            booked_ids = [v["post_id"] for v in self._state.booked_visits]
            if any(kw in lowered for kw in ["how was", "what did you think", "how did", "liked the flat", "after visiting"]):
                if len(booked_ids) == 1 and booked_ids[0] == "post_023":
                    return "The area was really noisy — definitely not what I'm looking for. I need somewhere quieter."
                if len(booked_ids) == 2 and booked_ids[1] == "post_052":
                    return "post_052 was nice and quiet, but there is no gym nearby, which is important to me."

        if "confirm" in lowered:
            for post_id, slots in self._slots_checked.items():
                for slot in slots:
                    if slot.lower() in lowered and self._slot_fits_buyer(slot):
                        self._client_confirmations[post_id] = slot
                        return f"I confirm {slot}."

        requested_fields = []
        for field in ["diet", "visit_availability", "occupation", "budget", "areas"]:
            if any(keyword in lowered for keyword in BUYER_FIELD_KEYWORDS[field]):
                requested_fields.append(field)
        if requested_fields:
            response_parts = []
            for field in requested_fields:
                if field == "diet":
                    if "diet" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("diet")
                    response_parts.append("I’m non-vegetarian")
                elif field == "visit_availability":
                    if "visit_availability" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("visit_availability")
                    if self._scenario["task_id"] == "task_visit_single_hidden_flex" and "hidden_flex_revealed" not in self._state.gathered_fields:
                        response_parts.append("right now, Tuesday after 6pm is the slot I had in mind")
                    else:
                        response_parts.append("my visit availability is " + " or ".join(profile["visit_availability"]))
                elif field == "occupation":
                    if "occupation" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("occupation")
                    response_parts.append(f"I work as a {profile['occupation']}")
                elif field == "budget":
                    if "budget" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("budget")
                    response_parts.append(f"my max budget is Rs. {profile['budget_max']}")
                elif field == "areas":
                    if "areas" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("areas")
                    response_parts.append("I prefer " + " or ".join(profile["areas"]))
            if response_parts:
                return ". ".join(response_parts) + "."

        missing = self._remaining_fields()
        if missing:
            next_field = missing[0]
            if next_field == "diet":
                self._state.gathered_fields.append("diet")
                return "I’m non-vegetarian."
            if next_field == "visit_availability":
                self._state.gathered_fields.append("visit_availability")
                return "My visit availability is " + " or ".join(profile["visit_availability"]) + "."
        return "Please continue with suitable options."

    def _seller_response(self, message: str) -> str:
        profile = self._scenario["seller_profile"]
        if not profile:
            return "No seller profile is defined."
        lowered = message.lower()
        if "confirm" in lowered:
            for slot in profile["calendar_slots"]:
                if slot.lower() in lowered:
                    self._seller_confirmations[self._dynamic_post_id or "post_dynamic_followup_1"] = slot
                    return f"Confirmed, {slot} works from the seller side."
        requested_fields = []
        for field in ["dietary", "occupation_requirement", "area", "rent", "listing_type", "calendar_slots"]:
            if any(keyword in lowered for keyword in SELLER_FIELD_KEYWORDS[field]):
                requested_fields.append(field)
        if requested_fields:
            response_parts = []
            for field in requested_fields:
                if field == "dietary":
                    if "dietary" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("dietary")
                    response_parts.append(f"the household is {profile['dietary']}")
                elif field == "occupation_requirement":
                    if "occupation_requirement" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("occupation_requirement")
                    response_parts.append(f"it’s for {profile['occupation_requirement']}")
                elif field == "area":
                    if "area" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("area")
                    response_parts.append(f"the area is {profile['area']}")
                elif field == "rent":
                    if "rent" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("rent")
                    response_parts.append(f"the rent is Rs. {profile['rent']}")
                elif field == "listing_type":
                    if "listing_type" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("listing_type")
                    response_parts.append(f"it is a {profile['listing_type']}")
                elif field == "calendar_slots":
                    if "calendar_slots" not in self._state.gathered_fields:
                        self._state.gathered_fields.append("calendar_slots")
                    response_parts.append("available slots are " + " or ".join(profile["calendar_slots"]))
            if response_parts:
                return ". ".join(response_parts) + "."
        if "description" in lowered or "about" in lowered:
            return profile["description"] + "."
        return "Yes, those listing details are correct."

    def _slot_fits_buyer(self, slot: str) -> bool:
        visible_slots = list(self._scenario["buyer_profile"]["visit_availability"])
        task_id = self._scenario["task_id"]
        if task_id == "task_visit_single_hidden_flex" and "hidden_flex_revealed" in self._state.gathered_fields:
            visible_slots.extend(self._scenario["buyer_profile"]["hidden_additional_availability"])
        if task_id == "task_visit_single":
            if slot in {"today 7pm", "tomorrow 7pm", "Saturday 11am", "Saturday 4pm"}:
                return True
        if task_id == "task_visit_multi":
            if slot in {"tomorrow 7pm", "Saturday 4pm", "Saturday 11am", "Sunday 2pm", "Sunday 4pm", "Sunday 5pm"}:
                return True
        if task_id == "task_visit_single_seller_followup":
            return slot in {"Saturday 4pm", "Sunday 5pm"}
        if task_id == "task_multi_visit_preference_evolution":
            # Buyer is flexible — accepts any slot from the slots we've checked
            return True
        return self._matches_any_slot(slot, visible_slots)

    def _handle_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> FlatmateRlObservation:
        result = self._execute_tool(tool_name, arguments)
        self._tool_results.append(result)
        reward = 0.1 if result.get("success") else -0.2
        self._tool_trace.append(
            {
                "step": self._state.step_count,
                "phase": self._state.phase,
                "tool": tool_name,
                "args": deepcopy(arguments),
                "success": bool(result.get("success")),
                "message": result.get("message", ""),
            }
        )
        done = self._done
        status = "completed" if done else "tool_result"
        reward, step_message, status, done = self._apply_loop_penalty(
            signature=self._action_signature("tool_call", tool_name=tool_name, arguments=arguments),
            reward=reward,
            message=result.get("message", ""),
            status=status,
            done=done,
        )
        self._total_reward += reward
        return self._observation(
            status=status,
            message=step_message,
            current_user_request=self._last_user_message,
            last_tool_result=result,
            reward=reward,
            done=done,
        )

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        phase_tools = self._phase_tools()
        if tool_name not in phase_tools:
            if self._state.phase == "buyer" and tool_name == "store_user_details" and self._state.buyer_profile_stored:
                return {"tool": tool_name, "success": True, "message": "Buyer profile already stored."}
            if self._state.phase == "seller" and tool_name == "store_seller_details" and self._state.seller_profile_stored:
                return {
                    "tool": tool_name,
                    "success": True,
                    "message": "Seller profile already stored.",
                    "post_id": self._dynamic_post_id,
                }
            self._record_violation(f"tool_not_available:{tool_name}")
            return {"tool": tool_name, "success": False, "message": f"Tool {tool_name} is not available in phase {self._state.phase}."}

        if self._state.phase == "buyer" and tool_name != "store_user_details" and not self._state.buyer_profile_stored:
            self._record_violation(f"store_user_details_required_before:{tool_name}")
            return {"tool": tool_name, "success": False, "message": f"store_user_details must be called before {tool_name}."}

        if self._state.phase == "seller" and tool_name != "store_seller_details" and not self._state.seller_profile_stored:
            self._record_violation(f"store_seller_details_required_before:{tool_name}")
            return {"tool": tool_name, "success": False, "message": f"store_seller_details must be called before {tool_name}."}

        handler = getattr(self, f"_tool_{tool_name}")
        return handler(arguments)

    def _tool_store_user_details(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        missing = [field for field in ["budget", "diet", "areas", "occupation", "visit_availability"] if field not in self._state.gathered_fields]
        if missing:
            return {"tool": "store_user_details", "success": False, "message": f"Missing buyer fields: {', '.join(missing)}."}
        self._state.buyer_profile_stored = True
        return {"tool": "store_user_details", "success": True, "message": "Buyer profile stored."}

    def _tool_search_posts(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        self._searched = True
        results = []
        negotiable_results = []
        rejected_for_slots = []
        buyer = self._scenario["buyer_profile"]
        gathered = set(self._state.gathered_fields)
        task_id = self._scenario["task_id"]
        is_negotiation = bool(self._scenario.get("scenario_creation_config", {}).get("negotiation_config"))

        for post_id in self._available_post_ids:
            post = self._posts.get(post_id)
            if post is None:
                continue
            if post["rent"] > buyer["budget_max"]:
                if is_negotiation and post.get("negotiable"):
                    negotiable_results.append(post_id)
                continue
            if post["area"] not in buyer["areas"]:
                continue
            if buyer["dietary"] == "non-veg" and post["diet"] == "veg only":
                continue
            # Multi-visit scenario: filter by discovered amenity preferences
            if task_id == "task_multi_visit_preference_evolution":
                amenities = post.get("amenities", {})
                if "quiet_area" in gathered and not amenities.get("quiet"):
                    continue
                if "gym_nearby" in gathered and not amenities.get("gym_nearby"):
                    continue
            if task_id == "task_visit_single_seller_followup":
                buyer_slots = set(buyer["visit_availability"])
                if not any(slot in buyer_slots for slot in post["calendar_slots"]):
                    rejected_for_slots.append(post_id)
                    continue
            results.append(post_id)

        if task_id == "task_visit_single_seller_followup" and not results:
            return {
                "tool": "search_posts",
                "success": True,
                "message": "Found 0 current posts compatible with the buyer's visit availability.",
                "post_ids": [],
                "rejected_for_slot_mismatch": rejected_for_slots,
            }
        if negotiable_results:
            return {
                "tool": "search_posts",
                "success": True,
                "message": (
                    f"Found {len(results)} posts within budget and "
                    f"{len(negotiable_results)} above budget but open to negotiation."
                ),
                "post_ids": results,
                "negotiable_post_ids": negotiable_results,
            }
        return {"tool": "search_posts", "success": True, "message": f"Found {len(results)} matching posts.", "post_ids": results}

    def _tool_close_buyer_conversation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        if self._scenario["task_id"] != "task_visit_single_seller_followup":
            return {
                "tool": "close_buyer_conversation",
                "success": False,
                "message": "Buyer conversation can only be closed this way in seller follow-up scenarios.",
            }
        if not self._searched:
            return {
                "tool": "close_buyer_conversation",
                "success": False,
                "message": "Search existing posts before closing the buyer conversation.",
            }

        buyer_closure = (
            "None of the current listings fit your weekend availability. "
            "I will follow up if a suitable listing comes in."
        )
        seller_message = self._scenario["seller_initial_message"]
        self._history.append({"role": "assistant", "content": buyer_closure})
        self._buyer_history.append({"role": "assistant", "content": buyer_closure})
        self._history.append({"role": "user", "content": seller_message})
        self._seller_history.append({"role": "user", "content": seller_message})
        self._last_user_message = seller_message
        self._state.phase = "seller"
        self._state.gathered_fields = ["area", "rent", "listing_type"]
        return {
            "tool": "close_buyer_conversation",
            "success": True,
            "message": "Buyer conversation closed; seller follow-up started.",
        }

    def _tool_match_location_preference(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        if not post_ids and self._state.phase == "seller" and self._dynamic_post_id:
            post_ids = [self._dynamic_post_id]
        buyer_areas = set(self._scenario["buyer_profile"]["areas"])
        matches = {}
        for post_id in post_ids:
            post = self._resolve_post(post_id)
            if not post:
                matches[post_id] = {"match": False, "reason": "unknown post"}
                continue
            matches[post_id] = {"match": post["area"] in buyer_areas}
            self._matched_posts[post_id] = matches[post_id]["match"]
        return {"tool": "match_location_preference", "success": True, "message": "Location matches evaluated.", "matches": matches}

    def _tool_get_commute_time(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        results = {}
        for post_id in post_ids:
            post = self._resolve_post(post_id)
            if not post:
                results[post_id] = None
                continue
            commute = post["commute_to_goregaon_mins"]
            self._commutes_checked[post_id] = commute
            results[post_id] = commute
        return {"tool": "get_commute_time", "success": True, "message": "Commute times fetched.", "commute_minutes": results}

    def _tool_check_calendar_slots(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        available_by_post: dict[str, list[str]] = {}
        pre_booked_by_post: dict[str, list[str]] = {}
        any_conflicts = False
        for post_id in post_ids:
            post = self._resolve_post(post_id)
            if not post:
                available_by_post[post_id] = []
                continue
            all_slots = list(post["calendar_slots"])
            pre_booked = list(post.get("pre_booked_slots", []))
            available = [s for s in all_slots if s not in pre_booked]
            self._slots_checked[post_id] = available
            available_by_post[post_id] = available
            if pre_booked:
                pre_booked_by_post[post_id] = pre_booked
                any_conflicts = True
        result: dict[str, Any] = {
            "tool": "check_calendar_slots",
            "success": True,
            "message": "Calendar slots fetched. Some slots are already booked by other buyers." if any_conflicts else "Calendar slots fetched.",
            "calendar_slots": available_by_post,
        }
        if any_conflicts:
            result["pre_booked_slots"] = pre_booked_by_post
        return result

    def _tool_shortlist(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        self._state.selected_posts = post_ids
        return {"tool": "shortlist", "success": True, "message": "Posts shortlisted.", "selected_posts": post_ids}

    def _buyer_profile_summary_for_seller(self) -> str:
        profile = self._scenario["buyer_profile"]
        return (
            f"buyer profile: budget up to Rs. {profile['budget_max']}; "
            f"dietary preference {profile['dietary']}; "
            f"preferred areas {', '.join(profile['areas'])}; "
            f"occupation {profile['occupation']}; "
            f"visit availability {', '.join(profile['visit_availability'])}"
        )

    def _tool_contact_poster(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id = arguments.get("post_id", "")
        time_text = arguments.get("time_text", "")
        post = self._resolve_post(post_id)
        if not post:
            return {"tool": "contact_poster", "success": False, "message": f"Unknown post {post_id}."}
        slots = self._slots_checked.get(post_id, [])
        if not time_text or time_text not in slots:
            return {"tool": "contact_poster", "success": False, "message": "Time must come from check_calendar_slots."}
        self._seller_history.append(
            {
                "role": "assistant",
                "content": (
                    f"Client selected {post_id}. Please review this {self._buyer_profile_summary_for_seller()}. "
                    f"Can you confirm the buyer profile is acceptable and that we can visit at {time_text}?"
                ),
            }
        )
        self._poster_confirmations[post_id] = time_text
        self._seller_profile_fit_confirmations[post_id] = True
        poster_message = f"Yes, confirmed. The buyer profile is acceptable and {time_text} works for the visit."
        self._seller_history.append({"role": "user", "content": poster_message})
        return {
            "tool": "contact_poster",
            "success": True,
            "message": f"Poster confirmed buyer profile fit and {time_text}.",
            "post_id": post_id,
            "time_text": time_text,
            "buyer_profile_shared": True,
            "seller_profile_fit_confirmed": True,
        }

    def _tool_book_viewing(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id = arguments.get("post_id", "")
        time_text = arguments.get("time_text", "")
        if post_id not in self._poster_confirmations or self._poster_confirmations[post_id] != time_text:
            return {"tool": "book_viewing", "success": False, "message": "Poster has not explicitly confirmed this time."}
        if not self._seller_profile_fit_confirmations.get(post_id):
            return {"tool": "book_viewing", "success": False, "message": "Poster has not confirmed the buyer profile fit."}
        if post_id not in self._client_confirmations or self._client_confirmations[post_id] != time_text:
            return {"tool": "book_viewing", "success": False, "message": "Client has not explicitly confirmed this time."}
        if self._scenario["task_id"] == "task_visit_multi" and post_id not in self._state.selected_posts:
            return {"tool": "book_viewing", "success": False, "message": "Client has not chosen this listing."}
        if any(entry["time"] == time_text for entry in self._state.booked_visits):
            return {"tool": "book_viewing", "success": False, "message": "Visit time overlaps an existing booking."}
        self._state.booked_visits.append({"post_id": post_id, "time": time_text})
        # Fire post-arrival events for multi-visit scenario
        if self._scenario["task_id"] == "task_multi_visit_preference_evolution":
            self._apply_post_arrival_event(len(self._state.booked_visits))
        if len(self._state.booked_visits) >= self._scenario["ground_truth"]["required_bookings"]:
            self._done = True
            self._state.done = True
            self._state.status = "completed"
        return {"tool": "book_viewing", "success": True, "message": f"Viewing booked for {post_id} at {time_text}.", "booked_visits": deepcopy(self._state.booked_visits)}

    # ------------------------------------------------------------------ #
    #  Scenario 1: Hidden-budget negotiation tools                        #
    # ------------------------------------------------------------------ #

    def _tool_propose_price_to_buyer(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_negotiation_hidden_budget":
            return {"tool": "propose_price_to_buyer", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        proposed_rent = int(arguments.get("proposed_rent", 0))
        config = self._scenario["scenario_creation_config"].get("negotiation_config", {})
        buyer_ceiling = config.get("buyer_ceiling", 0)
        self._negotiation_rounds_buyer += 1
        if proposed_rent <= buyer_ceiling:
            self._buyer_price_accepted = proposed_rent
            return {
                "tool": "propose_price_to_buyer",
                "success": True,
                "message": f"Buyer accepted Rs. {proposed_rent} for {post_id}.",
                "accepted": True,
                "proposed_rent": proposed_rent,
            }
        hint = " I could stretch a little, but not by much." if self._negotiation_rounds_buyer >= 2 else ""
        return {
            "tool": "propose_price_to_buyer",
            "success": True,
            "message": f"Buyer rejected Rs. {proposed_rent} — still too high.{hint}",
            "accepted": False,
            "proposed_rent": proposed_rent,
        }

    def _tool_propose_price_to_seller(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_negotiation_hidden_budget":
            return {"tool": "propose_price_to_seller", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        proposed_rent = int(arguments.get("proposed_rent", 0))
        config = self._scenario["scenario_creation_config"].get("negotiation_config", {})
        seller_floor = config.get("seller_floor", 0)
        self._negotiation_rounds_seller += 1
        self._seller_history.append(
            {
                "role": "assistant",
                "content": f"The buyer is interested in {post_id}. Would you accept Rs. {proposed_rent}?",
            }
        )
        if proposed_rent >= seller_floor:
            self._seller_price_accepted = proposed_rent
            self._seller_history.append({"role": "user", "content": f"Yes, I can accept Rs. {proposed_rent}."})
            return {
                "tool": "propose_price_to_seller",
                "success": True,
                "message": f"Seller accepted Rs. {proposed_rent} for {post_id}.",
                "accepted": True,
                "proposed_rent": proposed_rent,
            }
        hint = " Maybe a small discount is possible." if self._negotiation_rounds_seller >= 2 else ""
        self._seller_history.append({"role": "user", "content": f"I can't go as low as Rs. {proposed_rent}.{hint}"})
        return {
            "tool": "propose_price_to_seller",
            "success": True,
            "message": f"Seller rejected Rs. {proposed_rent} — can't go that low.{hint}",
            "accepted": False,
            "proposed_rent": proposed_rent,
        }

    def _tool_confirm_negotiated_deal(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_negotiation_hidden_budget":
            return {"tool": "confirm_negotiated_deal", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        agreed_rent = int(arguments.get("agreed_rent", 0))
        if self._buyer_price_accepted != agreed_rent:
            return {
                "tool": "confirm_negotiated_deal",
                "success": False,
                "message": f"Buyer has not yet accepted Rs. {agreed_rent}. Check buyer acceptance first.",
            }
        if self._seller_price_accepted != agreed_rent:
            return {
                "tool": "confirm_negotiated_deal",
                "success": False,
                "message": f"Seller has not yet accepted Rs. {agreed_rent}. Check seller acceptance first.",
            }
        self._negotiated_deal_closed = True
        self._state.booked_visits.append({"post_id": post_id, "time": "negotiated_deal", "agreed_rent": agreed_rent})
        self._done = True
        self._state.done = True
        self._state.status = "completed"
        return {
            "tool": "confirm_negotiated_deal",
            "success": True,
            "message": f"Deal confirmed for {post_id} at Rs. {agreed_rent}. Both buyer and seller have agreed.",
            "agreed_rent": agreed_rent,
        }

    # ------------------------------------------------------------------ #
    #  Scenario 2: Slot cancellation waitlist tools                       #
    # ------------------------------------------------------------------ #

    def _tool_add_to_waitlist(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_slot_cancellation_waitlist":
            return {"tool": "add_to_waitlist", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        post = self._resolve_post(post_id)
        if not post:
            return {"tool": "add_to_waitlist", "success": False, "message": f"Unknown post {post_id}."}
        config = self._scenario["scenario_creation_config"].get("cancellation_event", {})
        self._waitlist_active = True
        self._waitlist_post_id = post_id
        self._waitlist_slot = config.get("freed_slot", "")
        return {
            "tool": "add_to_waitlist",
            "success": True,
            "message": f"Buyer added to waitlist for {post_id}. Will notify if a slot opens up.",
            "post_id": post_id,
        }

    def _tool_notify_buyer_slot_freed(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_slot_cancellation_waitlist":
            return {"tool": "notify_buyer_slot_freed", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        slot = str(arguments.get("slot", self._waitlist_slot))
        if not self._cancellation_fired:
            return {"tool": "notify_buyer_slot_freed", "success": False, "message": "No cancellation event has occurred yet for this post."}
        if post_id != self._waitlist_post_id or slot != self._waitlist_slot:
            return {"tool": "notify_buyer_slot_freed", "success": False, "message": f"Freed slot is {self._waitlist_slot} for {self._waitlist_post_id}, not {slot} for {post_id}."}
        # Buyer is considered to have confirmed this slot
        self._client_confirmations[post_id] = slot
        self._slots_checked[post_id] = [slot]
        return {
            "tool": "notify_buyer_slot_freed",
            "success": True,
            "message": f"Buyer notified and confirmed {slot} for {post_id}. Ready to book.",
            "post_id": post_id,
            "slot": slot,
        }

    # ------------------------------------------------------------------ #
    #  Scenario 3: Multi-visit preference evolution tools                 #
    # ------------------------------------------------------------------ #

    def _tool_debrief_visit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_multi_visit_preference_evolution":
            return {"tool": "debrief_visit", "success": False, "message": "Not applicable in this scenario."}
        post_id = str(arguments.get("post_id", ""))
        user_feedback = str(arguments.get("user_feedback", "")).lower()
        new_prefs: list[str] = []
        if any(kw in user_feedback for kw in ["noisy", "noise", "loud"]):
            if "quiet_area" not in self._state.gathered_fields:
                self._state.gathered_fields.append("quiet_area")
                new_prefs.append("quiet_area")
        if any(kw in user_feedback for kw in ["gym", "fitness", "workout"]):
            if "gym_nearby" not in self._state.gathered_fields:
                self._state.gathered_fields.append("gym_nearby")
                new_prefs.append("gym_nearby")
        pref_str = ", ".join(new_prefs) if new_prefs else "none new"
        return {
            "tool": "debrief_visit",
            "success": True,
            "message": f"Visit to {post_id} debriefed. Discovered preferences: {pref_str}.",
            "post_id": post_id,
            "discovered_preferences": new_prefs,
        }

    def _tool_filter_new_arrivals(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._scenario["task_id"] != "task_multi_visit_preference_evolution":
            return {"tool": "filter_new_arrivals", "success": False, "message": "Not applicable in this scenario."}
        post_ids = list(arguments.get("post_ids", []))
        gathered = set(self._state.gathered_fields)
        buyer = self._scenario["buyer_profile"]
        buyer_areas = set(buyer["areas"])
        budget = buyer["budget_max"]
        relevant: list[str] = []
        irrelevant: list[str] = []
        for post_id in post_ids:
            post = self._posts.get(post_id)
            if not post:
                irrelevant.append(post_id)
                continue
            amenities = post.get("amenities", {})
            if post["area"] not in buyer_areas or post["rent"] > budget:
                irrelevant.append(post_id)
                continue
            if "quiet_area" in gathered and not amenities.get("quiet"):
                irrelevant.append(post_id)
                continue
            if "gym_nearby" in gathered and not amenities.get("gym_nearby"):
                irrelevant.append(post_id)
                continue
            relevant.append(post_id)
        return {
            "tool": "filter_new_arrivals",
            "success": True,
            "message": (
                f"Filtered {len(post_ids)} new listings: "
                f"{len(relevant)} relevant, {len(irrelevant)} irrelevant given current preferences."
            ),
            "relevant_post_ids": relevant,
            "irrelevant_post_ids": irrelevant,
        }

    def _apply_post_arrival_event(self, visit_number: int) -> None:
        """Inject new posts into the available pool after a visit milestone (Scenario 3)."""
        config = self._scenario.get("scenario_creation_config", {})
        for event in config.get("post_arrival_events", []):
            if event["after_visit"] == visit_number and visit_number not in self._post_arrivals_fired:
                self._post_arrivals_fired.add(visit_number)
                for new_post_id in event["new_post_ids"]:
                    if new_post_id in POSTS and new_post_id not in self._posts:
                        self._posts[new_post_id] = deepcopy(POSTS[new_post_id])
                    if new_post_id not in self._available_post_ids:
                        self._available_post_ids.append(new_post_id)

    def _tool_store_seller_details(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        missing = [field for field in ["area", "rent", "dietary", "listing_type", "occupation_requirement", "calendar_slots"] if field not in self._state.gathered_fields]
        if missing:
            return {"tool": "store_seller_details", "success": False, "message": f"Missing seller fields: {', '.join(missing)}."}
        self._state.seller_profile_stored = True
        self._dynamic_post_id = "post_dynamic_followup_1"
        seller = self._scenario["seller_profile"]
        self._posts[self._dynamic_post_id] = {
            "id": self._dynamic_post_id,
            "area": seller["area"],
            "rent": seller["rent"],
            "diet": seller["dietary"],
            "type": seller["listing_type"],
            "commute_to_goregaon_mins": seller["commute_to_goregaon_mins"],
            "constraints": list(seller["constraints"]),
            "calendar_slots": list(seller["calendar_slots"]),
            "description": seller["description"],
        }
        return {"tool": "store_seller_details", "success": True, "message": "Seller profile stored.", "post_id": self._dynamic_post_id}

    def _tool_check_table_slot_matches(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        if not post_ids and self._state.phase == "seller" and self._dynamic_post_id:
            post_ids = [self._dynamic_post_id]
        buyer_slots = set(self._scenario["buyer_profile"]["visit_availability"])
        matches = {}
        for post_id in post_ids:
            post = self._resolve_post(post_id)
            if not post:
                matches[post_id] = []
                continue
            overlap = [slot for slot in post["calendar_slots"] if slot in buyer_slots]
            matches[post_id] = overlap
            self._slots_checked[post_id] = list(post["calendar_slots"])
        return {"tool": "check_table_slot_matches", "success": True, "message": "Buyer-seller slot overlap checked.", "slot_matches": matches}

    def _infer_followup_post_and_time(self, arguments: dict[str, Any]) -> tuple[str, str]:
        post_id = str(arguments.get("post_id") or self._dynamic_post_id or "post_dynamic_followup_1")
        time_text = str(arguments.get("time_text") or "")

        if not time_text:
            slot_matches = arguments.get("slot_matches")
            if isinstance(slot_matches, dict):
                for key, value in slot_matches.items():
                    if not arguments.get("post_id"):
                        post_id = str(key)
                    if isinstance(value, list) and value:
                        time_text = str(value[0])
                        break
            if not time_text:
                calendar_slots = arguments.get("calendar_slots")
                if isinstance(calendar_slots, list) and calendar_slots:
                    time_text = str(calendar_slots[0])

        return post_id, time_text

    def _tool_confirm_seller_match(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id, time_text = self._infer_followup_post_and_time(arguments)
        post = self._resolve_post(post_id)
        if not post or time_text not in post["calendar_slots"]:
            return {"tool": "confirm_seller_match", "success": False, "message": "Selected seller slot is invalid."}
        self._seller_history.append({"role": "assistant", "content": f"Can we confirm {time_text} for {post_id}?"})
        self._seller_confirmations[post_id] = time_text
        self._seller_history.append({"role": "user", "content": f"Confirmed, {time_text} works from the seller side."})
        return {"tool": "confirm_seller_match", "success": True, "message": f"Seller confirmed {time_text}.", "post_id": post_id, "time_text": time_text}

    def _tool_offer_matched_listing_to_buyer(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id, time_text = self._infer_followup_post_and_time(arguments)
        if self._seller_confirmations.get(post_id) != time_text:
            return {"tool": "offer_matched_listing_to_buyer", "success": False, "message": "Seller has not confirmed this slot yet."}
        self._buyer_offer_confirmations[post_id] = time_text
        return {"tool": "offer_matched_listing_to_buyer", "success": True, "message": f"Buyer confirmed {time_text} for {post_id}.", "post_id": post_id, "time_text": time_text}

    def _tool_schedule_table_visit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id, time_text = self._infer_followup_post_and_time(arguments)
        if self._seller_confirmations.get(post_id) != time_text:
            return {"tool": "schedule_table_visit", "success": False, "message": "Seller confirmation missing for this slot."}
        if self._buyer_offer_confirmations.get(post_id) != time_text:
            return {"tool": "schedule_table_visit", "success": False, "message": "Buyer confirmation missing for this slot."}
        self._state.booked_visits.append({"post_id": post_id, "time": time_text})
        self._done = True
        self._state.done = True
        self._state.status = "completed"
        return {"tool": "schedule_table_visit", "success": True, "message": f"Viewing booked for {post_id} at {time_text}.", "booked_visits": deepcopy(self._state.booked_visits)}

    def _resolve_post(self, post_id: str) -> dict[str, Any] | None:
        return self._posts.get(post_id)

    def _maybe_finish_from_message(self) -> bool:
        if len(self._state.booked_visits) >= self._scenario["ground_truth"]["required_bookings"]:
            self._done = True
            self._state.done = True
            self._state.status = "completed"
            return True
        return False

    def _profile_stored(self) -> bool:
        return self._state.seller_profile_stored if self._state.phase == "seller" else self._state.buyer_profile_stored

    def _prerequisites_satisfied(self) -> dict[str, bool]:
        return {
            "details_stored": self._profile_stored(),
            "posts_searched": self._searched,
            "location_matched": any(self._matched_posts.values()),
            "slots_checked": bool(self._slots_checked),
            "buyer_confirmed": bool(self._client_confirmations or self._buyer_offer_confirmations),
            "poster_confirmed": bool(self._poster_confirmations or self._seller_confirmations),
        }

    def _tool_arguments_summary(self, arguments: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key, value in arguments.items():
            if isinstance(value, list):
                summary[key] = value if len(value) <= 3 else [*value[:3], f"... {len(value) - 3} more"]
            elif isinstance(value, dict):
                summary[key] = f"{len(value)} keys"
            else:
                summary[key] = value
        return summary

    def _recent_tool_calls(self) -> list[dict[str, Any]]:
        return [
            {
                "tool_name": trace.get("tool", ""),
                "tool_arguments_summary": self._tool_arguments_summary(dict(trace.get("args") or {})),
                "success": bool(trace.get("success")),
            }
            for trace in self._tool_trace[-5:]
        ]

    def _sanitize_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        sanitized = deepcopy(result)
        sanitized.pop("stored_profile", None)
        return sanitized

    def _feedback_summary(self, status: str, message: str, last_tool_result: dict[str, Any]) -> str:
        tool_name = str(last_tool_result.get("tool", ""))
        tool_message = str(last_tool_result.get("message", "")).strip()
        success = bool(last_tool_result.get("success"))

        if tool_name == "store_user_details" and "Missing buyer fields:" in tool_message:
            missing = tool_message.split("Missing buyer fields:", 1)[1].strip()
            return f"store_user_details failed: missing fields {missing}."
        if tool_name == "store_seller_details" and "Missing seller fields:" in tool_message:
            missing = tool_message.split("Missing seller fields:", 1)[1].strip()
            return f"store_seller_details failed: missing fields {missing}."
        if tool_name == "search_posts" and success and not last_tool_result.get("post_ids"):
            return "search_posts returned 0 results."
        if tool_name == "search_posts" and success:
            return f"search_posts returned {len(last_tool_result.get('post_ids', []))} result(s)."
        if tool_name == "store_seller_details" and success:
            post_id = str(last_tool_result.get("post_id", ""))
            return f"Seller profile stored{(' as ' + post_id) if post_id else ''}."
        if tool_name == "confirm_negotiated_deal" and success:
            return f"Deal confirmed at Rs. {last_tool_result.get('agreed_rent', '?')}."
        if tool_name == "add_to_waitlist" and success:
            return f"Buyer added to waitlist for {last_tool_result.get('post_id', '?')}."
        if tool_name == "notify_buyer_slot_freed" and success:
            return f"Buyer notified of freed slot {last_tool_result.get('slot', '?')} — ready to book."
        if tool_name == "debrief_visit" and success:
            prefs = last_tool_result.get("discovered_preferences", [])
            return f"debrief_visit succeeded. Discovered: {', '.join(prefs) if prefs else 'no new preferences'}."
        if tool_name == "filter_new_arrivals" and success:
            rel = last_tool_result.get("relevant_post_ids", [])
            return f"filter_new_arrivals: {len(rel)} relevant listing(s) found."
        if tool_name in {"match_location_preference", "check_table_slot_matches", "confirm_seller_match",
                         "offer_matched_listing_to_buyer", "check_calendar_slots", "contact_poster",
                         "propose_price_to_buyer", "propose_price_to_seller", "shortlist"} and success:
            return f"{tool_name} succeeded."
        if tool_name == "book_viewing" and success:
            return "Viewing booked."
        if "action_loop_detected" in self._violations:
            return "Loop detected: identical action repeated. Try a different action."
        if self._state.phase == "buyer" and not self._state.buyer_profile_stored:
            missing = self._remaining_fields()
            if missing:
                return f"Missing buyer fields: {', '.join(missing)}."
        if self._state.phase == "seller" and not self._state.seller_profile_stored:
            missing = self._remaining_fields()
            if missing:
                return f"Missing seller fields: {', '.join(missing)}."
        if message:
            return message
        if status == "ready":
            return "Scenario started."
        return ""

    def _strict_eval_observation(self, observation: FlatmateRlObservation) -> FlatmateRlObservation:
        payload = observation.model_dump()
        payload["scenario_id"] = ""
        payload["scenario_label"] = ""
        payload["difficulty"] = ""
        payload["gathered_fields"] = []
        payload["remaining_required_fields"] = []
        payload["violations"] = []
        payload["tool_trace"] = []
        payload["step_reward"] = 0.0
        payload["total_reward"] = 0.0
        payload["last_tool_result"] = self._sanitize_tool_result(payload["last_tool_result"])
        payload["tool_results"] = [self._sanitize_tool_result(item) for item in payload["tool_results"]]
        return FlatmateRlObservation.model_validate(payload)

    def _observation(
        self,
        *,
        status: str,
        message: str,
        current_user_request: str,
        last_tool_result: dict[str, Any],
        reward: float,
        done: bool,
    ) -> FlatmateRlObservation:
        self._state.status = status
        self._state.tool_trace = deepcopy(self._tool_trace)
        self._state.total_reward = self._total_reward
        observation = FlatmateRlObservation(
            status=status,
            scenario_id=self._scenario["task_id"],
            scenario_label=self._scenario["label"],
            difficulty=self._scenario["difficulty"],
            phase=self._state.phase,
            current_user_request=current_user_request,
            last_user_message=self._last_user_message,
            conversation_history=deepcopy(self._history),
            buyer_conversation_history=deepcopy(self._buyer_history),
            seller_conversation_history=deepcopy(self._seller_history),
            last_tool_result=deepcopy(last_tool_result),
            tool_results=deepcopy(self._tool_results),
            tool_trace=deepcopy(self._tool_trace),
            available_tools=self._phase_tools(),
            prerequisites_satisfied=self._prerequisites_satisfied(),
            recent_tool_calls=self._recent_tool_calls(),
            gathered_fields=list(self._state.gathered_fields),
            remaining_required_fields=self._remaining_fields(),
            selected_posts=list(self._state.selected_posts),
            booked_visits=deepcopy(self._state.booked_visits),
            profile_stored=self._profile_stored(),
            buyer_profile_stored=self._state.buyer_profile_stored,
            seller_profile_stored=self._state.seller_profile_stored,
            violations=list(self._violations),
            step_reward=reward,
            total_reward=self._total_reward,
            message=message,
            feedback_summary=self._feedback_summary(status, message, last_tool_result),
            reward=reward,
            done=done,
        )
        self._last_observation = observation
        if self._strict_eval_mode:
            return self._strict_eval_observation(observation)
        return observation
