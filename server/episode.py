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
    from .scenarios import POSTS, SCENARIOS
except ImportError:
    from models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState
    from server.heuristic_policy import expected_policy_action
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
]
SELLER_TOOLS = [
    "store_seller_details",
    "match_location_preference",
    "check_table_slot_matches",
    "confirm_seller_match",
    "offer_matched_listing_to_buyer",
    "schedule_table_visit",
]
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

    def reset(self, scenario_id: str | None = None) -> FlatmateRlObservation:
        selected = scenario_id or "task_visit_single"
        self._scenario = deepcopy(SCENARIOS[selected])
        self._posts = {post_id: deepcopy(POSTS[post_id]) for post_id in self._scenario["task_post_ids"]}
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
        self._seller_confirmations = {}
        self._buyer_offer_confirmations = {}
        self._dynamic_post_id = None
        self._searched = False
        self._done = False
        self._total_reward = 0.0
        self._last_action_signature = ""
        self._repeated_action_streak = 0
        self._last_observation = None

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
        return self._apply_expected_flow_penalty(observation, action, expected_action)

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
            payload = {
                "action_type": action.action_type,
                "assistant_message": action.assistant_message,
            }
        else:
            payload = {
                "action_type": action.action_type,
                "tool_name": action.tool_name,
                "tool_arguments": action.tool_arguments,
            }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _apply_expected_flow_penalty(
        self,
        observation: FlatmateRlObservation,
        actual_action: FlatmateRlAction,
        expected_action: FlatmateRlAction | None,
    ) -> FlatmateRlObservation:
        if self._actions_match_expected_flow(actual_action, expected_action):
            return observation

        self._record_violation("expected_flow_violation")
        self._total_reward -= 10.0
        self._done = True
        self._state.done = True
        self._state.status = "failed"
        self._state.total_reward = self._total_reward
        self._state.tool_trace = deepcopy(self._tool_trace)
        payload = observation.model_dump()
        payload["status"] = "failed"
        payload["done"] = True
        payload["violations"] = list(self._violations)
        payload["step_reward"] = float(payload.get("step_reward", 0.0)) - 10.0
        payload["total_reward"] = self._total_reward
        payload["reward"] = float(payload.get("reward", 0.0)) - 10.0
        payload["message"] = (
            f"{observation.message} Expected flow violation. "
            f"Expected next step `{self._describe_action(expected_action)}` but received `{self._describe_action(actual_action)}`."
        ).strip()
        raw_penalized = FlatmateRlObservation.model_validate(payload)
        self._last_observation = raw_penalized
        if self._strict_eval_mode:
            return self._strict_eval_observation(raw_penalized)
        return raw_penalized

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

        if self._scenario["task_id"] == "task_visit_single_hidden_flex":
            alternatives_offered = any(slot.lower() in lowered for slot in ["saturday", "sunday"])
            if alternatives_offered and "hidden_flex_revealed" not in self._state.gathered_fields:
                self._state.gathered_fields.append("hidden_flex_revealed")
            if alternatives_offered:
                if "sunday 5pm" in lowered:
                    return "I can make Sunday 5pm work, so I confirm Sunday 5pm."
                if "saturday 1pm" in lowered:
                    return "Saturday 1pm works for me too, so I confirm Saturday 1pm."

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
        if self._scenario["task_id"] == "task_visit_single_hidden_flex" and "hidden_flex_revealed" in self._state.gathered_fields:
            visible_slots.extend(self._scenario["buyer_profile"]["hidden_additional_availability"])
        if self._scenario["task_id"] == "task_visit_single":
            if slot in {"today 7pm", "tomorrow 7pm", "Saturday 11am", "Saturday 4pm"}:
                return True
        if self._scenario["task_id"] == "task_visit_multi":
            if slot in {"tomorrow 7pm", "Saturday 4pm", "Saturday 11am", "Sunday 2pm", "Sunday 4pm", "Sunday 5pm"}:
                return True
        if self._scenario["task_id"] == "task_visit_single_seller_followup":
            return slot in {"Saturday 4pm", "Sunday 5pm"}
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
        rejected_for_slots = []
        buyer = self._scenario["buyer_profile"]
        for post_id in self._scenario["task_post_ids"]:
            post = self._posts[post_id]
            if post["rent"] > buyer["budget_max"]:
                continue
            if post["area"] not in buyer["areas"]:
                continue
            if buyer["dietary"] == "non-veg" and post["diet"] == "veg only":
                continue
            if self._scenario["task_id"] == "task_visit_single_seller_followup":
                buyer_slots = set(buyer["visit_availability"])
                if not any(slot in buyer_slots for slot in post["calendar_slots"]):
                    rejected_for_slots.append(post_id)
                    continue
            results.append(post_id)
        if self._scenario["task_id"] == "task_visit_single_seller_followup" and not results:
            return {
                "tool": "search_posts",
                "success": True,
                "message": "Found 0 current posts compatible with the buyer's visit availability.",
                "post_ids": [],
                "rejected_for_slot_mismatch": rejected_for_slots,
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
        self._state.gathered_fields = ["area", "rent", "listing_type", "calendar_slots"]
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
        results = {}
        for post_id in post_ids:
            post = self._resolve_post(post_id)
            if not post:
                results[post_id] = []
                continue
            self._slots_checked[post_id] = list(post["calendar_slots"])
            results[post_id] = list(post["calendar_slots"])
        return {"tool": "check_calendar_slots", "success": True, "message": "Calendar slots fetched.", "calendar_slots": results}

    def _tool_shortlist(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_ids = list(arguments.get("post_ids", []))
        self._state.selected_posts = post_ids
        return {"tool": "shortlist", "success": True, "message": "Posts shortlisted.", "selected_posts": post_ids}

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
                "role": "user",
                "content": f"Client selected {post_id}. Can we visit at {time_text}?",
            }
        )
        self._poster_confirmations[post_id] = time_text
        poster_message = f"Yes, confirmed. {time_text} works for the visit."
        self._seller_history.append({"role": "assistant", "content": poster_message})
        return {"tool": "contact_poster", "success": True, "message": f"Poster confirmed {time_text}.", "post_id": post_id, "time_text": time_text}

    def _tool_book_viewing(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id = arguments.get("post_id", "")
        time_text = arguments.get("time_text", "")
        if post_id not in self._poster_confirmations or self._poster_confirmations[post_id] != time_text:
            return {"tool": "book_viewing", "success": False, "message": "Poster has not explicitly confirmed this time."}
        if post_id not in self._client_confirmations or self._client_confirmations[post_id] != time_text:
            return {"tool": "book_viewing", "success": False, "message": "Client has not explicitly confirmed this time."}
        if self._scenario["task_id"] == "task_visit_multi" and post_id not in self._state.selected_posts:
            return {"tool": "book_viewing", "success": False, "message": "Client has not chosen this listing."}
        if any(entry["time"] == time_text for entry in self._state.booked_visits):
            return {"tool": "book_viewing", "success": False, "message": "Visit time overlaps an existing booking."}
        self._state.booked_visits.append({"post_id": post_id, "time": time_text})
        if len(self._state.booked_visits) >= self._scenario["ground_truth"]["required_bookings"]:
            self._done = True
            self._state.done = True
            self._state.status = "completed"
        return {"tool": "book_viewing", "success": True, "message": f"Viewing booked for {post_id} at {time_text}.", "booked_visits": deepcopy(self._state.booked_visits)}

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
        post_id = str(arguments.get("post_id") or arguments.get("post") or self._dynamic_post_id or "post_dynamic_followup_1")
        time_text = str(arguments.get("time_text") or arguments.get("time") or arguments.get("slot") or "")
        if time_text:
            return post_id, time_text

        candidate_slots: list[str] = []
        slot_matches = arguments.get("slot_matches")
        if isinstance(slot_matches, dict):
            for key, value in slot_matches.items():
                if not arguments.get("post_id"):
                    post_id = str(key)
                if isinstance(value, list):
                    candidate_slots.extend(str(item) for item in value)
        calendar_slots = arguments.get("calendar_slots")
        if isinstance(calendar_slots, list):
            candidate_slots.extend(str(item) for item in calendar_slots)
        candidate_slots.extend(self._slots_checked.get(post_id, []))

        for preferred in ["Sunday 5pm", "Saturday 4pm"]:
            if preferred in candidate_slots:
                return post_id, preferred
        if candidate_slots:
            return post_id, candidate_slots[0]
        return post_id, time_text

    def _tool_confirm_seller_match(self, arguments: dict[str, Any]) -> dict[str, Any]:
        post_id, time_text = self._infer_followup_post_and_time(arguments)
        post = self._resolve_post(post_id)
        if not post or time_text not in post["calendar_slots"]:
            return {"tool": "confirm_seller_match", "success": False, "message": "Selected seller slot is invalid."}
        self._seller_history.append({"role": "user", "content": f"Can we confirm {time_text} for {post_id}?"})
        self._seller_confirmations[post_id] = time_text
        self._seller_history.append({"role": "assistant", "content": f"Confirmed, {time_text} works from the seller side."})
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

    def _sanitize_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        sanitized = deepcopy(result)
        sanitized.pop("stored_profile", None)
        return sanitized

    def _feedback_summary(self, status: str, message: str, last_tool_result: dict[str, Any]) -> str:
        tool_name = str(last_tool_result.get("tool", ""))
        tool_message = str(last_tool_result.get("message", "")).strip()
        if tool_name == "store_user_details" and "Missing buyer fields:" in tool_message:
            missing = tool_message.split("Missing buyer fields:", 1)[1].strip()
            return f"Ask the buyer for these missing fields before storing details: {missing}."
        if tool_name == "store_seller_details" and "Missing seller fields:" in tool_message:
            missing = tool_message.split("Missing seller fields:", 1)[1].strip()
            return f"Ask the seller for these missing fields before storing details: {missing}."
        if tool_name == "search_posts" and last_tool_result.get("success") and not last_tool_result.get("post_ids"):
            return "No current listing fits the buyer's stored visit availability. Close the buyer conversation before waiting for a seller follow-up."
        if tool_name == "search_posts" and last_tool_result.get("success"):
            return "Search results are ready. Evaluate matching listings next."
        if tool_name == "store_seller_details" and last_tool_result.get("success"):
            post_id = str(last_tool_result.get("post_id", "the new post"))
            return f"Seller profile is stored as {post_id}. Match this post against the stored buyer profile next."
        if tool_name == "match_location_preference" and last_tool_result.get("success") and self._state.phase == "seller":
            return "Location preference has been checked for the new seller post. Check buyer-seller slot overlap next."
        if tool_name == "check_table_slot_matches" and last_tool_result.get("success"):
            return "Buyer-seller slot overlap is available. Confirm one matching slot with the seller next."
        if tool_name == "confirm_seller_match" and last_tool_result.get("success"):
            return "Seller has confirmed the slot. Offer the matched listing and slot back to the buyer next."
        if tool_name == "offer_matched_listing_to_buyer" and last_tool_result.get("success"):
            return "Buyer has confirmed the matched slot. Schedule the table visit next."
        if tool_name == "check_calendar_slots" and last_tool_result.get("success"):
            return "Calendar slots are available. Ask the buyer to confirm one matching time before contacting the poster."
        if tool_name == "contact_poster" and last_tool_result.get("success"):
            return "The poster has confirmed the requested time. Book only after the buyer explicitly confirms the same slot."
        if tool_name == "book_viewing" and last_tool_result.get("success"):
            return "A viewing has been booked successfully."
        if "action_loop_detected" in self._violations:
            return "Repeated identical actions triggered loop protection. Change strategy instead of repeating the same step."
        if "expected_flow_violation" in self._violations:
            return "The broker deviated from the scenario's expected flow. Follow the next required action exactly."
        if self._state.phase == "buyer" and not self._state.buyer_profile_stored:
            missing = self._remaining_fields()
            if missing:
                return f"Collect the buyer information still needed to continue: {', '.join(missing)}."
        if self._state.phase == "seller" and not self._state.seller_profile_stored:
            missing = self._remaining_fields()
            if missing:
                return f"Collect the seller information still needed to continue: {', '.join(missing)}."
        if message:
            return message
        if status == "ready":
            return "Review the visible conversation and take the next valid step."
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
