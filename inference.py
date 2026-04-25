"""Inference runner for the Flatmate RL environment.

Uses the same Docker-backed pattern as `sudoku_rl/inference.py`.
It can either:
- call a chat model for the next action, or
- run purely with the built-in heuristic policy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider

try:
    from flatmate_rl.env_config import load_repo_env
    from flatmate_rl import FlatmateRlAction, FlatmateRlEnv
    from flatmate_rl.server.heuristic_policy import autopolicy_next_request, expected_policy_action
    from flatmate_rl.server.scenarios import SCENARIOS
except ImportError:
    from env_config import load_repo_env
    from __init__ import FlatmateRlAction, FlatmateRlEnv
    from server.heuristic_policy import autopolicy_next_request, expected_policy_action
    from server.scenarios import SCENARIOS

load_repo_env()


IMAGE_NAME = os.getenv("IMAGE_NAME") or "flatmate_rl:latest"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
MAX_STEPS_ENV = os.getenv("MAX_STEPS")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
MALFORMED_ACTION_PENALTY = -0.05

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a flatmate visit-scheduling broker agent.

    Return exactly one JSON object with this schema:
    {"action_type":"assistant_message","assistant_message":"..."}
    or
    {"action_type":"tool_call","tool_name":"...","tool_arguments":{...}}

    Example valid tool call:
    {"action_type":"tool_call","tool_name":"store_user_details","tool_arguments":{}}

    Rules:
    - Return JSON only.
    - Use only tools listed in available_tools.
    - Never put a tool name in action_type.
    - Follow the observation state exactly.
    - If a tool can perform the next required operation, call the tool immediately.
    - Do not send acknowledgement or progress messages such as "I will search now" when a tool call is needed.
    - Prefer safe, incremental progress toward storing user details, matching listings, and booking visits.
    """
).strip()


def log_start(task_id: str, model: str, source: str) -> None:
    print(f"[START] scenario={task_id} model={model} source={source}", flush=True)


def log_end(task_id: str, success: bool, steps: int, total_reward: float, booked_visits: int, final_status: str) -> None:
    print(
        f"[END] scenario={task_id} success={str(success).lower()} steps={steps} "
        f"total_reward={total_reward:.2f} booked_visits={booked_visits} final_status={final_status}",
        flush=True,
    )


def format_action(action: FlatmateRlAction | dict[str, Any] | None) -> str:
    if action is None:
        return "None"
    if isinstance(action, dict):
        try:
            action = FlatmateRlAction.model_validate(action)
        except Exception:
            return json.dumps(action, ensure_ascii=False, sort_keys=True)
    if action.action_type == "tool_call":
        return json.dumps(
            {
                "action_type": action.action_type,
                "tool_name": action.tool_name,
                "tool_arguments": action.tool_arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    return json.dumps(
        {
            "action_type": action.action_type,
            "assistant_message": action.assistant_message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def format_action_with_reasoning(action: FlatmateRlAction | None, reasoning: dict[str, Any] | None) -> str:
    if action is None:
        return "None"
    payload = json.loads(format_action(action))
    if reasoning and "error" not in reasoning:
        payload["reasoning"] = reasoning.get("decision_summary") or reasoning.get("why_this_action_now") or reasoning
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def actions_match(actual: FlatmateRlAction | None, expected: FlatmateRlAction | None) -> bool:
    if actual is None:
        return False
    if expected is None:
        return True
    if actual.action_type != expected.action_type:
        return False
    if actual.action_type == "assistant_message":
        return bool(actual.assistant_message.strip())
    return actual.tool_name == expected.tool_name


def scenario_check_snapshot(task_id: str, observation: Any) -> dict[str, Any]:
    scenario = SCENARIOS[task_id]
    required_tools = scenario["ground_truth"]["required_tool_calls"]
    used_tools = [item.get("tool", "") for item in observation.tool_trace]
    missing_required_tools = [tool for tool in required_tools if tool not in used_tools]
    required_info = scenario["ground_truth"]["required_info"]
    gathered_fields = set(observation.gathered_fields)
    missing_required_info = [field for field in required_info if field not in gathered_fields]
    return {
        "required_bookings": scenario["ground_truth"]["required_bookings"],
        "bookings_so_far": len(observation.booked_visits),
        "required_tools": required_tools,
        "used_tools": used_tools,
        "missing_required_tools": missing_required_tools,
        "required_info": required_info,
        "gathered_fields": list(observation.gathered_fields),
        "missing_required_info": missing_required_info,
        "violations": list(observation.violations),
        "done": bool(observation.done),
    }


def log_verbose_scenario(task_id: str) -> None:
    scenario = SCENARIOS[task_id]
    truth = scenario["ground_truth"]
    buyer = scenario["buyer_profile"]
    creation_config = scenario["scenario_creation_config"]
    print("[VERBOSE] scenario_definition", flush=True)
    print(
        json.dumps(
            {
                "task_id": scenario["task_id"],
                "label": scenario["label"],
                "description": scenario["description"],
                "task_post_ids": scenario["task_post_ids"],
                "buyer_profile": {
                    "budget_max": buyer["budget_max"],
                    "dietary": buyer["dietary"],
                    "areas": buyer["areas"],
                    "occupation": buyer["occupation"],
                    "visit_availability": buyer["visit_availability"],
                    "initial_disclosure_fields": buyer["initial_disclosure_fields"],
                },
                "expected_answers": creation_config.get("expected_answers", {}),
                "followup_seller_expected_answers": creation_config.get("followup_seller_expected_answers", {}),
                "ground_truth": truth,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print("[VERBOSE] scenario_check_functions", flush=True)
    print(
        json.dumps(
            {
                "field_tracking": "server/episode.py:_required_fields, _remaining_fields, _buyer_response",
                "tool_gating": "server/episode.py:_execute_tool",
                "buyer_store_check": "server/episode.py:_tool_store_user_details",
                "search_check": "server/episode.py:_tool_search_posts",
                "slot_fetch_check": "server/episode.py:_tool_check_calendar_slots",
                "poster_confirmation_check": "server/episode.py:_tool_contact_poster",
                "booking_check": "server/episode.py:_tool_book_viewing",
                "completion_check": "server/episode.py:_tool_book_viewing and _maybe_finish_from_message",
                "reward_check": "server/episode.py:_handle_assistant_message and _handle_tool_call",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def log_verbose_step(
    *,
    task_id: str,
    step: int,
    raw_observation: Any,
    policy_observation: Any,
    expected_action: FlatmateRlAction | None,
    actual_action: FlatmateRlAction | None,
    model_raw_response: str | None,
    model_debug_explanation: dict[str, Any] | None,
) -> None:
    user_prompt = build_user_prompt(step=step, observation=policy_observation)
    print(f"[VERBOSE] step={step} pre_step_checks", flush=True)
    print(
        json.dumps(
            {
                "full_state_checks": scenario_check_snapshot(task_id, raw_observation),
                "strict_eval": {
                    "scenario_id_visible": bool(policy_observation.scenario_id),
                    "scenario_label_visible": bool(policy_observation.scenario_label),
                    "gathered_fields_visible": policy_observation.gathered_fields,
                    "remaining_required_fields_visible": policy_observation.remaining_required_fields,
                    "violations_visible": policy_observation.violations,
                    "tool_trace_visible": policy_observation.tool_trace,
                    "total_reward_visible": policy_observation.total_reward,
                    "last_tool_result_visible": policy_observation.last_tool_result,
                },
                "expected_action_from_full_state": format_action(expected_action),
                "actual_action_from_policy_input": format_action(actual_action),
                "action_match": actions_match(actual_action, expected_action),
                "broker_feedback_payload": policy_observation.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"[VERBOSE] step={step} llm_system_prompt", flush=True)
    print(SYSTEM_PROMPT, flush=True)
    print(f"[VERBOSE] step={step} llm_user_prompt", flush=True)
    print(user_prompt, flush=True)
    print(f"[VERBOSE] step={step} llm_raw_response", flush=True)
    print(model_raw_response if model_raw_response is not None else "null", flush=True)
    print(f"[VERBOSE] step={step} llm_decision_explanation", flush=True)
    print(
        json.dumps(model_debug_explanation or {"message": "No model explanation available for this step."}, ensure_ascii=False, indent=2, sort_keys=True),
        flush=True,
    )


def log_verbose_post_step(task_id: str, step: int, observation: Any) -> None:
    print(f"[VERBOSE] step={step} post_step_checks", flush=True)
    print(
        json.dumps(
            scenario_check_snapshot(task_id, observation),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def extract_new_chat_entries(history: list[dict[str, Any]], start_idx: int) -> tuple[list[dict[str, Any]], int]:
    return history[start_idx:], len(history)


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _block(title: str, body: str) -> str:
    return f"{title}\n{body}"


def log_initial_conversation(observation: Any) -> tuple[int, int]:
    buyer_history = observation.buyer_conversation_history
    seller_history = observation.seller_conversation_history
    if buyer_history:
        print(_block("user initiated conversation", _clean_text(buyer_history[0].get("content", ""))), flush=True)
        print("", flush=True)
    if seller_history:
        print(_block("seller initiated conversation", _clean_text(seller_history[0].get("content", ""))), flush=True)
        print("", flush=True)
    return len(buyer_history), len(seller_history)


def log_step_report(
    *,
    step: int,
    action: FlatmateRlAction | None,
    expected_action: FlatmateRlAction | None,
    model_raw_response: str | None,
    model_debug_explanation: dict[str, Any] | None,
    buyer_entries: list[dict[str, Any]],
    seller_entries: list[dict[str, Any]],
    reward: float,
    total_reward: float,
    status: str,
    done: bool,
    tool_result: dict[str, Any],
    message: str,
    source: str,
    error: str | None,
) -> None:
    policy_status = "success" if actions_match(action, expected_action) else "violation"
    print(f"step {step}", flush=True)
    print("", flush=True)

    for entry in buyer_entries:
        label = "broker response :" if entry.get("role") == "assistant" else "buyer_chat :"
        print(_block(label, _clean_text(entry.get("content", ""))), flush=True)
        print("", flush=True)

    for entry in seller_entries:
        label = "broker response :" if entry.get("role") == "assistant" else "seller_chat :"
        print(_block(label, _clean_text(entry.get("content", ""))), flush=True)
        print("", flush=True)

    print(_block("expected_response :", format_action(expected_action)), flush=True)
    print("", flush=True)

    print(_block("llm_raw_response :", model_raw_response or "null"), flush=True)
    print("", flush=True)

    print(_block("llm_parsed_action :", format_action_with_reasoning(action, model_debug_explanation)), flush=True)
    print("", flush=True)

    print(_block("policy_check :", policy_status), flush=True)
    print("", flush=True)

    tool_used = action.tool_name if action is not None and action.action_type == "tool_call" else "none"
    print(_block("tool_used :", tool_used), flush=True)
    print("", flush=True)

    print(_block("tool_result :", json.dumps(tool_result, ensure_ascii=False, sort_keys=True)), flush=True)
    print("", flush=True)

    reward_text = (
        f"step_reward={reward:.2f}\n"
        f"total_reward={total_reward:.2f}\n"
        f"status={status}\n"
        f"done={str(done).lower()}\n"
        f"source={source}\n"
        f"message={_clean_text(message)}\n"
        f"error={error or 'null'}"
    )
    print(_block("reward :", reward_text), flush=True)
    print("\n" + ("-" * 60) + "\n", flush=True)


def build_user_prompt(step: int, observation: Any) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Phase: {observation.phase}
        Status: {observation.status}
        Current user request: {observation.current_user_request}
        Last user message: {observation.last_user_message}
        Available tools: {observation.available_tools}
        Gathered fields: {observation.gathered_fields}
        Remaining required fields: {observation.remaining_required_fields}
        Selected posts: {observation.selected_posts}
        Booked visits: {observation.booked_visits}
        Last tool result: {json.dumps(observation.last_tool_result, ensure_ascii=False)}
        Message: {observation.message}
        Feedback summary: {observation.feedback_summary}

        Buyer/Broker transcript:
        {json.dumps(observation.buyer_conversation_history[-8:], ensure_ascii=False)}

        Seller/Broker transcript:
        {json.dumps(observation.seller_conversation_history[-8:], ensure_ascii=False)}

        Return the next action as JSON only.
        """
    ).strip()


@dataclass
class ParsedAction:
    action: FlatmateRlAction | None
    error: str | None = None
    warning: str | None = None


def _schema_error_details(candidate: dict[str, Any]) -> str | None:
    action_type = candidate.get("action_type")
    if action_type not in {"assistant_message", "tool_call"}:
        return f"action_type must be 'assistant_message' or 'tool_call', got {action_type!r}"
    if action_type == "assistant_message" and not str(candidate.get("assistant_message", "")).strip():
        return "assistant_message is required when action_type is 'assistant_message'"
    if action_type == "tool_call" and not str(candidate.get("tool_name", "")).strip():
        return "tool_name is required when action_type is 'tool_call'"
    return None


def normalize_action_candidate(candidate: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    normalized = dict(candidate)
    warning = None
    action_type = str(normalized.get("action_type", "")).strip()
    tool_name = str(normalized.get("tool_name", "")).strip()

    if not action_type and tool_name:
        normalized["action_type"] = "tool_call"
        normalized.setdefault("tool_arguments", {})
        warning = "coerced missing action_type to tool_call because tool_name was present"
        return normalized, warning

    if action_type in {"assistant", "message"} and "assistant_message" in normalized:
        normalized["action_type"] = "assistant_message"
        normalized.pop("tool_name", None)
        normalized.pop("tool_arguments", None)
        warning = f"coerced action_type {action_type!r} to assistant_message"
        return normalized, warning

    if action_type and action_type not in {"assistant_message", "tool_call"} and not tool_name:
        normalized["action_type"] = "tool_call"
        normalized["tool_name"] = action_type
        normalized.setdefault("tool_arguments", {})
        warning = f"coerced invalid action_type {action_type!r} into tool_name"
        return normalized, warning

    return normalized, warning


def parse_action(text: str, *, strict: bool = True) -> ParsedAction:
    cleaned = (text or "").strip()
    if not cleaned:
        return ParsedAction(action=None, error="json_parse_failed: empty model response")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if strict:
            return ParsedAction(action=None, error=f"json_parse_failed: {exc.msg} at line {exc.lineno} column {exc.colno}")
        parsed = None

    if strict:
        if not isinstance(parsed, dict):
            return ParsedAction(action=None, error=f"schema_validation_failed: expected JSON object, got {type(parsed).__name__}")
        schema_detail = _schema_error_details(parsed)
        if schema_detail:
            return ParsedAction(action=None, error=f"schema_validation_failed: {schema_detail}")
        try:
            return ParsedAction(action=FlatmateRlAction.model_validate(parsed))
        except Exception as exc:
            return ParsedAction(action=None, error=f"schema_validation_failed: {exc}")

    candidates: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        candidates.append(parsed)

    for match in re.finditer(r"\{.*\}", cleaned, flags=re.DOTALL):
        try:
            extracted = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(extracted, dict):
            candidates.append(extracted)

    last_error = "schema_validation_failed: no valid action object found"
    for candidate in candidates:
        normalized, warning = normalize_action_candidate(candidate)
        try:
            return ParsedAction(action=FlatmateRlAction.model_validate(normalized), warning=warning)
        except Exception as exc:
            last_error = f"schema_validation_failed: {exc}"
            continue
    return ParsedAction(action=None, error=last_error)


def malformed_action_observation(observation: Any, details: str) -> Any:
    payload = observation.model_dump()
    malformed_result = {
        "error": "schema_validation_failed",
        "details": details,
        "expected_schema": FlatmateRlAction.model_json_schema(),
    }
    payload["status"] = "tool_result"
    payload["last_tool_result"] = malformed_result
    payload["tool_results"] = list(payload.get("tool_results", [])) + [malformed_result]
    payload["step_reward"] = MALFORMED_ACTION_PENALTY
    payload["reward"] = MALFORMED_ACTION_PENALTY
    payload["total_reward"] = float(payload.get("total_reward", 0.0)) + MALFORMED_ACTION_PENALTY
    payload["message"] = "Malformed action output. Return a valid FlatmateRlAction JSON object."
    payload["feedback_summary"] = "Use action_type='assistant_message' or action_type='tool_call' with a non-empty tool_name."
    payload["done"] = False
    return type(observation).model_validate(payload)


def apply_total_reward_adjustment(observation: Any, adjustment: float) -> Any:
    if not adjustment:
        return observation
    payload = observation.model_dump()
    payload["total_reward"] = float(payload.get("total_reward", 0.0)) + adjustment
    return type(observation).model_validate(payload)


def sanitize_observation_for_policy(observation: Any, strict_eval: bool) -> Any:
    if not strict_eval:
        return observation
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
    payload["last_tool_result"] = {
        key: value
        for key, value in payload["last_tool_result"].items()
        if key != "stored_profile"
    }
    payload["tool_results"] = [
        {key: value for key, value in item.items() if key != "stored_profile"}
        for item in payload["tool_results"]
    ]
    return type(observation).model_validate(payload)


def missing_fields_from_feedback(observation: dict[str, Any]) -> list[str]:
    feedback = " ".join(
        [
            str(observation.get("feedback_summary", "")),
            str(observation.get("message", "")),
            str(observation.get("last_tool_result", {}).get("message", "")),
        ]
    ).lower()
    fields = []
    for field in ["diet", "visit_availability", "occupation", "budget", "areas", "listing_choices"]:
        phrases = {
            "visit_availability": ["visit_availability", "visit availability"],
            "listing_choices": ["listing_choices", "listing choices"],
        }.get(field, [field])
        if any(phrase in feedback for phrase in phrases):
            fields.append(field)
    return fields


def heuristic_action(task_id: str, observation: Any) -> FlatmateRlAction:
    payload = expected_policy_action(task_id, observation.model_dump())
    if payload is None:
        raise RuntimeError("Heuristic policy produced no action for a non-terminal observation.")
    return FlatmateRlAction.model_validate(payload)


def build_explanation_prompt(step: int, observation: Any, action: FlatmateRlAction, raw_response: str) -> str:
    return textwrap.dedent(
        f"""
        You are auditing a broker policy decision in a flatmate scheduling environment.

        Explain the chosen action briefly as structured JSON only.

        Observation:
        {build_user_prompt(step=step, observation=observation)}

        Raw model response:
        {raw_response}

        Parsed chosen action:
        {format_action(action)}

        Return JSON with keys:
        - decision_summary
        - action_type
        - chosen_tool_or_message
        - why_this_action_now
        - evidence_from_state
        - why_not_other_tools
        - risks_or_uncertainties
        """
    ).strip()


def get_model_explanation(
    client: OpenAI,
    step: int,
    observation: Any,
    action: FlatmateRlAction,
    raw_response: str,
) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a debugging assistant. Explain the selected broker action briefly in JSON only. Do not reveal hidden reasoning. Report explicit decision factors from the visible state.",
                },
                {
                    "role": "user",
                    "content": build_explanation_prompt(
                        step=step,
                        observation=observation,
                        action=action,
                        raw_response=raw_response,
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=400,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        return {"message": text}
    except Exception as exc:
        return {"error": str(exc)}


def get_model_action(
    client: OpenAI,
    task_id: str,
    step: int,
    observation: Any,
    explain: bool,
    strict_parsing: bool,
) -> tuple[FlatmateRlAction | None, str, str | None, str, dict[str, Any] | None]:
    user_prompt = build_user_prompt(step=step, observation=observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = parse_action(text, strict=strict_parsing)
        if parsed.warning:
            print(f"[WARN] parser_coercion {parsed.warning}", flush=True)
        if parsed.action is not None:
            explanation = get_model_explanation(client, step, observation, parsed.action, text) if explain else None
            return parsed.action, "model", None, text, explanation
        if strict_parsing:
            explanation = {"error": parsed.error} if parsed.error else None
            return None, "model_parse_error", parsed.error or "invalid_model_output", text, explanation
        fallback_action = heuristic_action(task_id, observation)
        if parsed.error:
            print(f"[WARN] parser_fallback {parsed.error}", flush=True)
        explanation = {
            "message": "Primary model output could not be parsed, so heuristic fallback was used.",
            "raw_model_response": text,
            "parse_error": parsed.error,
            "fallback_action": json.loads(format_action(fallback_action)),
        }
        return fallback_action, "heuristic_fallback", f"unparseable_model_output={text!r}", text, explanation
    except Exception as exc:
        fallback_action = heuristic_action(task_id, observation)
        explanation = {
            "message": "Primary model call failed, so heuristic fallback was used.",
            "fallback_action": json.loads(format_action(fallback_action)),
        }
        return fallback_action, "heuristic_fallback", str(exc), "", explanation


async def run_scenario(
    env: FlatmateRlEnv,
    task_id: str,
    client: OpenAI | None,
    max_steps: int | None,
    strict_eval: bool,
    verbose: bool,
    strict_parsing: bool,
) -> dict[str, Any]:
    result = await env.reset(scenario_id=task_id)
    observation = result.observation
    limit = max_steps if max_steps is not None else 24
    steps_taken = 0
    buyer_logged_count = 0
    seller_logged_count = 0
    local_reward_adjustment = 0.0

    log_start(task_id=task_id, model=MODEL_NAME if client else "heuristic", source="model" if client else "heuristic")
    if verbose:
        log_verbose_scenario(task_id)
    buyer_logged_count, seller_logged_count = log_initial_conversation(observation)

    for step in range(1, limit + 1):
        if result.done:
            break

        policy_observation = sanitize_observation_for_policy(observation, strict_eval=strict_eval)
        expected_action = None
        model_raw_response = None
        model_debug_explanation = None
        expected_payload = expected_policy_action(task_id, observation.model_dump())
        if expected_payload is not None:
            expected_action = FlatmateRlAction.model_validate(expected_payload)
        if client is None:
            action = heuristic_action(task_id, policy_observation)
            source = "heuristic"
            error = None
        else:
            action, source, error, model_raw_response, model_debug_explanation = get_model_action(
                client=client,
                task_id=task_id,
                step=step,
                observation=policy_observation,
                explain=verbose,
                strict_parsing=strict_parsing,
            )
        if verbose:
            log_verbose_step(
                task_id=task_id,
                step=step,
                raw_observation=observation,
                policy_observation=policy_observation,
                expected_action=expected_action,
                actual_action=action,
                model_raw_response=model_raw_response,
                model_debug_explanation=model_debug_explanation,
            )

        if action is None:
            local_reward_adjustment += MALFORMED_ACTION_PENALTY
            observation = malformed_action_observation(observation, error or "invalid_model_output")
            result = SimpleNamespace(observation=observation, reward=MALFORMED_ACTION_PENALTY, done=False)
        else:
            result = await env.step(action)
            observation = result.observation
            observation = apply_total_reward_adjustment(observation, local_reward_adjustment)
            result = SimpleNamespace(observation=observation, reward=result.reward, done=result.done)
        steps_taken = step
        buyer_entries, buyer_logged_count = extract_new_chat_entries(
            observation.buyer_conversation_history,
            buyer_logged_count,
        )
        seller_entries, seller_logged_count = extract_new_chat_entries(
            observation.seller_conversation_history,
            seller_logged_count,
        )

        log_step_report(
            step=step,
            action=action,
            expected_action=expected_action,
            model_raw_response=model_raw_response,
            model_debug_explanation=model_debug_explanation,
            buyer_entries=buyer_entries,
            seller_entries=seller_entries,
            reward=float(result.reward or 0.0),
            total_reward=float(observation.total_reward),
            status=observation.status,
            done=result.done,
            tool_result=observation.last_tool_result,
            source=source,
            message=observation.message,
            error=error,
        )
        if verbose:
            log_verbose_post_step(task_id=task_id, step=step, observation=observation)

        if result.done:
            break

    success = bool(observation.booked_visits) and not observation.violations
    summary = {
        "scenario": task_id,
        "success": success,
        "steps": steps_taken,
        "total_reward": float(observation.total_reward),
        "booked_visits": observation.booked_visits,
        "violations": observation.violations,
        "status": observation.status,
    }
    log_end(
        task_id=task_id,
        success=success,
        steps=steps_taken,
        total_reward=float(observation.total_reward),
        booked_visits=len(observation.booked_visits),
        final_status=observation.status,
    )
    return summary


async def connect_env(image_name: str, startup_timeout_s: float) -> FlatmateRlEnv:
    provider = LocalDockerProvider()
    base_url = provider.start_container(image_name)
    provider.wait_for_ready(base_url, timeout_s=startup_timeout_s)
    env = FlatmateRlEnv(base_url=base_url, provider=provider)
    await env.connect()
    return env


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario id to run. May be provided multiple times.",
    )
    parser.add_argument("--max-steps", type=int, default=int(MAX_STEPS_ENV) if MAX_STEPS_ENV else None)
    parser.add_argument("--heuristic-only", action="store_true", help="Skip model calls and use only the heuristic policy.")
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--strict-eval", action="store_true", help="Hide scenario metadata and reward signals from the broker policy.")
    parser.add_argument(
        "--strict-parsing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject malformed action JSON instead of coercing it. Use --no-strict-parsing for legacy coercion.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print scenario checks, expected actions, and detailed state diagnostics.")
    args = parser.parse_args()

    scenario_ids = args.scenario_ids or [next(iter(SCENARIOS))]
    client = None if args.heuristic_only or not API_KEY else OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await connect_env(IMAGE_NAME, startup_timeout_s=args.startup_timeout)

    try:
        summaries = []
        for task_id in scenario_ids:
            summaries.append(
                await run_scenario(
                    env=env,
                    task_id=task_id,
                    client=client,
                    max_steps=args.max_steps,
                    strict_eval=args.strict_eval,
                    verbose=args.verbose,
                    strict_parsing=args.strict_parsing,
                )
            )
        print("[SUMMARY] " + json.dumps(summaries, ensure_ascii=False), flush=True)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
