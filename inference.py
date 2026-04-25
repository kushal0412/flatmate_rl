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
from typing import Any

from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider

try:
    from flatmate_rl import FlatmateRlAction, FlatmateRlEnv
    from flatmate_rl.server.heuristic_policy import autopolicy_next_request
    from flatmate_rl.server.scenarios import SCENARIOS
except ImportError:
    from __init__ import FlatmateRlAction, FlatmateRlEnv
    from server.heuristic_policy import autopolicy_next_request
    from server.scenarios import SCENARIOS


IMAGE_NAME = os.getenv("IMAGE_NAME") or "flatmate_rl:latest"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
MAX_STEPS_ENV = os.getenv("MAX_STEPS")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a flatmate visit-scheduling broker agent.

    Return exactly one JSON object with this schema:
    {"action_type":"assistant_message","assistant_message":"..."}
    or
    {"action_type":"tool_call","tool_name":"...","tool_arguments":{...}}

    Rules:
    - Return JSON only.
    - Use only tools listed in available_tools.
    - Follow the observation state exactly.
    - Prefer safe, incremental progress toward storing user details, matching listings, and booking visits.
    """
).strip()


def log_start(task_id: str, model: str, source: str) -> None:
    print(f"[START] scenario={task_id} model={model} source={source}", flush=True)


def log_step(
    step: int,
    action: FlatmateRlAction,
    reward: float,
    done: bool,
    status: str,
    phase: str,
    total_reward: float,
    bookings: int,
    violations: int,
    source: str,
    message: str,
    error: str | None,
) -> None:
    error_val = error if error else "null"
    message_clean = re.sub(r"\s+", " ", message or "").strip()
    label = "[TOOL]" if action.action_type == "tool_call" else "[STATE]"
    action_detail = f"tool={action.tool_name}" if action.action_type == "tool_call" else "action=assistant_message"
    print(
        f"{label} "
        f"step={step} {action_detail} "
        f"reward={reward:.2f} total_reward={total_reward:.2f} bookings={bookings} "
        f"violations={violations} phase={phase} status={status} done={str(done).lower()} "
        f"source={source} message={message_clean} error={error_val}",
        flush=True,
    )


def log_end(task_id: str, success: bool, steps: int, total_reward: float, booked_visits: int, final_status: str) -> None:
    print(
        f"[END] scenario={task_id} success={str(success).lower()} steps={steps} "
        f"total_reward={total_reward:.2f} booked_visits={booked_visits} final_status={final_status}",
        flush=True,
    )


def log_turn(step: int, channel: str, role: str, content: str) -> None:
    if role == "assistant":
        speaker = "broker"
        listener = "seller" if channel == "seller" else "buyer"
    else:
        speaker = "seller_simulator" if channel == "seller" else "buyer_simulator"
        listener = "broker"
    message_clean = re.sub(r"\s+", " ", content or "").strip()
    print(
        f"[TURN] step={step} {speaker} -> {listener}: {message_clean}",
        flush=True,
    )


def log_new_chat_entries(step: int, channel: str, history: list[dict[str, Any]], start_idx: int) -> int:
    for entry in history[start_idx:]:
        log_turn(step=step, channel=channel, role=entry.get("role", ""), content=entry.get("content", ""))
    return len(history)


def build_user_prompt(step: int, observation: Any) -> str:
    return textwrap.dedent(
        f"""
        Scenario: {observation.scenario_id}
        Label: {observation.scenario_label}
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
        Violations: {observation.violations}
        Last tool result: {json.dumps(observation.last_tool_result, ensure_ascii=False)}
        Total reward: {observation.total_reward}
        Message: {observation.message}

        Buyer/Broker transcript:
        {json.dumps(observation.buyer_conversation_history[-8:], ensure_ascii=False)}

        Seller/Broker transcript:
        {json.dumps(observation.seller_conversation_history[-8:], ensure_ascii=False)}

        Return the next action as JSON only.
        """
    ).strip()


def parse_action(text: str) -> FlatmateRlAction | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None

    candidates: list[dict[str, Any]] = []
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{.*\}", cleaned, flags=re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)

    for candidate in candidates:
        try:
            return FlatmateRlAction.model_validate(candidate)
        except Exception:
            continue
    return None


def heuristic_action(observation: Any) -> FlatmateRlAction:
    payload = autopolicy_next_request(observation.scenario_id, observation.model_dump())
    if payload is None:
        raise RuntimeError("Heuristic policy produced no action for a non-terminal observation.")
    return FlatmateRlAction.model_validate(payload)


def get_model_action(client: OpenAI, step: int, observation: Any) -> tuple[FlatmateRlAction, str, str | None]:
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
        action = parse_action(text)
        if action is not None:
            return action, "model", None
        return heuristic_action(observation), "heuristic_fallback", f"unparseable_model_output={text!r}"
    except Exception as exc:
        return heuristic_action(observation), "heuristic_fallback", str(exc)


async def run_scenario(env: FlatmateRlEnv, task_id: str, client: OpenAI | None, max_steps: int | None) -> dict[str, Any]:
    result = await env.reset(scenario_id=task_id)
    observation = result.observation
    limit = max_steps if max_steps is not None else 24
    steps_taken = 0
    buyer_logged_count = 0
    seller_logged_count = 0

    log_start(task_id=task_id, model=MODEL_NAME if client else "heuristic", source="model" if client else "heuristic")
    buyer_logged_count = log_new_chat_entries(
        step=0,
        channel="buyer",
        history=observation.buyer_conversation_history,
        start_idx=buyer_logged_count,
    )
    seller_logged_count = log_new_chat_entries(
        step=0,
        channel="seller",
        history=observation.seller_conversation_history,
        start_idx=seller_logged_count,
    )

    for step in range(1, limit + 1):
        if result.done:
            break

        if client is None:
            action = heuristic_action(observation)
            source = "heuristic"
            error = None
        else:
            action, source, error = get_model_action(client=client, step=step, observation=observation)

        result = await env.step(action)
        observation = result.observation
        steps_taken = step
        buyer_logged_count = log_new_chat_entries(
            step=step,
            channel="buyer",
            history=observation.buyer_conversation_history,
            start_idx=buyer_logged_count,
        )
        seller_logged_count = log_new_chat_entries(
            step=step,
            channel="seller",
            history=observation.seller_conversation_history,
            start_idx=seller_logged_count,
        )

        log_step(
            step=step,
            action=action,
            reward=float(result.reward or 0.0),
            done=result.done,
            status=observation.status,
            phase=observation.phase,
            total_reward=float(observation.total_reward),
            bookings=len(observation.booked_visits),
            violations=len(observation.violations),
            source=source,
            message=observation.message,
            error=error,
        )

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
    args = parser.parse_args()

    scenario_ids = args.scenario_ids or [next(iter(SCENARIOS))]
    client = None if args.heuristic_only or not API_KEY else OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await connect_env(IMAGE_NAME, startup_timeout_s=args.startup_timeout)

    try:
        summaries = []
        for task_id in scenario_ids:
            summaries.append(await run_scenario(env=env, task_id=task_id, client=client, max_steps=args.max_steps))
        print("[SUMMARY] " + json.dumps(summaries, ensure_ascii=False), flush=True)
    finally:
        await env.close()


if __name__ == "__main__":
    asyncio.run(main())
