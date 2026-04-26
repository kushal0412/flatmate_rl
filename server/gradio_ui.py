"""Thin Gradio renderer for the Flatmate RL OpenEnv environment."""

from __future__ import annotations

import html
import json
import logging
import os
import textwrap
from typing import Any

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

from openenv.core.env_server.serialization import serialize_observation

try:
    from ..env_config import load_repo_env
    from ..models import FlatmateRlAction
    from .scenarios import SCENARIOS
except ImportError:
    from env_config import load_repo_env
    from models import FlatmateRlAction
    from server.scenarios import SCENARIOS

load_repo_env()


logger = logging.getLogger("flatmate_rl.web")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qmeta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
MAX_UI_STEPS = int(os.getenv("MAX_UI_STEPS", os.getenv("MAX_STEPS", "24")))

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a flatmate visit-scheduling broker agent.

    Return exactly one JSON object with this schema:
    {"action_type":"assistant_message","assistant_message":"..."}
    or
    {"action_type":"tool_call","tool_name":"...","tool_arguments":{}}

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

CUSTOM_CSS = """
.flatmate-shell {
  max-width: 1400px;
  margin: 0 auto;
}
.flatmate-status {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  padding: 10px 12px;
  background: var(--background-fill-secondary);
  font-size: 13px;
}
.flatmate-status strong {
  font-weight: 800;
}
"""


def _task_choices() -> list[tuple[str, str]]:
    return [(scenario["label"], scenario_id) for scenario_id, scenario in SCENARIOS.items()]


def _serialize_reset(web_manager, observation) -> dict[str, Any]:
    serialized = serialize_observation(observation)
    state = web_manager.env.state
    web_manager.episode_state.episode_id = state.episode_id
    web_manager.episode_state.step_count = state.step_count
    web_manager.episode_state.current_observation = serialized["observation"]
    web_manager.episode_state.action_logs = []
    web_manager.episode_state.is_reset = True
    return serialized


def _chatbot_rows(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant" if entry.get("role") == "assistant" else "user",
            "content": str(entry.get("content", "")),
        }
        for entry in history
    ]


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _status_html(observation: dict[str, Any], llm_state: dict[str, Any]) -> str:
    action = llm_state.get("parsed_action") or {}
    action_label = "none"
    if action.get("action_type") == "tool_call":
        action_label = f"tool_call:{action.get('tool_name', '')}"
    elif action.get("action_type") == "assistant_message":
        action_label = "assistant_message"
    error = llm_state.get("error") or "none"
    return (
        '<div class="flatmate-status">'
        f'<strong>Model:</strong> {html.escape(MODEL_NAME)} &nbsp; '
        f'<strong>Phase:</strong> {html.escape(str(observation.get("phase", "")))} &nbsp; '
        f'<strong>Status:</strong> {html.escape(str(observation.get("status", "")))} &nbsp; '
        f'<strong>Done:</strong> {html.escape(str(_is_done(observation)).lower())} &nbsp; '
        f'<strong>Reward:</strong> {html.escape(str(observation.get("total_reward", 0.0)))} &nbsp; '
        f'<strong>Last action:</strong> {html.escape(action_label)} &nbsp; '
        f'<strong>LLM error:</strong> {html.escape(error)}'
        "</div>"
    )


def _feedback_payload(observation: dict[str, Any]) -> dict[str, Any]:
    """Return the exact observation payload visible to the broker policy."""
    return observation


def _is_done(observation: dict[str, Any]) -> bool:
    return bool(observation.get("done")) or str(observation.get("status", "")).lower() in {"completed", "failed"}


def _observation_from_serialized(serialized: dict[str, Any]) -> dict[str, Any]:
    observation = dict(serialized.get("observation") or {})
    # OpenEnv web serialization puts reward/done beside observation. Keep those
    # values in the renderer state so the UI and broker prompt see termination.
    observation["reward"] = serialized.get("reward")
    observation["done"] = serialized.get("done", False)
    return observation


def _build_user_prompt(step: int, observation: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}

        OpenEnv observation / broker feedback:
        {_json_text(_feedback_payload(observation))}

        Return the next action as JSON only.
        """
    ).strip()


def _client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN, API_KEY, or OPENAI_API_KEY before running LLM simulation.")
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _parse_action(raw_text: str) -> FlatmateRlAction:
    try:
        payload = json.loads((raw_text or "").strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"json_parse_failed: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"schema_validation_failed: expected JSON object, got {type(payload).__name__}")
    try:
        return FlatmateRlAction.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"schema_validation_failed: {exc}") from exc


def _call_broker_llm(observation: dict[str, Any], step: int) -> tuple[FlatmateRlAction, dict[str, Any]]:
    user_prompt = _build_user_prompt(step=step, observation=observation)
    completion = _client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    raw_text = (completion.choices[0].message.content or "").strip()
    action = _parse_action(raw_text)
    llm_state = {
        "step": step,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "raw_response": raw_text,
        "parsed_action": action.model_dump(exclude_none=True),
        "error": None,
    }
    return action, llm_state


def _error_llm_state(observation: dict[str, Any], step: int, exc: Exception) -> dict[str, Any]:
    return {
        "step": step,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": _build_user_prompt(step=step, observation=observation),
        "raw_response": "",
        "parsed_action": None,
        "error": str(exc),
    }


def _default_ui_state(task_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "observation": {},
        "llm_state": {
            "step": 0,
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": "",
            "raw_response": "",
            "parsed_action": None,
            "error": None,
        },
    }


def _outputs(task_id: str, observation: dict[str, Any], llm_state: dict[str, Any]) -> tuple[Any, ...]:
    state = {
        "task_id": task_id,
        "observation": observation,
        "llm_state": llm_state,
    }
    done = _is_done(observation)
    return (
        _status_html(observation, llm_state),
        _chatbot_rows(observation.get("buyer_conversation_history", [])),
        _chatbot_rows(observation.get("seller_conversation_history", [])),
        _feedback_payload(observation),
        llm_state,
        gr.update(interactive=not done),
        gr.update(interactive=not done),
        state,
    )


async def _ensure_observation(web_manager, task_id: str) -> dict[str, Any]:
    current = dict(web_manager.episode_state.current_observation or {})
    if current and current.get("scenario_id") == task_id:
        return current
    observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset, scenario_id=task_id)
    serialized = _serialize_reset(web_manager, observation)
    await web_manager._send_state_update()
    merged_observation = _observation_from_serialized(serialized)
    web_manager.episode_state.current_observation = merged_observation
    return merged_observation


def build_flatmate_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    del action_fields, metadata, is_chat_env, title, quick_start_md
    if gr is None:  # pragma: no cover
        raise ImportError("gradio is required to build the Flatmate UI.")

    default_task_id = next(iter(SCENARIOS))
    logger.info("build_flatmate_gradio_app:start")

    with gr.Blocks(title="FlatmateEnv LLM Simulation") as demo:
        app_state = gr.State(_default_ui_state(default_task_id))
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        with gr.Column(elem_classes=["flatmate-shell"]):
            gr.Markdown("# FlatmateEnv LLM Simulation")
            status = gr.HTML(value=_status_html({}, _default_ui_state(default_task_id)["llm_state"]))

            with gr.Row():
                task_dropdown = gr.Dropdown(_task_choices(), label="Scenario", value=default_task_id, scale=3)
                reset_btn = gr.Button("Reset", scale=1)
                next_btn = gr.Button("Run Step", variant="primary", scale=1)
                full_btn = gr.Button("Run Complete Simulation", scale=2)

            with gr.Row(equal_height=True):
                buyer_chat = gr.Chatbot(
                    label="Buyer/User LLM ↔ Broker Agent",
                    height=520,
                    scale=1,
                )
                seller_chat = gr.Chatbot(
                    label="Seller LLM ↔ Broker Agent",
                    height=520,
                    scale=1,
                )

            with gr.Row():
                feedback_json = gr.JSON(label="Exact OpenEnv Observation / Feedback Sent To Broker LLM", value={})
                llm_json = gr.JSON(label="Broker LLM Call, Raw Response, Parsed Action", value=_default_ui_state(default_task_id)["llm_state"])

        common_outputs = [
            status,
            buyer_chat,
            seller_chat,
            feedback_json,
            llm_json,
            next_btn,
            full_btn,
            app_state,
        ]

        async def reset_simulation(task_id: str):
            logger.info("callback:reset:start task_id=%s", task_id)
            observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset, scenario_id=task_id)
            serialized = _serialize_reset(web_manager, observation)
            await web_manager._send_state_update()
            merged_observation = _observation_from_serialized(serialized)
            web_manager.episode_state.current_observation = merged_observation
            llm_state = _default_ui_state(task_id)["llm_state"]
            return _outputs(task_id, merged_observation, llm_state)

        async def run_step(task_id: str, ui_state: dict[str, Any]):
            logger.info("callback:run_step:start task_id=%s", task_id)
            observation = await _ensure_observation(web_manager, task_id)
            if _is_done(observation):
                return _outputs(task_id, observation, dict((ui_state or {}).get("llm_state") or {}))

            step = int((ui_state or {}).get("llm_state", {}).get("step") or 0) + 1
            try:
                action, llm_state = _call_broker_llm(observation, step)
            except Exception as exc:
                logger.exception("callback:run_step:llm_error task_id=%s", task_id)
                llm_state = _error_llm_state(observation, step, exc)
                return _outputs(task_id, observation, llm_state)

            serialized = await web_manager.step_environment(action.model_dump(exclude_none=True))
            observation = _observation_from_serialized(serialized)
            web_manager.episode_state.current_observation = observation
            logger.info(
                "callback:run_step:after_step task_id=%s done=%s status=%s",
                task_id,
                observation.get("done"),
                observation.get("status"),
            )
            return _outputs(task_id, observation, llm_state)

        async def run_complete(task_id: str, ui_state: dict[str, Any]):
            logger.info("callback:run_complete:start task_id=%s", task_id)
            observation = await _ensure_observation(web_manager, task_id)
            llm_state = dict((ui_state or {}).get("llm_state") or _default_ui_state(task_id)["llm_state"])
            step = int(llm_state.get("step") or 0)
            yield _outputs(task_id, observation, llm_state)

            for _ in range(MAX_UI_STEPS):
                if _is_done(observation):
                    break
                step += 1
                try:
                    action, llm_state = _call_broker_llm(observation, step)
                except Exception as exc:
                    logger.exception("callback:run_complete:llm_error task_id=%s", task_id)
                    llm_state = _error_llm_state(observation, step, exc)
                    yield _outputs(task_id, observation, llm_state)
                    break
                serialized = await web_manager.step_environment(action.model_dump(exclude_none=True))
                observation = _observation_from_serialized(serialized)
                web_manager.episode_state.current_observation = observation
                yield _outputs(task_id, observation, llm_state)

            logger.info("callback:run_complete:done task_id=%s done=%s", task_id, _is_done(observation))

        reset_btn.click(reset_simulation, inputs=[task_dropdown], outputs=common_outputs)
        next_btn.click(run_step, inputs=[task_dropdown, app_state], outputs=common_outputs)
        full_btn.click(run_complete, inputs=[task_dropdown, app_state], outputs=common_outputs)
        task_dropdown.change(reset_simulation, inputs=[task_dropdown], outputs=common_outputs)

    logger.info("build_flatmate_gradio_app:done")
    return demo
