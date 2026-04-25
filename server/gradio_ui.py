"""Broker-app style debugging UI for the Flatmate RL environment."""

from __future__ import annotations

import html
import json
import logging
from typing import Any

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

from openenv.core.env_server.serialization import serialize_observation

try:
    from .scenarios import POSTS, SCENARIOS
    from .heuristic_policy import autopolicy_next_request
except ImportError:
    from server.scenarios import POSTS, SCENARIOS
    from server.heuristic_policy import autopolicy_next_request


BROKER_MODELS = ["heuristic_debug_policy"]
USER_MODELS = ["openenv_builtin_user"]
DEFAULT_BROKER_MODEL = BROKER_MODELS[0]
DEFAULT_USER_MODEL = USER_MODELS[0]
CHATBOT_USES_MESSAGES = True
logger = logging.getLogger("flatmate_rl.web")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

CUSTOM_CSS = """
.panel-surface {
  border: 1px solid var(--border-color-primary);
  border-radius: 8px;
  background: var(--background-fill-secondary);
  color: var(--body-text-color);
}
.muted-text {
  color: var(--body-text-color-subdued);
}
.task-card {
  padding: 14px 16px;
  margin: 8px 0 14px;
}
.task-card__head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
}
.task-card__title {
  font-size: 18px;
  font-weight: 800;
  color: var(--body-text-color);
}
.task-card__pill {
  font-size: 12px;
  font-weight: 700;
  color: var(--body-text-color);
  background: var(--block-background-fill);
  border: 1px solid var(--border-color-primary);
  border-radius: 999px;
  padding: 2px 8px;
}
.task-card__objective {
  font-size: 14px;
  margin-bottom: 10px;
}
.task-card__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  font-size: 13px;
}
.task-card__section-title {
  font-weight: 700;
  color: var(--body-text-color);
  margin-bottom: 4px;
}
.status-card {
  padding: 12px;
}
.status-card pre {
  white-space: pre-wrap;
  margin: 8px 0 0;
  font-size: 13px;
  color: inherit;
}
.score-reasons,
.score-awards {
  margin-top: 10px;
  font-size: 13px;
}
.score-reasons__title,
.score-awards__title {
  font-weight: 800;
  margin-bottom: 4px;
}
.score-reasons ul,
.score-awards ul {
  margin: 4px 0 0 18px;
  padding: 0;
}
.score-reasons li,
.score-awards li {
  margin: 3px 0;
}
.status-card__label {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 6px;
}
.status-card__score {
  font-size: 28px;
  font-weight: 800;
}
.status-pass {
  background: color-mix(in srgb, var(--color-green-500, #22c55e) 16%, var(--background-fill-primary));
  color: var(--body-text-color);
  border-color: color-mix(in srgb, var(--color-green-500, #22c55e) 45%, var(--border-color-primary));
}
.status-warn {
  background: color-mix(in srgb, var(--color-yellow-500, #eab308) 18%, var(--background-fill-primary));
  color: var(--body-text-color);
  border-color: color-mix(in srgb, var(--color-yellow-500, #eab308) 45%, var(--border-color-primary));
}
.status-fail {
  background: color-mix(in srgb, var(--color-red-500, #ef4444) 16%, var(--background-fill-primary));
  color: var(--body-text-color);
  border-color: color-mix(in srgb, var(--color-red-500, #ef4444) 45%, var(--border-color-primary));
}
.model-status {
  padding: 8px 10px;
}
.final-banner {
  margin-top: 10px;
  padding: 14px;
  font-size: 22px;
  font-weight: 800;
}
.final-banner__reason {
  font-size: 13px;
  font-weight: 500;
}
@media (max-width: 900px) {
  .task-card__grid {
    grid-template-columns: 1fr;
  }
}
"""


def _task_choices() -> list[tuple[str, str]]:
    return [(scenario["label"], scenario_id) for scenario_id, scenario in SCENARIOS.items()]


def _active_scenario(task_id: str) -> dict[str, Any]:
    return SCENARIOS[task_id]


def _serialize_reset(web_manager, observation):
    serialized = serialize_observation(observation)
    state = web_manager.env.state
    web_manager.episode_state.episode_id = state.episode_id
    web_manager.episode_state.step_count = state.step_count
    web_manager.episode_state.current_observation = serialized["observation"]
    web_manager.episode_state.action_logs = []
    web_manager.episode_state.is_reset = True
    return serialized


def _task_definition_html(task_id: str) -> str:
    scenario = _active_scenario(task_id)
    truth = scenario["ground_truth"]
    return (
        '<div class="task-card panel-surface">'
        '<div class="task-card__head">'
        f'<div class="task-card__title">{html.escape(scenario["label"])}</div>'
        f'<div class="task-card__pill">{html.escape(scenario["difficulty"])}</div>'
        "</div>"
        f'<div class="task-card__objective">{html.escape(scenario["description"])}</div>'
        '<div class="task-card__grid">'
        '<div><div class="task-card__section-title">Bookings</div>'
        f'<div>{truth["required_bookings"]}</div></div>'
        '<div><div class="task-card__section-title">Required Tools</div>'
        f'<div>{html.escape(", ".join(truth["required_tool_calls"]))}</div></div>'
        '<div><div class="task-card__section-title">Success Condition</div>'
        f'<div>{html.escape(truth["success_condition"])}</div></div>'
        "</div></div>"
    )


def _chatbot_rows(history: list[dict[str, Any]]) -> list[Any]:
    logger.info(
        "chatbot_rows:start uses_messages=%s history_len=%s sample_roles=%s",
        CHATBOT_USES_MESSAGES,
        len(history),
        [str(entry.get("role", "user")) for entry in history[:5]],
    )
    rows = [
        {
            "role": "assistant" if str(entry.get("role", "user")) == "assistant" else "user",
            "content": str(entry.get("content", "")),
        }
        for entry in history
    ]
    logger.info("chatbot_rows:done format=messages rows_len=%s", len(rows))
    return rows


def _build_chatbot(*, label: str, height: int):
    global CHATBOT_USES_MESSAGES
    logger.info("build_chatbot:start label=%s height=%s", label, height)
    try:
        chatbot = gr.Chatbot(label=label, type="messages", height=height)
        CHATBOT_USES_MESSAGES = True
    except TypeError:
        logger.warning("build_chatbot:type_messages_unsupported label=%s", label)
        chatbot = gr.Chatbot(label=label, height=height)
        # Gradio 6 in Docker still validates fallback chatbots using message dicts.
        CHATBOT_USES_MESSAGES = True
    logger.info(
        "build_chatbot:done label=%s detected_type=%s uses_messages=%s",
        label,
        getattr(chatbot, "type", None),
        CHATBOT_USES_MESSAGES,
    )
    return chatbot


def _user_data_rows(task_id: str, observation: dict[str, Any]) -> list[list[Any]]:
    scenario = _active_scenario(task_id)
    buyer = scenario["scenario_creation_config"]["expected_answers"]
    rows = [
        [
            f"buyer_{task_id}",
            scenario["description"],
            "buyer",
            buyer.get("user_sub_type", "flat"),
            buyer.get("location_pref_type", ""),
            ", ".join(buyer.get("areas", [])),
            buyer.get("budget_max"),
            None,
            buyer.get("price_range_negotiable"),
            buyer.get("is_price_range_fixed"),
            f"dietary={buyer.get('dietary')}; occupation={buyer.get('occupation')}",
        ]
    ]
    if scenario.get("seller_profile"):
        seller = scenario["seller_profile"]
        rows.append(
            [
                "seller_post_dynamic_followup_1",
                seller.get("description", ""),
                "seller",
                "flat",
                "specific_area",
                seller.get("area", ""),
                None,
                seller.get("rent"),
                False,
                True,
                f"dietary={seller.get('dietary')}; fit={seller.get('occupation_requirement')}",
            ]
        )
    return rows


def _user_data_json(task_id: str, observation: dict[str, Any]) -> dict[str, Any]:
    scenario = _active_scenario(task_id)
    payload: dict[str, Any] = {
        f"buyer_{task_id}": scenario["scenario_creation_config"]["expected_answers"],
    }
    if scenario.get("seller_profile"):
        payload["seller_post_dynamic_followup_1"] = scenario["seller_profile"]
    payload["observation_flags"] = {
        "buyer_profile_stored": observation.get("buyer_profile_stored", False),
        "seller_profile_stored": observation.get("seller_profile_stored", False),
    }
    return payload


def _storage_rows(task_id: str, observation: dict[str, Any]) -> list[list[Any]]:
    scenario = _active_scenario(task_id)
    truth = scenario["ground_truth"]
    buyer_stored = bool(observation.get("buyer_profile_stored"))
    seller_stored = bool(observation.get("seller_profile_stored"))
    rows = [
        [
            task_id,
            "stored" if buyer_stored else "pending",
            f"buyer_{task_id}",
            ", ".join(truth["required_info"]),
            "",
            ", ".join(truth["required_info"]) if buyer_stored else "",
            "",
            "{}",
            "" if buyer_stored else "buyer profile not stored yet",
        ]
    ]
    if scenario.get("seller_profile"):
        rows.append(
            [
                task_id,
                "stored" if seller_stored else "pending",
                "seller_post_dynamic_followup_1",
                "area, rent, dietary, listing_type, occupation_requirement, calendar_slots",
                "",
                "area, rent, dietary, listing_type, occupation_requirement, calendar_slots" if seller_stored else "",
                "",
                "{}",
                "" if seller_stored else "seller profile not stored yet",
            ]
        )
    return rows


def _storage_log_html(task_id: str, observation: dict[str, Any]) -> str:
    headers = [
        "task_id",
        "status",
        "buyer_user_id",
        "inserted_fields",
        "skipped_fields",
        "matched_expected_fields",
        "mismatched_expected_fields",
        "ignored_tool_args",
        "failure_reason",
    ]
    rows = _storage_rows(task_id, observation)
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="panel-surface status-card">'
        '<div style="overflow:auto;">'
        '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        f'<thead><tr style="text-align:left; border-bottom:1px solid var(--border-color-primary);">{head_html}</tr></thead>'
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape('' if value is None else str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        '<div class="panel-surface status-card">'
        '<div style="overflow:auto;">'
        '<table style="width:100%; border-collapse:collapse; font-size:13px;">'
        f'<thead><tr style="text-align:left; border-bottom:1px solid var(--border-color-primary);">{head_html}</tr></thead>'
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
        "</div>"
    )


def _json_html(value: Any) -> str:
    return (
        '<div class="panel-surface status-card">'
        f"<pre>{html.escape(json.dumps(value, indent=2, sort_keys=True))}</pre>"
        "</div>"
    )


def _user_data_explorer_html(task_id: str, observation: dict[str, Any]) -> str:
    return _html_table(
        [
            "user_id",
            "user_description",
            "user_type",
            "user_sub_type",
            "location_mode",
            "location_value",
            "buyer_max_budget",
            "seller_min_price",
            "price_range_negotiable",
            "is_price_range_fixed",
            "optional_preferences",
        ],
        _user_data_rows(task_id, observation),
    )


def _scenario_check_rows(task_id: str, observation: dict[str, Any]) -> list[list[Any]]:
    scenario = _active_scenario(task_id)
    buyer = scenario["scenario_creation_config"]["expected_answers"]
    rows = []
    checked_slots = {
        trace["tool"]: trace for trace in observation.get("tool_trace", [])
    }
    for post_id in scenario["task_post_ids"]:
        post = POSTS[post_id]
        match_location = post["area"] in buyer.get("areas", [])
        match_budget = post["rent"] <= buyer.get("budget_max", 0)
        match_diet = not (buyer.get("dietary") == "non-veg" and post["diet"] == "veg only")
        score = sum([match_location, match_budget, match_diet]) / 3
        status = "compatible" if score >= 1 else "partial" if score > 0 else "incompatible"
        reasons = []
        if not match_location:
            reasons.append("area mismatch")
        if not match_budget:
            reasons.append("over budget")
        if not match_diet:
            reasons.append("diet mismatch")
        if not reasons:
            reasons.append("matches scenario constraints")
        rows.append(
            [
                post_id,
                status,
                round(score, 2),
                f"selected={'yes' if post_id in observation.get('selected_posts', []) else 'no'}; booked={'yes' if any(item['post_id']==post_id for item in observation.get('booked_visits', [])) else 'no'}",
                "; ".join(reasons),
            ]
        )
    return rows


def _visit_scheduler_rows(task_id: str, observation: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in observation.get("booked_visits", []):
        rows.append(
            [
                f"buyer_{task_id}",
                f"seller_{item['post_id']}",
                item["post_id"],
                item["time"],
                item["time"],
                item["time"],
                "scheduled",
            ]
        )
    return rows


def _scenario_checks_html(task_id: str, observation: dict[str, Any]) -> str:
    return _html_table(
        ["Post ID", "Scenario Status", "Match Score", "State Flags", "Reasons"],
        _scenario_check_rows(task_id, observation),
    )


def _visit_scheduler_html(task_id: str, observation: dict[str, Any]) -> str:
    return _html_table(
        [
            "buyer_user_id",
            "seller_user_id",
            "post_id",
            "scheduled_date",
            "start_time",
            "end_time",
            "status",
        ],
        _visit_scheduler_rows(task_id, observation),
    )


def _score_html(observation: dict[str, Any]) -> str:
    total_reward = float(observation.get("total_reward", 0.0))
    step_reward = float(observation.get("step_reward", 0.0))
    violations = observation.get("violations", [])
    booked = observation.get("booked_visits", [])
    if observation.get("done"):
        status_class = "status-pass"
    elif violations:
        status_class = "status-fail"
    else:
        status_class = "status-warn"
    point_cuts = violations or ["No explicit point cuts recorded."]
    awards = []
    if observation.get("buyer_profile_stored"):
        awards.append("Buyer profile stored")
    if observation.get("seller_profile_stored"):
        awards.append("Seller profile stored")
    if booked:
        awards.extend(f"Booked {item['post_id']} at {item['time']}" for item in booked)
    if not awards:
        awards.append("No positive milestones yet.")
    award_html = "".join(f"<li>{html.escape(item)}</li>" for item in awards)
    cut_html = "".join(f"<li>{html.escape(item)}</li>" for item in point_cuts)
    return (
        f'<div class="status-card panel-surface {status_class}">'
        '<div class="status-card__label">Episode Reward</div>'
        f'<div class="status-card__score">{total_reward:.2f}</div>'
        f'<pre>step_reward={step_reward:.2f}\nbookings={len(booked)}\nviolations={len(violations)}</pre>'
        '<div class="score-awards"><div class="score-awards__title">Positive Reasons</div><ul>'
        f"{award_html}</ul></div>"
        '<div class="score-reasons"><div class="score-reasons__title">Point Cuts</div><ul>'
        f"{cut_html}</ul></div>"
        "</div>"
    )


def _tool_log_rows(observation: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for trace in observation.get("tool_trace", []):
        rows.append(
            [
                trace.get("step", ""),
                trace.get("tool", ""),
                _format_short_json(trace.get("args", {})),
                trace.get("message", ""),
            ]
        )
    return rows


def _episode_status_dump(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": observation.get("status", "ready"),
        "phase": observation.get("phase", "buyer"),
        "done": observation.get("done", False),
        "scenario_id": observation.get("scenario_id", ""),
        "step_reward": observation.get("step_reward", 0.0),
        "total_reward": observation.get("total_reward", 0.0),
        "last_tool_result": observation.get("last_tool_result", {}),
    }


def _live_env_dump(observation: dict[str, Any]) -> dict[str, Any]:
    return observation


def _post_rows(task_id: str, observation: dict[str, Any]) -> list[list[Any]]:
    rows = []
    selected = set(observation.get("selected_posts", []))
    booked = {item["post_id"] for item in observation.get("booked_visits", [])}
    for post_id in _active_scenario(task_id)["task_post_ids"]:
        post = POSTS[post_id]
        status = "booked" if post_id in booked else "selected" if post_id in selected else "available"
        rows.append([post["id"], post["area"], post["rent"], post["diet"], post["type"], post["commute_to_goregaon_mins"], status])
    return rows


def _model_status_html(broker_model: str, user_model: str) -> str:
    return f'<div id="model-status" class="panel-surface model-status">Broker policy: {html.escape(broker_model)} | User simulator: {html.escape(user_model)}</div>'


def _final_banner_html(observation: dict[str, Any]) -> str:
    if not observation.get("done"):
        return ""
    booked = observation.get("booked_visits", [])
    reason = "Episode completed."
    if booked:
        reason = "Booked visits: " + ", ".join(f"{item['post_id']} at {item['time']}" for item in booked)
    return f'<div class="final-banner panel-surface">Complete<div class="final-banner__reason">{html.escape(reason)}</div></div>'


def _format_short_json(payload: Any) -> str:
    text = json.dumps(payload if payload not in (None, "") else {}, ensure_ascii=False)
    return text if len(text) <= 120 else text[:117] + "..."


def _tool_choice_update(observation: dict[str, Any]):
    tools = observation.get("available_tools", [])
    return gr.update(choices=tools, value=(tools[0] if tools else None))

def _seller_chat_label(task_id: str, post_id: str | None = None) -> str:
    if post_id:
        return f"seller_id=seller_{post_id} broker chat"
    return f"seller_id=seller_{task_id} broker chat"


def _extract_contacted_post_ids(observation: dict[str, Any]) -> list[str]:
    post_ids: list[str] = []
    for trace in observation.get("tool_trace", []):
        args = trace.get("args", {})
        post_id = args.get("post_id")
        if isinstance(post_id, str) and post_id not in post_ids:
            post_ids.append(post_id)
    for item in observation.get("booked_visits", []):
        post_id = item.get("post_id")
        if isinstance(post_id, str) and post_id not in post_ids:
            post_ids.append(post_id)
    return post_ids


def _default_ui_state(task_id: str, observation: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "buyer_chat_filter": "__latest__",
        "post_chat_filter": "__active__",
        "observation": observation or {},
    }


def _normalize_ui_state(task_id: str, observation: dict[str, Any], ui_state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(ui_state or {})
    if state.get("task_id") != task_id:
        state = _default_ui_state(task_id, observation)
    else:
        state.setdefault("buyer_chat_filter", "__latest__")
        state.setdefault("post_chat_filter", "__active__")
        state["task_id"] = task_id
        state["observation"] = observation
    return state


def _buyer_chat_filter_choices(observation: dict[str, Any]) -> list[tuple[str, str]]:
    choices = [("Latest buyer chat", "__latest__")]
    if observation.get("buyer_conversation_history"):
        choices.append(("Buyer Chat 1", "buyer_chat_1"))
    return choices


def _post_chat_filter_choices(task_id: str, observation: dict[str, Any]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    if observation.get("seller_conversation_history"):
        choices.append((_seller_chat_label(task_id), "__seller_lead__"))
    choices.append(("latest seller/post broker chat", "__active__"))
    for post_id in _extract_contacted_post_ids(observation):
        choices.append((_seller_chat_label(task_id, post_id), post_id))
    return choices


def _buyer_chat_filter_update(observation: dict[str, Any], ui_state: dict[str, Any]):
    choices = _buyer_chat_filter_choices(observation)
    valid_values = {value for _, value in choices}
    selected = ui_state.get("buyer_chat_filter") or "__latest__"
    if selected not in valid_values:
        selected = "__latest__"
        ui_state["buyer_chat_filter"] = selected
    return gr.update(choices=choices, value=selected)


def _post_chat_filter_update(task_id: str, observation: dict[str, Any], ui_state: dict[str, Any]):
    choices = _post_chat_filter_choices(task_id, observation)
    valid_values = {value for _, value in choices}
    selected = ui_state.get("post_chat_filter") or "__active__"
    if selected not in valid_values:
        selected = "__active__"
        ui_state["post_chat_filter"] = selected
    return gr.update(choices=choices, value=selected)


def _filtered_buyer_chat(observation: dict[str, Any], ui_state: dict[str, Any]):
    selected = ui_state.get("buyer_chat_filter") or "__latest__"
    label = "User ↔ Broker"
    if selected != "__latest__" and observation.get("buyer_conversation_history"):
        label = "User ↔ Broker (Buyer Chat 1)"
    return gr.update(value=_chatbot_rows(observation.get("buyer_conversation_history", [])), label=label)


def _current_post_chat(task_id: str, observation: dict[str, Any], ui_state: dict[str, Any]):
    selected = ui_state.get("post_chat_filter") or "__active__"
    if selected == "__seller_lead__":
        return gr.update(
            value=_chatbot_rows(observation.get("seller_conversation_history", [])),
            label=_seller_chat_label(task_id),
        )
    if selected != "__active__":
        return gr.update(
            value=_chatbot_rows(observation.get("seller_conversation_history", [])),
            label=_seller_chat_label(task_id, selected),
        )
    active_post = _extract_contacted_post_ids(observation)
    label = _seller_chat_label(task_id, active_post[-1]) if active_post else "latest seller/post broker chat"
    return gr.update(value=_chatbot_rows(observation.get("seller_conversation_history", [])), label=label)


def _ui_values(task_id: str, broker_model: str, user_model: str, payload: dict[str, Any], ui_state: dict[str, Any] | None = None) -> tuple[Any, ...]:
    observation = payload.get("observation", {})
    ui_state = _normalize_ui_state(task_id, observation, ui_state)
    logger.info(
        "ui_values:start task_id=%s broker_model=%s user_model=%s done=%s phase=%s buyer_history=%s seller_history=%s tools=%s violations=%s",
        task_id,
        broker_model,
        user_model,
        observation.get("done"),
        observation.get("phase"),
        len(observation.get("buyer_conversation_history", [])),
        len(observation.get("seller_conversation_history", [])),
        len(observation.get("tool_trace", [])),
        len(observation.get("violations", [])),
    )
    violations_text = "\n".join(observation.get("violations", [])) if observation.get("violations") else "No violations detected."
    done = bool(observation.get("done"))
    next_btn_update = gr.update(interactive=not done)
    full_btn_update = gr.update(interactive=not done)
    values = (
        _task_definition_html(task_id),
        _buyer_chat_filter_update(observation, ui_state),
        _filtered_buyer_chat(observation, ui_state),
        _post_chat_filter_update(task_id, observation, ui_state),
        _current_post_chat(task_id, observation, ui_state),
        _user_data_explorer_html(task_id, observation),
        _json_html(_user_data_json(task_id, observation)),
        _storage_log_html(task_id, observation),
        _scenario_checks_html(task_id, observation),
        _visit_scheduler_html(task_id, observation),
        _score_html(observation),
        _tool_log_rows(observation),
        _episode_status_dump(observation),
        _live_env_dump(observation),
        violations_text,
        _post_rows(task_id, observation),
        _model_status_html(broker_model, user_model),
        _final_banner_html(observation),
        next_btn_update,
        full_btn_update,
        _tool_choice_update(observation),
        ui_state,
    )
    logger.info("ui_values:done task_id=%s outputs=%s", task_id, len(values))
    return values


def build_flatmate_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    del action_fields, metadata, is_chat_env, title, quick_start_md
    if gr is None:  # pragma: no cover
        raise ImportError("gradio is required to build the Flatmate UI.")

    default_task_id = next(iter(SCENARIOS))

    logger.info("build_flatmate_gradio_app:start")
    with gr.Blocks(title="FlatmateEnv - Visit Scheduling Simulator") as demo:
        app_state = gr.State(_default_ui_state(default_task_id, {}))
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(
            """
            # 🏠 FlatmateEnv — Visit Scheduling Simulator
            Multi-agent flatmate visit scheduling evaluation
            """
        )

        model_status = gr.HTML(value=_model_status_html(DEFAULT_BROKER_MODEL, DEFAULT_USER_MODEL), label="Model Status")
        task_definition = gr.HTML(value=_task_definition_html(default_task_id), label="Task Definition")

        gr.Markdown("### Model Configuration")
        with gr.Row():
            task_dropdown = gr.Dropdown(_task_choices(), label="Task", value=default_task_id)
            broker_model = gr.Dropdown(BROKER_MODELS, label="Broker Model", value=DEFAULT_BROKER_MODEL)
            user_model = gr.Dropdown(USER_MODELS, label="User Model", value=DEFAULT_USER_MODEL)

        with gr.Row():
            with gr.Column(scale=6):
                gr.Markdown("## Simulation")
                with gr.Row():
                    with gr.Column(scale=1):
                        buyer_chat_filter = gr.Dropdown(
                            choices=[("Latest buyer chat", "__latest__")],
                            value="__latest__",
                            label="Buyer-Broker Chats",
                        )
                        chatbot = _build_chatbot(label="User ↔ Broker", height=500)
                    with gr.Column(scale=1):
                        post_chat_filter = gr.Dropdown(
                            choices=[("latest seller/post broker chat", "__active__")],
                            value="__active__",
                            label="Seller-Broker Chats",
                        )
                        post_agent_chat = _build_chatbot(label="Broker ↔ Post Owner", height=500)

                with gr.Tabs():
                    with gr.Tab("User Data Explorer"):
                        user_data_table = gr.HTML(
                            value=_user_data_explorer_html(default_task_id, {}),
                            label="User Data Explorer",
                        )
                        user_data_json = gr.HTML(
                            value=_json_html(_user_data_json(default_task_id, {})),
                            label="User Detail Documents",
                        )
                    with gr.Tab("User Detail Storage Log"):
                        user_storage_log = gr.HTML(
                            value=_storage_log_html(default_task_id, {}),
                            label="User Detail Storage Log",
                        )
                    with gr.Tab("Scenario Checks"):
                        scenario_checks = gr.HTML(
                            value=_scenario_checks_html(default_task_id, {}),
                            label="Scenario Checks",
                        )
                    with gr.Tab("Visit Scheduler"):
                        visit_scheduler = gr.HTML(
                            value=_visit_scheduler_html(default_task_id, {}),
                            label="Visit Scheduler",
                        )

                with gr.Row():
                    next_btn = gr.Button("▶ Run Next Turn", variant="primary")
                    full_btn = gr.Button("⚡ Run Full Simulation")
                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset")
                tool_name = gr.Dropdown(choices=[], label="Tool", visible=False)

            with gr.Column(scale=4):
                gr.Markdown("## Live Evaluation")
                gr.Markdown("### Score Panel")
                score_panel = gr.HTML(value=_score_html({}))

                gr.Markdown("### Constraint Violations")
                violations = gr.Textbox(label="Violations & Flags", value="No violations detected.", lines=4, interactive=False)

                final_banner = gr.HTML(value="")

                gr.Markdown("### Tool Call Log")
                tool_log = gr.Dataframe(
                    headers=["Turn", "Tool", "Args Summary", "Result Summary"],
                    value=[],
                    interactive=False,
                )

                gr.Markdown("### Episode Status")
                episode_status = gr.JSON(label="Episode Failure / Status", value=_episode_status_dump({}))

                gr.Markdown("### Environment State")
                live_env = gr.JSON(label="Live Env State", value={})

        with gr.Accordion("Post Database Reference", open=False):
            post_db = gr.Dataframe(
                headers=["ID", "Area", "Rent", "Diet", "Type", "Commute_Goregaon", "Status"],
                value=_post_rows(default_task_id, {}),
                interactive=False,
            )

        common_outputs = [
            task_definition,
            buyer_chat_filter,
            chatbot,
            post_chat_filter,
            post_agent_chat,
            user_data_table,
            user_data_json,
            user_storage_log,
            scenario_checks,
            visit_scheduler,
            score_panel,
            tool_log,
            episode_status,
            live_env,
            violations,
            post_db,
            model_status,
            final_banner,
            next_btn,
            full_btn,
            tool_name,
            app_state,
        ]

        async def reset_simulation(task_id: str, broker_model_name: str, user_model_name: str, ui_state: dict[str, Any]):
            logger.info(
                "callback:reset:start task_id=%s broker_model=%s user_model=%s",
                task_id,
                broker_model_name,
                user_model_name,
            )
            try:
                observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset, scenario_id=task_id)
                logger.info("callback:reset:after_env_reset task_id=%s", task_id)
                serialized = _serialize_reset(web_manager, observation)
                logger.info(
                    "callback:reset:serialized task_id=%s obs_keys=%s",
                    task_id,
                    sorted(serialized.get("observation", {}).keys()),
                )
                await web_manager._send_state_update()
                logger.info("callback:reset:after_state_update task_id=%s", task_id)
                return _ui_values(task_id, broker_model_name, user_model_name, serialized, ui_state)
            except Exception:
                logger.exception("callback:reset:error task_id=%s", task_id)
                raise

        async def run_manual_message(task_id: str, broker_model_name: str, user_model_name: str, message: str):
            logger.info(
                "callback:manual_message:start task_id=%s message_len=%s",
                task_id,
                len(message or ""),
            )
            try:
                request_payload = {"action_type": "assistant_message", "assistant_message": message}
                logger.info("callback:manual_message:request task_id=%s payload=%s", task_id, request_payload)
                serialized = await web_manager.step_environment(request_payload)
                logger.info(
                    "callback:manual_message:after_step task_id=%s done=%s phase=%s",
                    task_id,
                    serialized.get("observation", {}).get("done"),
                    serialized.get("observation", {}).get("phase"),
                )
                return _ui_values(task_id, broker_model_name, user_model_name, serialized)
            except Exception:
                logger.exception("callback:manual_message:error task_id=%s", task_id)
                raise

        async def run_manual_tool(task_id: str, broker_model_name: str, user_model_name: str, selected_tool: str, raw_arguments: str):
            logger.info(
                "callback:manual_tool:start task_id=%s tool=%s raw_arguments=%s",
                task_id,
                selected_tool,
                raw_arguments,
            )
            try:
                parsed_arguments = json.loads(raw_arguments or "{}")
                request_payload = {
                    "action_type": "tool_call",
                    "tool_name": selected_tool,
                    "tool_arguments": parsed_arguments,
                }
                logger.info("callback:manual_tool:request task_id=%s payload=%s", task_id, request_payload)
                serialized = await web_manager.step_environment(request_payload)
                logger.info(
                    "callback:manual_tool:after_step task_id=%s done=%s phase=%s",
                    task_id,
                    serialized.get("observation", {}).get("done"),
                    serialized.get("observation", {}).get("phase"),
                )
                return _ui_values(task_id, broker_model_name, user_model_name, serialized)
            except Exception:
                logger.exception("callback:manual_tool:error task_id=%s tool=%s", task_id, selected_tool)
                raise

        async def run_next_turn(task_id: str, broker_model_name: str, user_model_name: str, ui_state: dict[str, Any]):
            logger.info("callback:next_turn:start task_id=%s", task_id)
            try:
                current = dict(web_manager.episode_state.current_observation or {})
                logger.info(
                    "callback:next_turn:current task_id=%s has_current=%s current_scenario=%s done=%s phase=%s",
                    task_id,
                    bool(current),
                    current.get("scenario_id"),
                    current.get("done"),
                    current.get("phase"),
                )
                if not current or current.get("scenario_id") != task_id:
                    observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset, scenario_id=task_id)
                    logger.info("callback:next_turn:after_env_reset task_id=%s", task_id)
                    serialized = _serialize_reset(web_manager, observation)
                    await web_manager._send_state_update()
                    current = serialized["observation"]
                request_payload = autopolicy_next_request(task_id, current)
                logger.info("callback:next_turn:autopolicy task_id=%s payload=%s", task_id, request_payload)
                if request_payload is None:
                    return _ui_values(task_id, broker_model_name, user_model_name, {"observation": current}, ui_state)
                serialized = await web_manager.step_environment(request_payload)
                logger.info(
                    "callback:next_turn:after_step task_id=%s done=%s phase=%s",
                    task_id,
                    serialized.get("observation", {}).get("done"),
                    serialized.get("observation", {}).get("phase"),
                )
                return _ui_values(task_id, broker_model_name, user_model_name, serialized, ui_state)
            except Exception:
                logger.exception("callback:next_turn:error task_id=%s", task_id)
                raise

        async def run_full_simulation(task_id: str, broker_model_name: str, user_model_name: str, ui_state: dict[str, Any]):
            logger.info("callback:full_simulation:start task_id=%s", task_id)
            try:
                current = dict(web_manager.episode_state.current_observation or {})
                logger.info(
                    "callback:full_simulation:current task_id=%s has_current=%s current_scenario=%s done=%s phase=%s",
                    task_id,
                    bool(current),
                    current.get("scenario_id"),
                    current.get("done"),
                    current.get("phase"),
                )
                if not current or current.get("scenario_id") != task_id:
                    observation = await web_manager._run_sync_in_thread_pool(web_manager.env.reset, scenario_id=task_id)
                    logger.info("callback:full_simulation:after_env_reset task_id=%s", task_id)
                    serialized = _serialize_reset(web_manager, observation)
                    await web_manager._send_state_update()
                    current = serialized["observation"]
                last_payload = {"observation": current}
                for step_index in range(20):
                    current = last_payload["observation"]
                    logger.info(
                        "callback:full_simulation:loop task_id=%s step_index=%s done=%s phase=%s",
                        task_id,
                        step_index,
                        current.get("done"),
                        current.get("phase"),
                    )
                    if current.get("done"):
                        break
                    request_payload = autopolicy_next_request(task_id, current)
                    logger.info(
                        "callback:full_simulation:autopolicy task_id=%s step_index=%s payload=%s",
                        task_id,
                        step_index,
                        request_payload,
                    )
                    if request_payload is None:
                        break
                    last_payload = await web_manager.step_environment(request_payload)
                logger.info("callback:full_simulation:done task_id=%s", task_id)
                return _ui_values(task_id, broker_model_name, user_model_name, last_payload, ui_state)
            except Exception:
                logger.exception("callback:full_simulation:error task_id=%s", task_id)
                raise

        def set_buyer_chat_filter(ui_state: dict[str, Any], selection: str):
            ui_state = dict(ui_state or {})
            ui_state["buyer_chat_filter"] = selection or "__latest__"
            observation = ui_state.get("observation", {})
            task_id = ui_state.get("task_id", default_task_id)
            ui_state = _normalize_ui_state(task_id, observation, ui_state)
            return _filtered_buyer_chat(observation, ui_state), _buyer_chat_filter_update(observation, ui_state), ui_state

        def set_post_chat_filter(ui_state: dict[str, Any], selection: str):
            ui_state = dict(ui_state or {})
            ui_state["post_chat_filter"] = selection or "__active__"
            observation = ui_state.get("observation", {})
            task_id = ui_state.get("task_id", default_task_id)
            ui_state = _normalize_ui_state(task_id, observation, ui_state)
            return _current_post_chat(task_id, observation, ui_state), _post_chat_filter_update(task_id, observation, ui_state), ui_state

        reset_btn.click(reset_simulation, inputs=[task_dropdown, broker_model, user_model, app_state], outputs=common_outputs)
        next_btn.click(run_next_turn, inputs=[task_dropdown, broker_model, user_model, app_state], outputs=common_outputs)
        full_btn.click(run_full_simulation, inputs=[task_dropdown, broker_model, user_model, app_state], outputs=common_outputs)
        buyer_chat_filter.change(set_buyer_chat_filter, inputs=[app_state, buyer_chat_filter], outputs=[chatbot, buyer_chat_filter, app_state])
        post_chat_filter.change(set_post_chat_filter, inputs=[app_state, post_chat_filter], outputs=[post_agent_chat, post_chat_filter, app_state])
    logger.info("build_flatmate_gradio_app:done")
    return demo
