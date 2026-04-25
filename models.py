"""Data models for the Flatmate RL environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator


ActionType = Literal["assistant_message", "tool_call"]


class FlatmateRlAction(Action):
    """One assistant turn or one backend tool call."""

    action_type: ActionType = Field(..., description="assistant_message or tool_call.")
    assistant_message: str = Field(default="", description="Broker message shown to the active user.")
    tool_name: str = Field(default="", description="Tool name when action_type='tool_call'.")
    tool_arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")

    @model_validator(mode="after")
    def validate_shape(self) -> "FlatmateRlAction":
        if self.action_type == "assistant_message":
            if not self.assistant_message.strip():
                raise ValueError("assistant_message is required when action_type='assistant_message'.")
            if self.tool_name or self.tool_arguments:
                raise ValueError("tool_name and tool_arguments must be empty for assistant_message actions.")
        if self.action_type == "tool_call" and not self.tool_name.strip():
            raise ValueError("tool_name is required when action_type='tool_call'.")
        return self


class FlatmateRlObservation(Observation):
    """Visible state returned after reset and each step."""

    status: str = Field(default="ready", description="ready, user_response, tool_result, completed, or failed.")
    scenario_id: str = Field(default="", description="Active scenario id.")
    scenario_label: str = Field(default="", description="Human-readable scenario label.")
    difficulty: str = Field(default="", description="Scenario difficulty.")
    phase: str = Field(default="buyer", description="buyer or seller.")
    current_user_request: str = Field(default="", description="Latest visible user-side request or reply.")
    last_user_message: str = Field(default="", description="Most recent user/seller message.")
    conversation_history: list[dict[str, Any]] = Field(default_factory=list, description="Visible transcript.")
    buyer_conversation_history: list[dict[str, Any]] = Field(default_factory=list, description="Buyer/broker transcript.")
    seller_conversation_history: list[dict[str, Any]] = Field(default_factory=list, description="Seller/broker transcript.")
    last_tool_result: dict[str, Any] = Field(default_factory=dict, description="Most recent tool result.")
    tool_results: list[dict[str, Any]] = Field(default_factory=list, description="All tool results so far.")
    tool_trace: list[dict[str, Any]] = Field(default_factory=list, description="Tool call trace with args and outcomes.")
    available_tools: list[str] = Field(default_factory=list, description="Tools available in the current phase.")
    gathered_fields: list[str] = Field(default_factory=list, description="Fields gathered so far for the active phase.")
    remaining_required_fields: list[str] = Field(default_factory=list, description="Required fields still missing.")
    selected_posts: list[str] = Field(default_factory=list, description="Posts selected by the buyer.")
    booked_visits: list[dict[str, str]] = Field(default_factory=list, description="Booked visits.")
    profile_stored: bool = Field(default=False, description="Whether the current phase profile is stored.")
    buyer_profile_stored: bool = Field(default=False, description="Whether buyer details are stored.")
    seller_profile_stored: bool = Field(default=False, description="Whether seller details are stored.")
    violations: list[str] = Field(default_factory=list, description="Ordering or policy violations.")
    step_reward: float = Field(default=0.0, description="Reward from the most recent step.")
    total_reward: float = Field(default=0.0, description="Cumulative reward for the episode.")
    message: str = Field(default="", description="Human-readable step summary.")


class FlatmateRlState(State):
    """Server-side state snapshot for the active episode."""

    scenario_id: str = Field(default="", description="Active scenario id.")
    phase: str = Field(default="buyer", description="Current episode phase.")
    status: str = Field(default="ready", description="Current episode status.")
    gathered_fields: list[str] = Field(default_factory=list, description="Gathered fields in the active phase.")
    selected_posts: list[str] = Field(default_factory=list, description="Selected post ids.")
    booked_visits: list[dict[str, str]] = Field(default_factory=list, description="Booked visits.")
    buyer_profile_stored: bool = Field(default=False, description="Whether buyer details are stored.")
    seller_profile_stored: bool = Field(default=False, description="Whether seller details are stored.")
    tool_trace: list[dict[str, Any]] = Field(default_factory=list, description="Structured tool trace.")
    total_reward: float = Field(default=0.0, description="Cumulative reward for the episode.")
    done: bool = Field(default=False, description="Whether the episode is complete.")
