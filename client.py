"""Client for the Flatmate RL environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState


class FlatmateRlEnv(EnvClient[FlatmateRlAction, FlatmateRlObservation, FlatmateRlState]):
    """Thin HTTP/WebSocket client for the Flatmate RL environment."""

    def _step_payload(self, action: FlatmateRlAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[FlatmateRlObservation]:
        observation = FlatmateRlObservation.model_validate(
            {
                **payload.get("observation", {}),
                "reward": payload.get("reward"),
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> FlatmateRlState:
        return FlatmateRlState.model_validate(payload)
