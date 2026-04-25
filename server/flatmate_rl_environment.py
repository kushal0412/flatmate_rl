"""OpenEnv environment wrapper for Flatmate RL."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState
    from .episode import FlatmateEpisode
except ImportError:
    from models import FlatmateRlAction, FlatmateRlObservation, FlatmateRlState
    from server.episode import FlatmateEpisode


class FlatmateRlEnvironment(Environment):
    """Deterministic RL environment for Flatmate visit scheduling."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = FlatmateRlState(episode_id=str(uuid4()), step_count=0)
        self._episode = FlatmateEpisode()

    def reset(self, scenario_id: str | None = None, seed: int | None = None) -> FlatmateRlObservation:
        del seed
        self._state = FlatmateRlState(episode_id=str(uuid4()), step_count=0)
        observation = self._episode.reset(scenario_id=scenario_id)
        self._sync_state(observation)
        return observation

    def step(self, action: FlatmateRlAction) -> FlatmateRlObservation:  # type: ignore[override]
        self._state.step_count += 1
        observation = self._episode.step(action)
        self._sync_state(observation)
        return observation

    @property
    def state(self) -> FlatmateRlState:
        return self._state

    def _sync_state(self, observation: FlatmateRlObservation) -> None:
        self._state.scenario_id = observation.scenario_id
        self._state.phase = observation.phase
        self._state.status = observation.status
        self._state.gathered_fields = list(observation.gathered_fields)
        self._state.selected_posts = list(observation.selected_posts)
        self._state.booked_visits = list(observation.booked_visits)
        self._state.buyer_profile_stored = observation.buyer_profile_stored
        self._state.seller_profile_stored = observation.seller_profile_stored
        self._state.tool_trace = list(observation.tool_trace)
        self._state.total_reward = observation.total_reward
        self._state.done = observation.done
