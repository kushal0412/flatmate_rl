---
title: Flatmate RL
emoji: 🏠
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - flatmate
  - scheduling
  - reinforcement-learning
---

# Flatmate RL

Flatmate RL models flatmate-share search as a multi-step reinforcement-learning environment for broker agents. The task is not only retrieval. An agent must gather missing buyer or seller details, inspect listing metadata, check availability, coordinate both sides, and only schedule visits when the required constraints and confirmations are satisfied.

The environment focuses on the operational loop behind a real flatmate search:

1. Find posts that might match.
2. Filter out bad or unavailable listings.
3. Ask the buyer for missing preferences.
4. Contact the listing owner or flatmate.
5. Check whether the flat is still available.
6. Match both sides on budget, lifestyle, location, commute, and visit time.
7. Schedule visits without double-booking or assuming consent.
8. Keep going when preferences change after real visits.

Flatmate RL turns that workflow into a deterministic OpenEnv environment. At each step, the policy emits either a natural-language `assistant_message` or a structured `tool_call`. The environment updates conversation state, validates tool order and arguments, tracks buyer/seller confirmations, applies rewards and penalties, and returns a structured observation for the next policy decision.

Read the full project writeup: [Flatmate RL: Training Broker Agents for Real Flatmate Search](flatmate_rl.md).

This environment converts the `broker_app` visit-scheduling scenarios into a deterministic RL task where an agent must:

- gather missing buyer or seller details through conversation
- call the right environment tools in the right order
- respect scheduling and confirmation guardrails
- book valid visits only after the required checks and confirmations succeed

It also includes a FastAPI/OpenEnv server and a custom Gradio UI mounted at `/web`.

## Environment Type

`flatmate_rl` is built for RL over operational agent workflows:

- **Runtime:** OpenEnv environment served through FastAPI.
- **Interaction style:** mixed natural-language and structured tool-calling.
- **Task domain:** Mumbai flatmate-share brokerage.
- **Episode shape:** multi-step buyer/seller conversations with hidden state, tool prerequisites, confirmations, and terminal success conditions.
- **Action schema:** `assistant_message` or `tool_call`.
- **Observation schema:** visible conversation state, available tools, gathered fields, selected posts, booked visits, violations, and rewards.
- **Deployment shape:** Docker Space on Hugging Face, with local Docker support.

This makes the environment useful for training policies that need to learn *when* to ask for information, *which* tool to call, *what arguments* to pass, and *when not to book*.

## What This Environment Does

`flatmate_rl` simulates a housing broker workflow around flatmate-share listings in Mumbai. The agent interacts with the environment one step at a time using either:

- an `assistant_message` action to talk to the active user
- a `tool_call` action to use broker tools such as profile storage, listing search, slot checks, poster contact, and booking

The environment tracks:

- which required fields have been gathered
- which fields are still missing
- which posts were selected
- whether buyer or seller data was stored
- whether tool order rules were violated
- whether visits were successfully booked

The simulator is deterministic by design so it is easier to use for RL training, regression testing, and reward iteration.

## Scenario Types

The environment contains scenario families that model real flat-search failure modes:

- `task_visit_single`
  One valid visit must be booked.
- `task_visit_single_hidden_flex`
  The buyer initially exposes only one slot, but hidden flexibility can be unlocked if the agent offers concrete alternatives.
- `task_visit_multi`
  At least two valid non-overlapping visits must be booked.
- `task_visit_single_seller_followup`
  The first buyer flow cannot book a visit, then a seller follow-up creates a new listing that can be matched and scheduled.
- `task_negotiation_hidden_budget`
  A strong listing is above the buyer's stated budget, but both buyer and seller have hidden negotiable price bounds.
- `task_slot_cancellation_waitlist`
  A desired listing is fully booked until a cancellation opens a previously unavailable slot.
- `task_multi_visit_preference_evolution`
  The buyer discovers new preferences after visits, so the agent must debrief, update the profile, filter new arrivals, and keep searching.
- `task_visit_conflict_check`
  A listing has some slots already reserved by other buyers, so the agent must read the conflict data and propose only an open slot.

Scenario declarations live in:

- [server/scenario_factory.py](server/scenario_factory.py)
- [server/scenarios.py](server/scenarios.py)

## Synthetic Data And No-Leakage Design

The scenarios are synthetic. The environment does not depend on scraped listings, real buyer names, real seller names, phone numbers, emails, or private housing records.

Scenario data is created through small factory helpers in [server/scenario_factory.py](server/scenario_factory.py):

- `build_buyer_profile`
- `build_seller_profile`
- `build_post`
- `build_ground_truth`
- `build_visit_scenario`

Seeded variation lives in [server/scenario_variants.py](server/scenario_variants.py). It uses Python's deterministic `random.Random(f"{task_id}:{seed}")` pattern to create repeatable train/test variants while preserving the scenario's solution structure. Today it varies safe surface values such as:

- buyer occupation
- rent and budget amounts
- generated buyer/seller opening messages

This is the random-detail framework for the environment: all generated details should come from a seed, remain synthetic, and be applied only to fields that do not change the answer key. The current scenarios do not require named buyer or seller identities. If names are added later, generate them in this variant layer, keep them synthetic, and never place real contact details in scenario files or observations.

The important constraint is that seeded variants preserve:

- task id
- post ids
- required tools
- feasible slots
- required bookings
- phase transitions
- canonical success path

That means train and test episodes can have different visible details without leaking a different solution rule into the prompt. If future scenarios need person names or richer contact details, add them through the same seeded variant layer using synthetic-only values. Do not add real names, phone numbers, emails, addresses, or scraped listing text.

For stricter evaluation, set:

```bash
STRICT_EVAL_MODE=true
```

Strict eval mode hides direct scenario labels, difficulty, gathered/remaining fields, violations, tool traces, and rewards from the observation while still allowing sanitized tool results. Use this when you want to reduce prompt leakage during model evaluation.

## Action Format

`FlatmateRlAction` supports two action types:

- `assistant_message`
- `tool_call`

Example assistant action:

```python
from flatmate_rl import FlatmateRlAction

FlatmateRlAction(
    action_type="assistant_message",
    assistant_message="Please share your dietary preference and visit availability.",
)
```

Example tool action:

```python
from flatmate_rl import FlatmateRlAction

FlatmateRlAction(
    action_type="tool_call",
    tool_name="check_calendar_slots",
    tool_arguments={"post_ids": ["post_023", "post_031"]},
)
```

## Observation Format

Each reset or step returns a `FlatmateRlObservation` with fields such as:

- `status`
- `scenario_id`
- `phase`
- `conversation_history`
- `last_tool_result`
- `available_tools`
- `gathered_fields`
- `remaining_required_fields`
- `selected_posts`
- `booked_visits`
- `violations`
- `message`

This gives an RL policy enough structured state to learn the broker workflow while still preserving the conversation transcript.

## Tooling Model

The broker-side tool space includes these buyer-phase tools:

- `store_user_details`
- `search_posts`
- `close_buyer_conversation`
- `match_location_preference`
- `get_commute_time`
- `check_calendar_slots`
- `shortlist`
- `contact_poster`
- `book_viewing`

The seller-follow-up phase adds:

- `store_seller_details`
- `check_table_slot_matches`
- `confirm_seller_match`
- `offer_matched_listing_to_buyer`
- `schedule_table_visit`

The environment enforces sequencing constraints. For example:

- searching before `store_user_details` fails
- seller follow-up tools cannot be used before `store_seller_details`
- bookings fail if the required confirmations are missing
- unknown tools or missing required tool arguments terminate the episode with a hallucination penalty
- repeated successful tool calls and non-canonical ordering are penalized

## Quick Start

```python
from flatmate_rl import FlatmateRlAction
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment

env = FlatmateRlEnvironment()

obs = env.reset(scenario_id="task_visit_single")
print(obs.last_user_message)
print(obs.remaining_required_fields)

obs = env.step(
    FlatmateRlAction(
        action_type="assistant_message",
        assistant_message="Please share your dietary preference and visit availability.",
    )
)
print(obs.last_user_message)

obs = env.step(
    FlatmateRlAction(
        action_type="tool_call",
        tool_name="store_user_details",
        tool_arguments={},
    )
)
print(obs.last_tool_result)
```

## Training An RL Agent

The action space is mixed discrete-plus-structured:

- choose whether to send a message or call a tool
- if sending a message, generate natural language
- if calling a tool, choose the tool and valid JSON arguments

In practice, the easiest setup is usually:

1. use an LLM policy or seq2seq policy that emits a structured action object
2. serialize the observation into a prompt
3. parse the model response into `FlatmateRlAction`
4. step the environment
5. train from `step_reward`, `total_reward`, `done`, `violations`, `booked_visits`, and `last_tool_result`

Good training progressions are:

- imitation/SFT on correct broker trajectories
- GRPO/PPO/REINFORCE with endpoint reward
- terminal-reward training where the candidate action is replayed into a full rollout
- held-out seeded evaluation with strict eval mode enabled

### Local In-Process Training Loop

The example below shows a minimal policy-gradient style skeleton. It is intentionally simple and is meant to show how to interact with the environment, not to be a production trainer.

```python
from __future__ import annotations

import random
from dataclasses import dataclass

from flatmate_rl import FlatmateRlAction
from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment


SCENARIOS = [
    "task_visit_single",
    "task_visit_single_hidden_flex",
    "task_visit_multi",
    "task_visit_single_seller_followup",
]


@dataclass
class Transition:
    observation_text: str
    action: FlatmateRlAction
    reward: float
    done: bool


def flatten_observation(obs) -> str:
    return (
        f"scenario={obs.scenario_id}\n"
        f"phase={obs.phase}\n"
        f"status={obs.status}\n"
        f"remaining={obs.remaining_required_fields}\n"
        f"available_tools={obs.available_tools}\n"
        f"selected_posts={obs.selected_posts}\n"
        f"booked_visits={obs.booked_visits}\n"
        f"violations={obs.violations}\n"
        f"message={obs.message}\n"
        f"last_user_message={obs.last_user_message}\n"
    )


class DummyPolicy:
    def act(self, obs) -> FlatmateRlAction:
        remaining = set(obs.remaining_required_fields)

        if "diet" in remaining or "visit_availability" in remaining:
            return FlatmateRlAction(
                action_type="assistant_message",
                assistant_message="Please share your dietary preference and visit availability.",
            )

        if not obs.buyer_profile_stored and obs.phase == "buyer":
            return FlatmateRlAction(
                action_type="tool_call",
                tool_name="store_user_details",
                tool_arguments={},
            )

        if obs.phase == "buyer" and "search_posts" in obs.available_tools:
            return FlatmateRlAction(
                action_type="tool_call",
                tool_name="search_posts",
                tool_arguments={},
            )

        available_tools = obs.available_tools
        fallback_tool = available_tools[0] if available_tools else "store_user_details"
        return FlatmateRlAction(
            action_type="tool_call",
            tool_name=fallback_tool,
            tool_arguments={},
        )

    def update(self, trajectory: list[Transition]) -> None:
        # Replace with PPO / GRPO / REINFORCE / DPO / imitation loss, etc.
        pass


def compute_reward(obs) -> float:
    reward = 0.0
    reward += 5.0 * len(obs.booked_visits)
    reward -= 1.0 * len(obs.violations)

    last_tool = obs.last_tool_result or {}
    if last_tool.get("success") is True:
        reward += 0.2
    if last_tool.get("success") is False:
        reward -= 0.5

    if obs.done:
        reward += 10.0
    return reward


def train(num_episodes: int = 1000, max_steps: int = 20) -> None:
    env = FlatmateRlEnvironment()
    policy = DummyPolicy()

    for episode_idx in range(num_episodes):
        scenario_id = random.choice(SCENARIOS)
        obs = env.reset(scenario_id=scenario_id)
        trajectory: list[Transition] = []

        for _ in range(max_steps):
            action = policy.act(obs)
            next_obs = env.step(action)
            reward = compute_reward(next_obs)

            trajectory.append(
                Transition(
                    observation_text=flatten_observation(obs),
                    action=action,
                    reward=reward,
                    done=next_obs.done,
                )
            )

            obs = next_obs
            if obs.done:
                break

        policy.update(trajectory)

        if episode_idx % 50 == 0:
            print(
                f"episode={episode_idx} "
                f"scenario={scenario_id} "
                f"done={obs.done} "
                f"bookings={len(obs.booked_visits)} "
                f"violations={len(obs.violations)}"
            )


if __name__ == "__main__":
    train()
```

### Endpoint Training Loop

When training against Docker or the Hugging Face Space, keep the episode on the websocket endpoint. A websocket session holds one environment instance across `reset` and `step`.

```python
from __future__ import annotations

import asyncio
import json
from urllib.parse import urlparse

import websockets


def ws_url_from_http(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws"


class FlatmateWsEnv:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.ws_url = ws_url_from_http(base_url)
        self.timeout_s = timeout_s
        self.ws = None

    async def __aenter__(self):
        self.ws = await websockets.connect(
            self.ws_url,
            open_timeout=self.timeout_s,
            ping_timeout=self.timeout_s,
        )
        return self

    async def __aexit__(self, *_exc):
        if self.ws is not None:
            await self.ws.send(json.dumps({"type": "close"}))
            await self.ws.close()

    async def _send(self, payload: dict) -> dict:
        assert self.ws is not None
        await self.ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(self.ws.recv(), timeout=self.timeout_s)
        data = json.loads(raw)
        obs = data.get("observation", data)
        obs["reward"] = data.get("reward", obs.get("step_reward", 0.0))
        obs["done"] = data.get("done", obs.get("done", False))
        return obs

    async def reset(self, scenario_id: str, seed: int | None = None) -> dict:
        data = {"scenario_id": scenario_id}
        if seed is not None:
            data["seed"] = seed
        return await self._send({"type": "reset", "data": data})

    async def step(self, action: dict) -> dict:
        return await self._send({"type": "step", "data": action})


async def rollout(base_url: str) -> None:
    async with FlatmateWsEnv(base_url) as env:
        obs = await env.reset("task_visit_single", seed=7)

        action = {
            "action_type": "assistant_message",
            "assistant_message": "Please share your dietary preference and visit availability.",
        }
        obs = await env.step(action)
        print(obs["status"], obs["reward"], obs["done"])


asyncio.run(rollout("http://127.0.0.1:8000"))
```

Use the same client for the hosted Space by changing the base URL to the Space app URL.

## Running With Docker

Build and run locally:

```bash
cd flatmate_rl
docker build -t flatmate_rl .
docker run --rm -p 8000:8000 flatmate_rl
```

Open the UI:

```text
http://127.0.0.1:8000/web
```

Use the websocket endpoint for training:

```text
ws://127.0.0.1:8000/ws
```

The Dockerfile uses the OpenEnv base image, installs dependencies with `uv`, sets `ENABLE_WEB_INTERFACE=true`, exposes the app on port `8000`, and starts:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Hugging Face Space Deployment

The deployed Space is:

```text
https://huggingface.co/spaces/kushalExplores/flatmate_rl
```

The Space is configured as a Docker Space in the README metadata:

```yaml
sdk: docker
app_port: 8000
base_path: /web
```

The OpenEnv deployment config is in [openenv.yaml](openenv.yaml):

```yaml
spec_version: 1
name: flatmate_rl
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

How it is deployed:

1. Hugging Face builds [Dockerfile](Dockerfile).
2. The image copies this repo into `/app/env`.
3. Dependencies are installed with `uv sync`.
4. The runtime starts `server.app:app` with Uvicorn on port `8000`.
5. `server.app` creates the OpenEnv FastAPI app with `FlatmateRlEnvironment`, `FlatmateRlAction`, and `FlatmateRlObservation`.
6. The custom Gradio interface is mounted at `/web`.

The public Space page is the stable share URL. For programmatic training, use the app websocket endpoint exposed by the running Space:

```text
wss://kushalexplores-flatmate-rl.hf.space/ws
```

For the browser UI, open:

```text
https://kushalexplores-flatmate-rl.hf.space/web
```

If Hugging Face changes the direct app subdomain, open the Space page and use the app link shown there.

The server is configured with `max_concurrent_envs=4`, so keep GRPO/PPO reward workers conservative at first. Increase rollout concurrency only after the endpoint is stable.

## Training Strategy

For serious training:

1. Collect balanced rollouts across all scenario ids and seeds.
2. Keep train/test seeds separate.
3. Start with SFT or imitation on valid JSON actions.
4. Add RL reward from the environment endpoint.
5. Penalize malformed JSON, unknown tools, missing arguments, invalid booking attempts, repeated successful tools, and non-canonical ordering.
6. Reward correct information gathering, correct tool order, valid slot coordination, negotiated deals, waitlist handling, and successful terminal completion.
7. Evaluate with held-out seeds and `STRICT_EVAL_MODE=true`.

## Web UI

The environment exposes a custom Gradio UI at `/web`.

It includes:

- scenario selector
- transcript viewer
- assistant-message controls
- tool-call runner with JSON arguments
- live gathered/remaining field panels
- selected posts, booked visits, violations
- request/response payload panes

Run locally:

```bash
cd flatmate_rl
uv run --project . server
```

Then open:

```text
http://127.0.0.1:8000/web
```

## Local Development

```bash
cd flatmate_rl
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

Or with `uv`:

```bash
cd flatmate_rl
uv sync
uv run --project . pytest
uv run --project . server
```

## Tests

The test suite checks:

- scenario parity against `broker_app`
- ordering guardrails
- single-visit booking flow
- hidden-flex slot behavior
- multi-booking flow
- seller-follow-up scheduling

Run:

```bash
flatmate_rl/.venv/bin/python -m pytest flatmate_rl/tests/test_flatmate_rl.py
```

## Repository Layout

```text
flatmate_rl/
├── Dockerfile
├── README.md
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── server/
│   ├── app.py
│   ├── episode.py
│   ├── flatmate_rl_environment.py
│   ├── gradio_ui.py
│   ├── scenario_factory.py
│   ├── scenario_variants.py
│   └── scenarios.py
└── tests/
    └── test_flatmate_rl.py
```

## Notes

- The environment is deterministic and designed for RL experimentation, not as a drop-in replacement for the original multi-LLM broker simulator.
- The current Python 3.13 Anaconda runtime in this workspace can crash when importing parts of `openenv`; using the local Python 3.12 virtualenv is the safer path for testing here.
