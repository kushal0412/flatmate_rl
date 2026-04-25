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

An OpenEnv environment for training and evaluating agents on broker-style flatmate visit scheduling.

This environment converts the `broker_app` visit-scheduling scenarios into a deterministic RL task where an agent must:

- gather missing buyer or seller details through conversation
- call the right environment tools in the right order
- respect scheduling and confirmation guardrails
- book valid visits only after the required checks and confirmations succeed

It also includes a custom Gradio UI mounted at `/web`.

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

## Included Scenarios

The environment mirrors the main `broker_app` scenarios:

- `task_visit_single`
  One valid visit must be booked.
- `task_visit_single_hidden_flex`
  The buyer initially exposes only one slot, but hidden flexibility can be unlocked if the agent offers concrete alternatives.
- `task_visit_multi`
  At least two valid non-overlapping visits must be booked.
- `task_visit_single_seller_followup`
  The first buyer flow cannot book a visit, then a seller follow-up creates a new listing that can be matched and scheduled.

Scenario declarations live in:

- [server/scenario_factory.py](/Users/kushaljaisinghani/Documents/sample_envs/flatmate_rl/server/scenario_factory.py)
- [server/scenarios.py](/Users/kushaljaisinghani/Documents/sample_envs/flatmate_rl/server/scenarios.py)

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

## Training an RL Agent

The action space is mixed discrete-plus-structured:

- choose whether to send a message or call a tool
- if sending a message, generate natural language
- if calling a tool, choose the tool and valid JSON arguments

In practice, the easiest setup is usually:

1. use an LLM policy or seq2seq policy that emits a structured action object
2. compute reward from `done`, `violations`, `booked_visits`, and `last_tool_result`
3. train with policy gradient, GRPO, PPO, or offline imitation plus RL fine-tuning

### Example Training Loop

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

### Recommended Training Strategy

For serious training, a better progression is:

1. Start with supervised trajectories for correct broker flows.
2. Fine-tune with RL on sparse success reward plus shaping reward.
3. Penalize:
   `violations`, failed tool calls, missing storage steps, invalid booking attempts.
4. Reward:
   correct information gathering, correct tool order, valid slot coordination, successful booking completion.

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

## Docker

This repo includes a Dockerfile similar to `sudoku_rl`.

It enables the web interface by default:

```dockerfile
ENV ENABLE_WEB_INTERFACE=true
```

Build and run:

```bash
cd flatmate_rl
docker build -t flatmate_rl .
docker run -p 8000:8000 flatmate_rl
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
│   └── scenarios.py
└── tests/
    └── test_flatmate_rl.py
```

## Notes

- The environment is deterministic and designed for RL experimentation, not as a drop-in replacement for the original multi-LLM broker simulator.
- The current Python 3.13 Anaconda runtime in this workspace can crash when importing parts of `openenv`; using the local Python 3.12 virtualenv is the safer path for testing here.
# flatmate_rl
