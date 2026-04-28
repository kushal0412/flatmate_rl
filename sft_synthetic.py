"""Synthetic SFT data generation and evaluation for Flatmate RL.

The generator rolls out the built-in heuristic policy and writes one supervised
example per broker decision. Each row is compatible with chat-style SFT loaders:

{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

try:
    from flatmate_rl import FlatmateRlAction
    from flatmate_rl.inference import (
        SYSTEM_PROMPT,
        build_user_prompt,
        format_action,
        malformed_action_observation,
        parse_action,
        sanitize_observation_for_policy,
    )
    from flatmate_rl.server.flatmate_rl_environment import FlatmateRlEnvironment
    from flatmate_rl.server.heuristic_policy import autopolicy_next_request, expected_policy_action
    from flatmate_rl.server.scenarios import SCENARIOS
except ImportError:
    from __init__ import FlatmateRlAction
    from inference import (
        SYSTEM_PROMPT,
        build_user_prompt,
        format_action,
        malformed_action_observation,
        parse_action,
        sanitize_observation_for_policy,
    )
    from server.flatmate_rl_environment import FlatmateRlEnvironment
    from server.heuristic_policy import autopolicy_next_request, expected_policy_action
    from server.scenarios import SCENARIOS


DEFAULT_OUTPUT_DIR = Path("data/sft")


def parse_seed_spec(value: str) -> list[int]:
    """Parse seed specs like '0:100', '0:100:5', or '1,2,3'."""
    value = value.strip()
    if not value:
        return []
    if "," in value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if ":" in value:
        parts = [int(item) for item in value.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Invalid seed range: {value!r}")
        return list(range(start, stop, step))
    return [int(value)]


def action_to_json(action: FlatmateRlAction) -> str:
    return format_action(action)


def make_sft_example(
    *,
    scenario_id: str,
    seed: int,
    step: int,
    observation: Any,
    action: FlatmateRlAction,
) -> dict[str, Any]:
    prompt = build_user_prompt(step=step, observation=observation)
    completion = action_to_json(action)
    return {
        "id": f"{scenario_id}:seed={seed}:step={step}",
        "scenario_id": scenario_id,
        "seed": seed,
        "step": step,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "prompt": prompt,
        "completion": completion,
        "action": json.loads(completion),
    }


def rollout_sft_examples(
    *,
    scenario_id: str,
    seed: int,
    max_steps: int,
    strict_eval: bool,
) -> Iterator[dict[str, Any]]:
    env = FlatmateRlEnvironment()
    observation = env.reset(scenario_id=scenario_id, seed=seed)

    for step in range(1, max_steps + 1):
        if observation.done:
            break
        payload = autopolicy_next_request(scenario_id, observation.model_dump())
        if payload is None:
            break
        action = FlatmateRlAction.model_validate(payload)
        policy_observation = sanitize_observation_for_policy(observation, strict_eval=strict_eval)
        yield make_sft_example(
            scenario_id=scenario_id,
            seed=seed,
            step=step,
            observation=policy_observation,
            action=action,
        )
        observation = env.step(action)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def generate_dataset(
    *,
    output_dir: Path,
    scenario_ids: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    max_steps: int,
    strict_eval: bool,
) -> dict[str, Any]:
    def rows_for(seeds: list[int]) -> Iterator[dict[str, Any]]:
        for scenario_id in scenario_ids:
            for seed in seeds:
                yield from rollout_sft_examples(
                    scenario_id=scenario_id,
                    seed=seed,
                    max_steps=max_steps,
                    strict_eval=strict_eval,
                )

    train_count = write_jsonl(output_dir / "train.jsonl", rows_for(train_seeds))
    eval_count = write_jsonl(output_dir / "eval.jsonl", rows_for(eval_seeds))
    manifest = {
        "scenario_ids": scenario_ids,
        "train_seeds": train_seeds,
        "eval_seeds": eval_seeds,
        "max_steps": max_steps,
        "strict_eval": strict_eval,
        "files": {
            "train": str(output_dir / "train.jsonl"),
            "eval": str(output_dir / "eval.jsonl"),
        },
        "counts": {
            "train_examples": train_count,
            "eval_examples": eval_count,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def episode_success(scenario_id: str, observation: Any) -> bool:
    required_bookings = int(SCENARIOS[scenario_id]["ground_truth"]["required_bookings"])
    return bool(observation.done) and len(observation.booked_visits) >= required_bookings and not observation.violations


def evaluate_policy(
    *,
    policy_name: str,
    scenario_ids: list[str],
    seeds: list[int],
    max_steps: int,
    action_fn,
    strict_eval: bool,
) -> dict[str, Any]:
    episodes = []
    for scenario_id in scenario_ids:
        for seed in seeds:
            env = FlatmateRlEnvironment()
            observation = env.reset(scenario_id=scenario_id, seed=seed)
            parse_errors = 0
            steps_taken = 0

            for step in range(1, max_steps + 1):
                if observation.done:
                    break
                policy_observation = sanitize_observation_for_policy(observation, strict_eval=strict_eval)
                action = action_fn(scenario_id, step, observation, policy_observation)
                if action is None:
                    parse_errors += 1
                    observation = malformed_action_observation(observation, "model returned no valid action")
                else:
                    observation = env.step(action)
                steps_taken = step

            episodes.append(
                {
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "success": episode_success(scenario_id, observation),
                    "done": bool(observation.done),
                    "steps": steps_taken,
                    "total_reward": float(observation.total_reward),
                    "booked_visits": len(observation.booked_visits),
                    "violations": list(observation.violations),
                    "parse_errors": parse_errors,
                    "status": observation.status,
                }
            )

    rewards = [item["total_reward"] for item in episodes]
    success_count = sum(1 for item in episodes if item["success"])
    summary = {
        "policy": policy_name,
        "episodes": len(episodes),
        "successes": success_count,
        "success_rate": success_count / len(episodes) if episodes else 0.0,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "mean_steps": statistics.fmean(item["steps"] for item in episodes) if episodes else 0.0,
        "parse_errors": sum(item["parse_errors"] for item in episodes),
        "strict_eval": strict_eval,
        "scenario_ids": scenario_ids,
        "seeds": seeds,
    }
    return {"summary": summary, "episodes": episodes}


def heuristic_action_fn(scenario_id: str, _step: int, observation: Any, _policy_observation: Any) -> FlatmateRlAction | None:
    payload = expected_policy_action(scenario_id, observation.model_dump())
    if payload is None:
        return None
    return FlatmateRlAction.model_validate(payload)


def load_hf_policy(model_path: str, max_new_tokens: int, temperature: float):
    """Load a local Hugging Face causal LM and return an environment action fn."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Model evaluation requires optional dependencies. Install transformers and torch first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    model.eval()

    def render_messages(messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages) + "\n\nASSISTANT:\n"

    def action_fn(_scenario_id: str, step: int, _observation: Any, policy_observation: Any) -> FlatmateRlAction | None:
        prompt = build_user_prompt(step=step, observation=policy_observation)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        rendered = render_messages(messages)
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output[0][inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        parsed = parse_action(text, strict=True)
        return parsed.action

    return action_fn


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario id to include. May be supplied multiple times. Defaults to all scenarios.",
    )
    parser.add_argument("--seeds", default="1000:1020", help="Held-out seeds, e.g. '1000:1020' or '1,2,3'.")
    parser.add_argument("--max-steps", type=int, default=28)
    parser.add_argument(
        "--strict-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide scenario labels, reward, tool trace, and direct field trackers from policy input.",
    )
    parser.add_argument("--details-path", type=Path, help="Optional JSON path for per-episode results.")


def cmd_generate(args: argparse.Namespace) -> None:
    manifest = generate_dataset(
        output_dir=args.output_dir,
        scenario_ids=args.scenario_ids or sorted(SCENARIOS.keys()),
        train_seeds=parse_seed_spec(args.train_seeds),
        eval_seeds=parse_seed_spec(args.eval_seeds),
        max_steps=args.max_steps,
        strict_eval=args.strict_eval,
    )
    print_json(manifest)


def cmd_eval_heuristic(args: argparse.Namespace) -> None:
    result = evaluate_policy(
        policy_name="heuristic",
        scenario_ids=args.scenario_ids or sorted(SCENARIOS.keys()),
        seeds=parse_seed_spec(args.seeds),
        max_steps=args.max_steps,
        action_fn=heuristic_action_fn,
        strict_eval=args.strict_eval,
    )
    if args.details_path:
        args.details_path.parent.mkdir(parents=True, exist_ok=True)
        args.details_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_json(result["summary"])


def cmd_eval_model(args: argparse.Namespace) -> None:
    action_fn = load_hf_policy(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    result = evaluate_policy(
        policy_name=args.model_path,
        scenario_ids=args.scenario_ids or sorted(SCENARIOS.keys()),
        seeds=parse_seed_spec(args.seeds),
        max_steps=args.max_steps,
        action_fn=action_fn,
        strict_eval=args.strict_eval,
    )
    if args.details_path:
        args.details_path.parent.mkdir(parents=True, exist_ok=True)
        args.details_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_json(result["summary"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data and evaluate Flatmate RL policies.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate train/eval JSONL from heuristic rollouts.")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    generate.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        choices=sorted(SCENARIOS.keys()),
        help="Scenario id to include. May be supplied multiple times. Defaults to all scenarios.",
    )
    generate.add_argument("--train-seeds", default="0:100", help="Training seeds, e.g. '0:100' or '1,2,3'.")
    generate.add_argument("--eval-seeds", default="1000:1020", help="Held-out eval seeds.")
    generate.add_argument("--max-steps", type=int, default=28)
    generate.add_argument(
        "--strict-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate policy prompts with strict-eval sanitization.",
    )
    generate.set_defaults(func=cmd_generate)

    eval_heuristic = subparsers.add_parser("eval-heuristic", help="Evaluate the built-in heuristic on held-out seeds.")
    add_common_eval_args(eval_heuristic)
    eval_heuristic.set_defaults(func=cmd_eval_heuristic)

    eval_model = subparsers.add_parser("eval-model", help="Evaluate a local Hugging Face SFT checkpoint.")
    add_common_eval_args(eval_model)
    eval_model.add_argument("--model-path", required=True)
    eval_model.add_argument("--max-new-tokens", type=int, default=256)
    eval_model.add_argument("--temperature", type=float, default=0.0)
    eval_model.set_defaults(func=cmd_eval_model)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
