"""Seeded value variants for Flatmate RL scenarios.

Variants intentionally preserve episode flow: same task id, post ids, required
tools, required bookings, feasible slots, and phase transitions. Only safe
surface values are shifted so train/test episodes can differ without changing
the canonical solution structure.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any


OCCUPATIONS = [
    "software engineer at a startup",
    "backend engineer at a fintech company",
    "data engineer at a healthtech company",
    "platform engineer at a SaaS company",
]

RENT_DELTAS = [-1200, -800, -500, 0, 500, 800, 1200]


def _format_rs(amount: int) -> str:
    return f"{amount:,}"


def _buyer_message(scenario: dict[str, Any]) -> str:
    buyer = scenario["buyer_profile"]
    areas = " or ".join(buyer["areas"])
    budget = _format_rs(int(buyer["budget_max"]))
    occupation = buyer["occupation"]
    availability = " or ".join(buyer["visit_availability"])
    task_id = scenario["task_id"]

    if task_id == "task_visit_multi":
        return (
            "Hi, I want to line up visits for at least two good flatmate-share options before deciding. "
            f"My budget is Rs. {budget}, I'm focused on {areas}, and I work in Goregaon East as a {occupation}."
        )
    if task_id == "task_visit_single_seller_followup":
        return (
            f"Hi, I'm looking for a flatmate-share in {areas}. My budget is Rs. {budget}, "
            f"I work in Goregaon East as a {occupation}, and {availability} are the only times I can visit."
        )
    if "visit_availability" in buyer.get("initial_disclosure_fields", []):
        return (
            f"Hi, I'm looking for a flatmate-share around {areas}. My budget is Rs. {budget}, "
            f"I work in Goregaon East as a {occupation}, and {availability} is the slot I can do right now."
        )
    return (
        "Hi, I'm looking for a flatmate-share near Goregaon East. "
        f"My budget is up to Rs. {budget} and I'm mainly considering {areas} "
        f"because I work as a {occupation}."
    )


def _seller_message(scenario: dict[str, Any]) -> str:
    seller = scenario.get("seller_profile")
    if not seller:
        return scenario.get("seller_initial_message", "")
    return (
        f"Hi, I want help listing a new flatmate-share opening in {seller['area']}. "
        f"The rent is around Rs. {_format_rs(int(seller['rent']))} for a {seller['listing_type']}. "
        "I can tell you more about the listing and available visit times."
    )


def _shift_amount(value: Any, delta: int) -> Any:
    if value is None:
        return None
    return max(1, int(value) + delta)


def apply_seed_variant(
    scenario: dict[str, Any],
    posts: dict[str, dict[str, Any]],
    seed: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return seeded copies of a scenario and its posts.

    The same seed always produces the same value variant for a scenario. A
    missing seed returns the original values, preserving existing tests and
    default behavior.
    """

    variant_scenario = deepcopy(scenario)
    variant_posts = deepcopy(posts)
    if seed is None:
        return variant_scenario, variant_posts

    rng = random.Random(f"{variant_scenario['task_id']}:{seed}")
    rent_delta = rng.choice(RENT_DELTAS)
    occupation = rng.choice(OCCUPATIONS)

    buyer = variant_scenario["buyer_profile"]
    buyer["budget_max"] = _shift_amount(buyer["budget_max"], rent_delta)
    if buyer.get("hidden_budget_ceiling") is not None:
        buyer["hidden_budget_ceiling"] = _shift_amount(buyer["hidden_budget_ceiling"], rent_delta)
    buyer["occupation"] = occupation

    expected_answers = variant_scenario["scenario_creation_config"].get("expected_answers", {})
    if "budget_max" in expected_answers:
        expected_answers["budget_max"] = buyer["budget_max"]
    if "occupation" in expected_answers:
        expected_answers["occupation"] = buyer["occupation"]

    negotiation_config = variant_scenario["scenario_creation_config"].get("negotiation_config", {})
    for key in ("buyer_ceiling", "seller_floor"):
        if key in negotiation_config:
            negotiation_config[key] = _shift_amount(negotiation_config[key], rent_delta)

    for post in variant_posts.values():
        post["rent"] = _shift_amount(post["rent"], rent_delta)

    seller = variant_scenario.get("seller_profile")
    if seller:
        seller["rent"] = _shift_amount(seller["rent"], rent_delta)
        followup = variant_scenario["scenario_creation_config"].get("followup_seller_expected_answers", {})
        if "rent" in followup:
            followup["rent"] = seller["rent"]

    variant_scenario["initial_user_message"] = _buyer_message(variant_scenario)
    if seller:
        variant_scenario["seller_initial_message"] = _seller_message(variant_scenario)
    variant_scenario["scenario_creation_config"]["variant"] = {
        "seed": seed,
        "rent_delta": rent_delta,
        "occupation": occupation,
    }
    return variant_scenario, variant_posts
