"""Shared builders for Flatmate visit-scheduling scenarios."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_post(
    *,
    post_id: str,
    area: str,
    rent: int,
    diet: str,
    listing_type: str,
    commute_to_goregaon_mins: int,
    constraints: list[str],
    calendar_slots: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "id": post_id,
        "area": area,
        "rent": rent,
        "diet": diet,
        "type": listing_type,
        "commute_to_goregaon_mins": commute_to_goregaon_mins,
        "constraints": list(constraints),
        "calendar_slots": list(calendar_slots),
        "description": description,
    }


def build_buyer_profile(
    *,
    budget_max: int,
    dietary: str,
    areas: list[str],
    occupation: str,
    visit_availability: list[str],
    initial_disclosure_fields: list[str],
    hidden_additional_availability: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "budget_max": budget_max,
        "dietary": dietary,
        "areas": list(areas),
        "occupation": occupation,
        "visit_availability": list(visit_availability),
        "initial_disclosure_fields": list(initial_disclosure_fields),
        "hidden_additional_availability": list(hidden_additional_availability or []),
    }


def build_seller_profile(
    *,
    area: str,
    rent: int,
    dietary: str,
    listing_type: str,
    occupation_requirement: str,
    calendar_slots: list[str],
    description: str,
    commute_to_goregaon_mins: int,
    constraints: list[str],
) -> dict[str, Any]:
    return {
        "area": area,
        "rent": rent,
        "dietary": dietary,
        "listing_type": listing_type,
        "occupation_requirement": occupation_requirement,
        "calendar_slots": list(calendar_slots),
        "description": description,
        "commute_to_goregaon_mins": commute_to_goregaon_mins,
        "constraints": list(constraints),
    }


def build_ground_truth(
    *,
    optimal_posts: list[str],
    acceptable_posts: list[str],
    dealbreaker_posts: list[str],
    required_bookings: int,
    required_tool_calls: list[str],
    required_info: list[str],
    success_condition: str,
    min_viable_turns: int,
    schedule_feasible_posts: list[str] | None = None,
    max_schedule_feasible_visits: int | None = None,
) -> dict[str, Any]:
    payload = {
        "optimal_posts": list(optimal_posts),
        "acceptable_posts": list(acceptable_posts),
        "dealbreaker_posts": list(dealbreaker_posts),
        "required_bookings": required_bookings,
        "required_tool_calls": list(required_tool_calls),
        "required_info": list(required_info),
        "success_condition": success_condition,
        "min_viable_turns": min_viable_turns,
    }
    if schedule_feasible_posts is not None:
        payload["schedule_feasible_posts"] = list(schedule_feasible_posts)
    if max_schedule_feasible_visits is not None:
        payload["max_schedule_feasible_visits"] = max_schedule_feasible_visits
    return payload


def build_visit_scenario(
    *,
    task_id: str,
    label: str,
    difficulty: str,
    description: str,
    task_post_ids: list[str],
    buyer_profile: dict[str, Any],
    ground_truth: dict[str, Any],
    scenario_creation_config: dict[str, Any],
    initial_user_message: str,
    seller_profile: dict[str, Any] | None = None,
    seller_initial_message: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "label": label,
        "difficulty": difficulty,
        "description": description,
        "task_post_ids": list(task_post_ids),
        "buyer_profile": deepcopy(buyer_profile),
        "seller_profile": deepcopy(seller_profile),
        "ground_truth": deepcopy(ground_truth),
        "scenario_creation_config": deepcopy(scenario_creation_config),
        "initial_user_message": initial_user_message,
        "seller_initial_message": seller_initial_message or "",
    }
