# Flatmate RL: Training Broker Agents for Real Flatmate Search

Finding a flatmate-share is rarely a clean search problem. A person who has just moved to a city has to scan flat and flatmate groups, sort through repeated posts and spam, identify listings that match their budget and preferences, message owners or current flatmates, check whether the place is still available, coordinate visit slots, and then repeat the whole process when one detail does not line up.

The hard part is not only "find a relevant listing." The hard part is the loop:

1. Find posts that might match.
2. Filter out bad or unavailable listings.
3. Ask the buyer for missing preferences.
4. Contact the listing owner or flatmate.
5. Check whether the flat is still available.
6. Match both sides on budget, lifestyle, location, commute, and visit time.
7. Schedule visits without double-booking or assuming consent.
8. Keep going when preferences change after real visits.

At the end of this process, most people still do not have a clean path forward. They end up contacting multiple brokers, repeating the same preferences, re-checking the same details, and hoping one broker happens to have the right listing at the right time. The search becomes painful, slow, and unreliable.

This is the problem Flatmate RL models. A human broker is essentially an agent operating in a messy housing environment: they have access to listings, talk to owners and buyers, check calendars, negotiate on behalf of both sides, and carry context across days or weeks. Flatmate RL turns that workflow into an OpenEnv reinforcement-learning environment.

## The Environment At A Glance

```text
+---------------------------+
|  Flatmate RL Environment  |
+---------------------------+
| Buyer and seller chats    |
| Listings and preferences  |
| Calendar slots            |
| Tool rules and rewards    |
+-------------+-------------+
              |
              v
+---------------------------+
|        Broker Agent       |
+---------------------------+
| Reads the current state   |
| Chooses the next action   |
+-------------+-------------+
              |
              v
+---------------------------+
|          Action           |
+---------------------------+
| 1. Send a message         |
| 2. Call a broker tool     |
+-------------+-------------+
              |
              v
+---------------------------+
|       Updated State       |
+---------------------------+
| New facts are stored      |
| Mistakes are penalized    |
| Valid progress is rewarded|
+---------------------------+
```

This is the core training loop. The agent sees the current housing situation, chooses one next step, and the environment checks whether that step moved the broker workflow forward.

## What Flatmate RL Models

Flatmate RL simulates a broker for flatmate-share discovery and visit scheduling. The agent does not just rank listings. It must make step-by-step decisions under constraints.

At each step, the policy can either:

- send an `assistant_message` to the active buyer or seller
- call a broker tool with structured JSON arguments

The environment tracks the state that matters in a real broker workflow:

- buyer and seller conversation history
- missing required fields such as diet, areas, occupation, budget, and visit availability
- selected posts and matched listings
- available broker tools for the current phase
- calendar slots, pre-booked slots, and visit confirmations
- booked visits
- tool trace and policy violations
- step reward and total reward

That makes the task more useful than a static recommendation benchmark. The model has to learn the process that converts an uncertain lead into a confirmed visit or deal.

The state is not just a chat transcript. It carries operational facts like selected posts, available slots, already-booked times, confirmations, and failed tool calls. That is why the policy has to learn both conversation and execution.

## Why This Needs an Agent

Flat hunting through social posts breaks down because relevant information is scattered and time-sensitive. A post may be good but buried in spam. A listing may look relevant but already be booked. A buyer may say they are only available Tuesday, but become flexible when shown strong Saturday or Sunday options. A seller may arrive later with a better listing after the first buyer flow fails.

A useful broker agent needs to handle those cases without losing state. It should know when to ask a question, when to search, when to match by area, when to check commute, when to ask the buyer to choose between options, when to contact a poster, and when it is finally safe to book.

Flatmate RL makes those decisions trainable and measurable.

## Tools The Agent Learns To Use

The broker action space is intentionally mixed: natural language plus structured tools.

Buyer-side tools include:

- `store_user_details`
- `search_posts`
- `match_location_preference`
- `get_commute_time`
- `check_calendar_slots`
- `shortlist`
- `contact_poster`
- `book_viewing`
- `close_buyer_conversation`

Seller and advanced-flow tools include:

- `store_seller_details`
- `check_table_slot_matches`
- `confirm_seller_match`
- `offer_matched_listing_to_buyer`
- `schedule_table_visit`
- `propose_price_to_buyer`
- `propose_price_to_seller`
- `confirm_negotiated_deal`
- `add_to_waitlist`
- `notify_buyer_slot_freed`
- `debrief_visit`
- `filter_new_arrivals`

The environment enforces sequencing. For example, searching before storing user details fails. Booking before checking calendar slots fails. Booking without both buyer and poster confirmation fails. Repeating the same successful tool call gets penalized. These constraints force the model to learn a broker workflow instead of learning to call tools randomly.

## Scenarios From The Code

The scenario definitions live in [`server/scenarios.py`](server/scenarios.py). They are built to represent common real flat-search problems rather than toy tasks.

| Scenario | Real-world problem | What the agent must learn |
| --- | --- | --- |
| `task_visit_single` | A buyer wants one suitable flatmate-share near Andheri West or Jogeshwari. | Ask for missing diet and availability, search, match listings, check slots, contact the poster, and book only after both sides confirm. |
| `task_visit_single_hidden_flex` | The buyer first shares only Tuesday availability, but good listings have Saturday or Sunday slots. | Surface concrete alternative slots and unlock hidden flexibility instead of giving up too early. |
| `task_visit_multi` | The buyer wants to compare multiple options before deciding. | Shortlist several matching listings, ask which ones to pursue, and book at least two non-overlapping visits. |
| `task_visit_single_seller_followup` | No current listing fits, then a new seller contacts the broker with a relevant property. | Close the first buyer flow, gather seller details, create a new listing, match it to the saved buyer, and schedule the visit. |
| `task_negotiation_hidden_budget` | A listing is above the buyer's stated budget, but both sides have hidden negotiation room. | Probe buyer and seller price limits, discover overlap, and close a negotiated deal. |
| `task_slot_cancellation_waitlist` | A desired flat has all slots pre-booked, but a cancellation opens one later. | Add the buyer to a waitlist, react when the slot opens, notify the buyer, and complete the booking. |
| `task_multi_visit_preference_evolution` | The buyer only discovers true preferences after visiting places. | Debrief after visits, update the buyer profile, filter new arrivals, and keep searching until the final match is found. |
| `task_visit_conflict_check` | A listing shows several slots, but some are already taken by other buyers. | Read `pre_booked_slots`, avoid unavailable times, and propose only the actually open slot. |

These examples mirror the lifecycle of a real broker conversation. The best action is often not a single recommendation. It is the next correct operational step.

## Example: A Single Confirmed Visit

In `task_visit_single`, the buyer starts with budget, area, and occupation. The agent still needs diet and visit availability. A successful policy learns a sequence like this:

```json
{"action_type": "assistant_message", "assistant_message": "Please share your dietary preference and visit availability."}
```

Then it can store details, search posts, match locations, check commute, and inspect calendar slots:

```json
{"action_type": "tool_call", "tool_name": "check_calendar_slots", "tool_arguments": {"post_ids": ["post_023"]}}
```

Only after the buyer confirms the slot and the poster confirms both the buyer profile and the same time should the agent book:

```json
{"action_type": "tool_call", "tool_name": "book_viewing", "tool_arguments": {"post_id": "post_023", "time_text": "Saturday 11am"}}
```

That final booking is not just a search result. It is the result of a valid chain of information gathering, matching, calendar checking, and consent.

## Example: Preferences Change After Visits

The `task_multi_visit_preference_evolution` scenario captures a problem brokers see often: buyers do not know every preference upfront.

The buyer first visits `post_023` and realizes the area is too noisy. The agent must debrief the visit and update the profile. Then new listings arrive. Some are relevant, some are not. After another visit, the buyer realizes they also want a nearby gym. The agent has to filter new arrivals again and keep searching until it finds `post_067`, which is quiet and has gym access nearby.

This is important for model training because the target changes during the episode. The model must improve its plan as it gathers real feedback.

## Example: Negotiation Is A Sequential Skill

In `task_negotiation_hidden_budget`, the listed rent is Rs. 24,000. The buyer's stated budget is Rs. 20,000, but their hidden ceiling is Rs. 22,000. The seller's hidden floor is Rs. 21,000.

A static matcher might reject the listing as too expensive. A broker agent should recognize that the listing is negotiable, probe both sides, discover the overlap, and close at an accepted rent. The scenario rewards the model for using `propose_price_to_buyer`, `propose_price_to_seller`, and `confirm_negotiated_deal` in the right order.

That makes the environment closer to real housing workflows, where a deal can take multiple rounds rather than one retrieval call.

## Reward And Evaluation

The reward design encourages operational correctness:

- small positive reward for successful tool progress
- penalties for failed tools, invalid order, redundant calls, calendar mismatches, missing consent, and loops
- completion reward when the scenario's required booking or deal condition is satisfied

Regression tests in [`tests/test_flatmate_rl.py`](tests/test_flatmate_rl.py) and [`tests/test_reward_regression.py`](tests/test_reward_regression.py) check core behavior such as single-visit booking, hidden flexibility, multi-booking, seller follow-up, strict evaluation hiding, and reward stability.

The result is an environment where model performance can improve in a measurable way: fewer invalid tool calls, better ordering, more valid bookings, and stronger handling of edge cases like pre-booked slots or changed preferences.

## Training Shape

The notebooks in this repository train policies that emit Flatmate RL JSON actions against the live Space endpoint. The practical training path is:

1. Collect valid broker trajectories from heuristic or scripted policies.
2. Supervise a model to emit well-formed action JSON.
3. Fine-tune with RL so the model learns from delayed success, failed tools, and environment feedback.
4. Evaluate on the same scenario IDs with fixed seeds and reward regression checks.

The important part is that the model is not only learning what to say. It is learning when to talk, when to use tools, what arguments to pass, and how to recover when the housing workflow changes.

## Why This Matters

A better flatmate search system should not behave like a passive search box. It should behave like a reliable broker that works for the user:

- it remembers preferences
- it searches and filters continuously
- it checks whether listings are still available
- it coordinates with owners and flatmates
- it handles calendar conflicts
- it negotiates when there is room
- it follows up when new supply appears
- it books only after both sides consent

Flatmate RL provides a controlled environment for training that kind of agent. It turns the messy broker workflow into repeatable episodes with structured state, realistic scenarios, strict tool contracts, and measurable rewards.

The end goal is not just to find a flat post. The goal is to train models that can manage the whole housing search loop until a buyer and a compatible flatmate or owner reach a real conclusion.
