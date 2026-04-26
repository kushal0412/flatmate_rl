# Flatmate RL: Training Agents For Real Flatmate Search

Finding a flatmate sounds simple until you actually do it.

You open a Facebook group, WhatsApp group, Telegram channel, or listing board. You see hundreds of posts. Some are repeated. Some are old. Some do not mention rent. Some do not mention diet, gender preference, deposit, move-in date, or whether visits are even possible.

Then the real work begins.

You message one person. They do not reply.

You message another. The room is already taken.

You find a good place, but the visit slot clashes with work.

You visit one flat and realize the area is too noisy.

You change your preferences, and now you have to start searching again.

This is why flatmate search is not just a search problem. It is a coordination problem.

## The Real Problem

Most flatmate platforms and groups are built like feeds.

They show posts, but they do not manage the search.

```text
+----------------------+      +----------------------+
| Flatmate group/feed  | ---> | User does all work   |
+----------------------+      +----------------------+
| Repeated posts       |      | Filter listings      |
| Missing details      |      | Message posters      |
| Outdated listings    |      | Check availability   |
| Unclear visit slots  |      | Remember preferences |
+----------------------+      +----------------------+
```

The interface gives you information, but the burden stays on you.

You have to remember which post matched your budget, which poster replied, which room was vegetarian-only, which visit was possible on Saturday, and which place you rejected after seeing it.

That is a lot of hidden work.

## What A Broker Actually Does

A good human broker does not just show listings.

They manage the process.

They ask for missing preferences. They filter bad leads. They check whether a flat is still available. They coordinate with owners or current flatmates. They schedule visits. They remember feedback after each visit.

In other words, a broker turns a messy feed into a workflow.

```text
+-------------+     +-------------+     +-------------+
| Understand  | --> | Match       | --> | Coordinate  |
| the buyer   |     | listings    |     | visits      |
+-------------+     +-------------+     +-------------+
        |                                      |
        v                                      v
+-------------+                        +-------------+
| Update      | <--------------------- | Learn from  |
| preferences |                        | feedback    |
+-------------+                        +-------------+
```

Flatmate RL is built around this idea: train an agent to behave more like a reliable broker than a passive search box.

## Feed Search vs Agent Search

Here is the difference in concrete terms.

| Situation | Feed-based search | Agent-led search |
| --- | --- | --- |
| Repeated listing | You see the same Andheri West room posted multiple times. | The agent treats duplicates as one lead and checks if it is still active. |
| Missing details | A post says "DM for rent" and gives no diet or move-in details. | The agent asks for missing fields before considering it a serious option. |
| Budget mismatch | You skip a Rs. 24,000 room because your budget is Rs. 20,000. | The agent can check whether negotiation is possible. |
| Visit timing | You ask "Can I visit Saturday?" and wait for a reply. | The agent checks open slots and proposes only valid times. |
| Preference change | After a visit, you realize you need a quieter area. | The agent updates your profile and changes future recommendations. |
| New listing later | A good seller appears two days after you stopped checking. | The agent can match new supply back to your saved preferences. |

That is the product gap Flatmate RL is trying to model.

## What Flatmate RL Is

Flatmate RL is an OpenEnv reinforcement-learning environment for training broker-style agents.

The agent is placed inside a simulated flatmate-search workflow. It sees the current search state: preferences, listings, messages, calendar slots, and previous actions.

At every step, it must choose one action:

- send a message when more information or confirmation is needed
- call a structured broker tool

The goal is not just to recommend a post. The goal is to complete the workflow correctly.

## The Environment In One Diagram

```text
+-----------------------------+
| Current housing situation   |
+-----------------------------+
| Buyer preferences           |
| Seller/listing details      |
| Chat history                |
| Calendar slots              |
+-------------+---------------+
              |
              v
+-----------------------------+
| Broker agent decides        |
+-----------------------------+
| Ask a question?             |
| Search posts?               |
| Check slots?                |
| Contact poster?             |
| Book visit?                 |
+-------------+---------------+
              |
              v
+-----------------------------+
| Environment checks action   |
+-----------------------------+
| Did it follow the rules?    |
| Did it move the task ahead? |
| Should it get reward?       |
+-------------+---------------+
              |
              v
+-----------------------------+
| Updated state               |
+-----------------------------+
| New facts are stored        |
| Bad actions are penalized   |
| Good progress is rewarded   |
+-----------------------------+
```

This loop teaches the model when to talk, when to use tools, what arguments to pass, and when it is safe to book a visit.

## The Tools

The agent has tools that mirror the work a broker would do.

Buyer-side tools include:

- `store_user_details`
- `search_posts`
- `match_location_preference`
- `get_commute_time`
- `check_calendar_slots`
- `shortlist`
- `contact_poster`
- `book_viewing`

Advanced tools handle seller follow-up, negotiation, waitlists, cancellations, and feedback after visits.

The important part is sequencing.

The agent cannot just book a visit because a listing looks good. It first needs enough buyer details, a matching listing, available calendar slots, buyer confirmation, and poster confirmation.

## Example: Booking One Visit

Suppose a buyer says:

> I want a room near Andheri West, budget around Rs. 22,000. I work in media.

That is not enough to book a visit.

The agent should first ask for missing details, such as diet preference and visit availability.

Then it can search posts, match the area, check commute, inspect calendar slots, contact the poster, and book only after both sides agree.

```text
Buyer request
     |
     v
Ask missing details
     |
     v
Search and match listings
     |
     v
Check visit slots
     |
     v
Confirm buyer + poster
     |
     v
Book viewing
```

This is a small example, but it captures the main difference.

A search engine can return a listing. A broker agent has to finish the job.

## Example: Preferences Change After A Visit

Flatmate search changes after real visits.

A buyer may start by saying they want Andheri West. After visiting one flat, they may realize the building is too noisy. After another visit, they may decide they need a gym nearby.

A normal feed does not remember that.

Flatmate RL rewards an agent that can debrief the visit, update the buyer profile, and continue the search with the new information.

```text
Visit flat
   |
   v
Buyer gives feedback
   |
   v
Agent updates preferences
   |
   v
New search is more specific
```

This matters because real users do not know every preference upfront. A useful agent must learn during the process.

## Example: Negotiation

Sometimes a listing looks too expensive, but the deal is still possible.

Imagine a room listed at Rs. 24,000. The buyer says their budget is Rs. 20,000, but they can stretch to Rs. 22,000. The seller wants Rs. 24,000, but would accept Rs. 21,000.

A static filter may reject the listing.

A broker agent can check whether there is overlap and propose a price that both sides accept.

That is another reason this is not just retrieval. The best outcome may require several careful steps.

## Why Reinforcement Learning Fits

Flatmate search has delayed outcomes.

An early mistake can break the whole workflow. If the agent books before checking calendar slots, the booking is invalid. If it contacts a poster before collecting buyer details, the poster may not have enough information. If it ignores feedback after a visit, it keeps recommending the wrong places.

Reinforcement learning is useful because the model can learn from the result of the whole sequence, not just from one message.

Good behavior gets rewarded:

- collecting missing details
- using tools in the right order
- avoiding unavailable slots
- getting both sides to confirm
- adapting after feedback

Bad behavior gets penalized:

- calling tools too early
- repeating actions
- booking without consent
- ignoring calendar conflicts
- getting stuck in loops

Over time, the overall loss should go down because the agent makes fewer workflow errors. In this environment, lower error means fewer invalid tool calls, fewer missed confirmations, fewer bad slot choices, and more completed bookings or deals.

## What Success Looks Like

A strong Flatmate RL agent should feel less like a chatbot and more like an operational assistant.

It should remember what the buyer wants. It should search continuously. It should filter noisy supply. It should coordinate with sellers. It should handle calendar conflicts. It should negotiate when there is room. It should adapt after visits.

Most importantly, it should only book when the workflow is actually valid.

That is the real-world problem Flatmate RL tries to make trainable: turning messy flatmate search into a repeatable agent task with state, tools, feedback, and measurable progress.
