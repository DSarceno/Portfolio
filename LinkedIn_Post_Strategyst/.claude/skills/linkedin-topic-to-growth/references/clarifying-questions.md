# Clarifying Questions (The Gate)

Ask these **before** producing any plan. Present them as a single numbered list so the
user can answer in one message. If the user answers only some, fill the rest with the
documented defaults from the input schema and flag each as an assumption.

Ask the questions in the user's language. Remind the user that the **posts themselves will
be written in Spanish**.

## The 10 Questions and Why Each Matters

| # | Question | Why it is asked |
|---|----------|-----------------|
| 1 | **What is the general topic?** | The seed of every pillar, post, and image. Without a concrete topic, content is generic. |
| 2 | **What is your main goal on LinkedIn?** (grow followers, get clients, build authority, attract recruiters, sell a product, educate, build a personal brand) | Determines CTAs, post angles, and which metrics matter. A "get clients" plan looks very different from a "build authority" plan. |
| 3 | **Who is your target audience?** | Controls vocabulary, depth, pain points, and examples. Posts must speak to a specific reader. |
| 4 | **What is your current LinkedIn level?** (beginner, intermediate, advanced) | Sets realistic frequency, format ambition, and how much foundational vs. advanced content to include. |
| 5 | **How often do you want to post?** (daily, 3x/week, weekly) | Defines calendar density and the number of full posts to draft. |
| 6 | **What tone should posts use?** (professional, bold, educational, inspirational, technical, simple, storytelling) | Shapes voice, sentence rhythm, and hook style across every post. |
| 7 | **Beginner-friendly, expert-level, or mixed?** | Controls technical depth and how much jargon is defined inline. |
| 8 | **Mostly text, carousels, image posts, polls, articles, or mixed?** | Drives the format column of the chronogram and the image-prompt workload. |
| 9 | **What offer / product / service / personal brand should this support?** | Anchors CTAs and the conversion narrative without making every post salesy. |
| 10 | **Anything to avoid?** (topics, claims, styles, formats) | Hard constraints. Never violate these in any output. |

## Topic-Narrowing Sub-Question

If the topic is too broad or ambiguous (e.g., "technology", "finance", "AI"), ask ONE
focused follow-up before the main 10, such as:

> "«[topic]» is broad — which slice is yours? For example: a specific subfield, a problem
> you solve, a method you use, or an audience you serve."

## Defaults When the User Skips

If the user says "just go" or skips optional answers, use these and list them under
**Assumptions**:

- main_goal → `authority`
- target_audience → inferred from topic
- linkedin_level → `intermediate`
- posting_frequency → `3x/week`
- tone → `professional + educational`
- depth → `mixed`
- formats → `mixed`
- offer → personal brand (no hard sell)
- avoid → none
- period_days → `30`
