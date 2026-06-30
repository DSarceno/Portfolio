# LinkedIn Topic-to-Growth System

A Claude Code skill that turns a single general topic into a complete LinkedIn content and
growth system: strategy, content pillars, a dated posting calendar, ready-to-publish posts,
and an image/carousel prompt for every post.

> **Language policy:** documentation and strategy scaffolding are in English; all
> publish-ready LinkedIn post copy is generated in **Spanish**.

---

## Overview

The skill guides Claude through a structured workflow. It first asks the user ten
strategic questions, then produces a full, specific, ready-to-copy LinkedIn growth plan.
It is designed to avoid the two failure modes of AI content: generic filler and invented
facts.

## Purpose

Help a user grow authority, audience, clients, or a personal brand on LinkedIn around a
chosen topic — consistently and credibly — without starting from a blank page each week.

## Features

- **Strategic intake** — 10 clarifying questions gate every plan.
- **Strategy summary** — positioning, audience angle, main message, differentiation, goal.
- **4–6 content pillars** — each with purpose, value, post ideas, and best format.
- **Dated chronogram** — frequency-aware calendar (default 30 days) with hook, format,
  goal, CTA, and visual idea per post.
- **Full posts in Spanish** — hook, body, insight, takeaway, CTA, hashtags, engagement
  question, all publish-ready.
- **Image/carousel prompts** — one AI-image-generator brief per post.
- **Growth strategy** — posting times, commenting, networking, repurposing, engagement
  loops, weekly metrics.
- **Quality control** — every post passes a 10-point gate; assumptions are marked.

## Project Structure

```
LinkedIn_Post_Strategyst/
├── README.md                          # this file
├── CLAUDE.md                          # long-term project context for maintainers
├── .claude/skills/linkedin-topic-to-growth/
│   ├── SKILL.md                       # the skill: frontmatter, workflow, input schema
│   └── references/
│       ├── clarifying-questions.md    # the 10 questions + rationale + defaults
│       ├── content-pillars.md         # pillar strategy + format mapping
│       ├── chronogram.md              # calendar table structure
│       ├── post-generation.md         # post anatomy + rules (Spanish output)
│       ├── image-prompts.md           # image/carousel prompt rules
│       ├── growth-strategy.md         # times, commenting, networking, metrics
│       └── quality-checklist.md       # per-post + system quality gates
├── docs/
│   └── USER_GUIDE.md                  # non-technical step-by-step guide + FAQ
├── examples/
│   └── example-session-es.md          # worked example with Spanish output
└── prompts/
    └── prompt.md                      # original source specification (provenance)
```

## Installation

No build step. The skill is a set of Markdown files that Claude Code loads automatically.

1. Keep the `.claude/skills/linkedin-topic-to-growth/` folder in your project (or copy it to
   `~/.claude/skills/` to make it available in every project).
2. Open Claude Code in this repository.
3. The skill is discovered automatically; invoke it by name or by describing the task
   (e.g., "build me a LinkedIn growth plan about X").

## Configuration

Behavior is configured through the **inputs you provide when answering the clarifying
questions** — there are no config files. See the input schema in
[SKILL.md](.claude/skills/linkedin-topic-to-growth/SKILL.md) for fields, defaults, and
allowed values. To change defaults permanently, edit the defaults table in
[clarifying-questions.md](.claude/skills/linkedin-topic-to-growth/references/clarifying-questions.md).

## Inputs and Outputs

**Inputs (required):** topic, main goal, target audience.
**Inputs (optional):** LinkedIn level, posting frequency, tone, depth, formats, offer,
things to avoid, period length. Defaults apply when skipped.

**Outputs:** Strategy Summary (A), Content Pillars (B), Chronogram (C), Full Posts in
Spanish (D), Image/Carousel Prompts (E), Growth Strategy (F), and an Assumptions block.

## How the Workflow Operates

1. **Gate** — narrow a vague topic, then ask any missing required inputs. (No plan before
   answers.)
2. **Generate** — Strategy → Pillars → Chronogram → Posts → Image prompts → Growth
   strategy, loading each reference file only at its step.
3. **Validate** — run every post through the quality checklist; fix failures.
4. **Present** — clear A–G headings, tables, copy-ready post blocks, and an Assumptions
   block.

## Step-by-Step Usage Guide

1. Open Claude Code here.
2. Say what you want: *"Use the LinkedIn growth skill for the topic 'data engineering for
   startups'."*
3. Answer the 10 questions (or say "use defaults").
4. Review the strategy and calendar.
5. Copy the Spanish posts and the image prompts.
6. Generate images with your image tool of choice using the prompts.
7. Publish on the scheduled days; track the weekly metrics.

## Example User Session

See [examples/example-session-es.md](examples/example-session-es.md) for a complete worked
example (intake answers → strategy → pillars → calendar → Spanish posts → image prompts).

## Customization Options

- **Change defaults:** edit the defaults table in `clarifying-questions.md`.
- **Add/adjust pillars logic:** edit `content-pillars.md`.
- **Change post structure or length:** edit `post-generation.md`.
- **Adjust quality bar:** edit `quality-checklist.md`.
- **Output language:** the Spanish rule lives in `SKILL.md` and `post-generation.md`.

## Best Practices

- Answer the audience and offer questions concretely — specificity drives quality.
- Confirm posting times against your own analytics after 2–3 weeks.
- Edit generated posts in your own voice before publishing.
- Keep overt selling to roughly one in five posts.

## Limitations

- Does not post to LinkedIn or call the LinkedIn API; it produces copy and plans.
- Does not generate images; it produces prompts for an external image generator.
- Posting-time advice is general guidance, not personalized analytics.
- Quality depends on the specificity of the user's answers.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output feels generic | Vague intake answers | Re-answer with a specific audience and offer. |
| Posts came out in English | Language rule missed | Ask Claude to regenerate posts in Spanish. |
| Calendar too long to read | 30-day daily plan | Ask for week 1 in full, schedule the rest. |
| A claim looks invented | Unsourced stat slipped in | Ask Claude to remove/reframe; report which line. |
| Skill not found | Wrong location | Ensure the folder is under `.claude/skills/`. |

## Future Improvement Ideas

- Optional analytics-driven posting-time personalization.
- Direct LinkedIn API publishing integration.
- A/B hook generator for top posts.
- Multi-language post output toggle.
- Brand-kit input (fonts/colors) feeding the image prompts.

## License

License placeholder — add your chosen license here (e.g., MIT).
