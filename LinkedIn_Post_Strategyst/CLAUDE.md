# CLAUDE.md — LinkedIn Topic-to-Growth System

Long-term project context for maintainers and future Claude sessions. Read this before
modifying the skill.

## Project Purpose

Provide a Claude Code skill that converts a single general topic into a complete LinkedIn
growth system: strategy, content pillars, a dated posting calendar, ready-to-publish posts
(in Spanish), and per-post image/carousel prompts — specific, credible, and non-generic.

## Overall Architecture

A **progressive-disclosure skill**: a lean `SKILL.md` orchestrator plus seven `references/`
files that are loaded only when the workflow reaches each step. This keeps the always-loaded
footprint small while supporting deep, step-specific guidance.

```
.claude/skills/linkedin-topic-to-growth/
  SKILL.md            # orchestrator: gate, input schema, workflow, output format
  references/
    clarifying-questions.md   # step 1
    content-pillars.md        # step 3
    chronogram.md             # step 4
    post-generation.md        # step 5
    image-prompts.md          # step 6
    growth-strategy.md        # step 7
    quality-checklist.md      # step 8 (validation gate)
```

Supporting docs (outside the skill): `README.md`, `docs/USER_GUIDE.md`,
`examples/example-session-es.md`, and `prompts/prompt.md` (the original spec / provenance).

## Design Philosophy

- **Ask before planning.** Generic output is the enemy; the clarifying-questions gate is
  the single most important design decision.
- **Specificity over volume.** Every artifact must be unmistakably about the user's topic.
- **Credibility over hype.** Never fabricate facts; mark assumptions explicitly.
- **Progressive disclosure.** Heavy guidance lives in references, loaded on demand.
- **Separation of concerns.** Each reference file owns exactly one stage of the workflow.

## Core Objectives

1. Strategic intake gate. 2. Strategy summary. 3. 4–6 content pillars. 4. Dated chronogram.
5. Full posts in Spanish. 6. Image prompts. 7. Growth strategy. 8. Quality validation.

## Functional Requirements

- Collect the 10 inputs (3 required, rest optional with defaults).
- Produce sections A–G exactly as defined in the source spec.
- All publish-ready post copy in **Spanish**; scaffolding/docs in English.
- Default period 30 days; calendar density follows posting frequency.
- Every post validated against the quality checklist before presentation.

## Non-Functional Requirements

- **Portability:** plain Markdown, no dependencies, no build step.
- **Maintainability:** one concern per file; defaults centralized.
- **Token efficiency:** references loaded only when needed.
- **Discoverability:** `SKILL.md` description is trigger-only ("Use when…").

## User Workflow

Invoke skill → answer 10 questions (or accept defaults) → receive A–G output + Assumptions
→ copy Spanish posts and image prompts → generate images externally → publish on schedule →
review weekly metrics.

## Internal Workflow

See the numbered workflow in `SKILL.md`. Order: gate → strategy → pillars → chronogram →
posts → image prompts → growth strategy → quality control → assumptions. Each step loads
its reference file just-in-time.

## Prompt Engineering Strategy

- **Trigger-only description** in frontmatter (per skill SDO): describes when to use, not
  what it does, so the body is actually read.
- **Hard gate** rendered as a small flowchart at a non-obvious decision point.
- **Red-flags list** to catch the high-risk failures (English copy, invented stats,
  generic motivation, planning before intake).
- **Defaults table** so the skill degrades gracefully when the user skips inputs.

## Content Generation Strategy

- Posts follow a fixed anatomy (hook → body → insight → takeaway → CTA → question →
  hashtags) for consistency and quality control.
- Tone and depth are mapped explicitly to inputs.
- Carousels produce slide-by-slide outlines plus a Spanish caption.
- Long calendars are batched (full posts for an initial batch + scheduled remainder) rather
  than truncated silently.

## LinkedIn Growth Philosophy

Consistency + audience relevance + genuine engagement beats virality tricks. Comments and
saves are weighted over likes; overt selling is capped at ~1 in 5 posts; posting times are
general guidance to be validated against the user's own analytics.

## Content Quality Standards

Codified in `quality-checklist.md`: strong hook, clear audience, concrete value, no generic
advice, readable formatting, one idea, meaningful CTA, no unsupported claims, not spammy,
human and credible. Plus system-level checks (language, pillar coverage, hook/format
variety, offer balance, constraints, assumptions, specificity).

## Validation Rules

- No post is presented until it passes the per-post gate.
- The `avoid` input is a hard constraint across all outputs.
- Every assumption appears in the Assumptions block.

## Assumptions (project-level)

- Output target is LinkedIn only.
- The user (or another tool) generates images from the prompts.
- Spanish is the post language unless the user overrides it.
- Posting-time data is general, not personalized.

## Constraints

- No posting to LinkedIn, no image generation, no external API calls within the skill.
- No fabricated statistics, studies, or quotes.

## Naming Conventions

- Skill name: `linkedin-topic-to-growth` (lowercase, hyphenated).
- Reference files: lowercase, hyphenated, named by the workflow stage they own.
- Output sections labeled A–G to match the source specification.

## File Organization

Skill under `.claude/skills/`; user-facing docs under root + `docs/`; worked example under
`examples/`; original spec preserved under `prompts/prompt.md`.

## Future Roadmap

- Analytics-driven posting-time personalization.
- LinkedIn publishing integration.
- Hook A/B generator.
- Multi-language output toggle.
- Brand-kit input feeding image prompts.

## Extension Points

- **Defaults:** `clarifying-questions.md` defaults table.
- **New pillar logic / formats:** `content-pillars.md`, `post-generation.md`.
- **New output sections:** add a reference file + a workflow step in `SKILL.md`.
- **Quality bar:** `quality-checklist.md`.

## Known Limitations

See README "Limitations". Quality scales with the specificity of user answers.

## Decisions Made During Development

- **Artifact = Claude Code skill** (not a raw system prompt or API tool JSON), per user
  choice — most usable inside Claude Code.
- **Structured folders** for repo layout, per user choice.
- **Docs in English, posts in Spanish**, per the source spec and user confirmation.
- **Progressive disclosure** over a single monolithic prompt, for maintainability and
  token efficiency.
- Original `prompts/prompt.md` kept untouched as provenance.

## Should Remain Configurable

Input defaults, pillar mix logic, post length/anatomy, quality thresholds, posting-time
recommendations, output language.

## Do Not Modify Without Careful Consideration

- The **clarifying-questions gate** (removing it reintroduces generic output).
- The **no-invented-facts** rule.
- The **Spanish post-copy** rule.
- The **trigger-only frontmatter description** (changing it to summarize the workflow can
  cause the body to be skipped).
