# LinkedIn Chronogram (Posting Calendar)

Build a dated calendar for `period_days` (default **30**). Rotate the content pillars so no
two consecutive posts share a pillar where avoidable. Match the number of posts to
`posting_frequency`.

## Frequency → Cadence

| Frequency | Posts in 30 days | Pattern |
|-----------|------------------|---------|
| `daily` | ~30 | every day |
| `3x_week` | ~13 | e.g., Mon / Wed / Fri |
| `weekly` | ~4 | one anchor day (e.g., Tue) |

If `period_days` differs from 30, scale proportionally.

## Calendar Table Structure

Render the calendar as a table. Each scheduled post is one row:

| Field | Description |
|-------|-------------|
| **Date / Day #** | Absolute date when known, else "Día N". |
| **Pillar** | Which content pillar this post belongs to. |
| **Topic** | The specific subject of the post. |
| **Hook** | The opening line / scroll-stopper (one sentence). |
| **Format** | text / carousel / image / poll / article. |
| **Goal** | What this single post is meant to achieve (reach, replies, saves, clicks, DMs). |
| **CTA** | The action requested of the reader. |
| **Image / Carousel idea** | One-line visual concept (expanded later in image prompts). |

Example row:

| Día 1 | Pilar 2: Autoridad | Por qué la mayoría mide mal la retención | "La métrica que crees que importa probablemente te está mintiendo." | Carrusel | Saves + autoridad | "Guárdalo para tu próxima revisión de métricas." | Carrusel de 6 slides, gráfico de cohortes |

## Rules

- **Pillar rotation:** distribute pillars evenly across the period; cluster Convert posts
  sparsely (~1 in 5).
- **Format variety:** honor the user's `formats` choice; if `mixed`, vary across rows.
- **Hooks are unique:** no two rows may reuse the same hook structure.
- **Goal per post is singular:** one clear objective per row.
- **Anchor cadence:** for weekly/3x cadences, keep posting days consistent week to week.
- The chronogram is the single source of truth that the full posts (section D) expand from.
