# Project Summary — FIFA World Cup 2026 Quiniela Predictor V2

Publication-ready technical content derived from the **FIFA World Cup 2026 Quiniela Predictor
V2** repository. Every article is grounded in the project's own documentation
(`CLAUDE.md`, `MARCO_TEORICO.md`, `docs/MODELING_GUIDE.md`, `README.md`) and implementation —
no invented results.

- **Medium articles:** English. **LinkedIn posts + carousels:** Spanish.
- **Fully visual and Medium-ready:** every display equation is a rendered PNG (Medium does not
  render LaTeX), inline math is converted to Unicode, and each article carries a hero banner,
  2–3 data figures/diagrams, captions, and pull quotes.

## What the system is

An end-to-end quantitative forecasting system for the 2026 FIFA World Cup (48 teams, 12 groups,
104 matches). It predicts outcomes (H/D/A) and scorelines, runs a Monte-Carlo simulation of the
whole tournament, and emits **quiniela pick sheets in four risk profiles** (`safe`, `balanced`,
`aggressive`, `contrarian`), updated matchday-by-matchday. Its explicit objective is
**maximizing expected value in prediction pools**, not academic accuracy: a calibrated
probability layer is deliberately decoupled from a pick-selection layer.

## Technical pillars

- **Ratings:** competition-aware Elo (K=80 WC, K=8 friendly) + goal-margin multiplier + time
  decay; Constantinou–Fenton dual PI rating; rolling form; z-scored ensemble; James–Stein
  shrinkage toward confederation priors.
- **Outcome models:** multinomial logistic + XGBoost + Poisson, **Skellam** for H/D/A and a
  Dixon-Coles grid for exact scorelines; LASSO feature selection.
- **Squad value (A.2):** Transfermarkt values (Kaggle `player-scores`) by citizenship as-of
  date — most predictive covariate after Elo.
- **Simulation:** Monte-Carlo with prediction caching + vectorized group stage (200×–1500×);
  every pair pre-scored with the full blended model.
- **The arbiter:** a leakage-free cross-tournament backtester (train `< Y`, test `== Y`) that
  accepts/rejects every modelling change by held-out log-loss.

## Deliverables

| # | Medium article (EN) | LinkedIn post (ES) | Carousel (ES) |
|---|---|---|---|
| 1 | `01_skellam_over_score_grid.md` | `01_linkedin_post.md` | `01_linkedin_carousel.md` |
| 2 | `02_the_champion_list_is_a_liar.md` | `02_linkedin_post.md` | `02_linkedin_carousel.md` |
| 3 | `03_one_source_of_truth.md` | `03_linkedin_post.md` | `03_linkedin_carousel.md` |
| 4 | `04_caching_a_monte_carlo_simulator.md` | `04_linkedin_post.md` | `04_linkedin_carousel.md` |
| 5 | `05_probability_is_not_a_decision.md` | `05_linkedin_post.md` | `05_linkedin_carousel.md` |

Each article is 1,800–3,000 words and follows the requested structure (title, subtitle, reading
time, SEO keywords, Medium tags, then Introduction → Background Theory → System Design →
Experiments & Results → Lessons Learned → Future Work → Conclusion).

## Visual assets

- `figures/` — 5 hero banners (`hero01–05.png`) + 12 data figures/diagrams
  (`fig01–fig12_*.png`): Skellam-vs-grid, time decay, shrinkage, the squad-value A/B, the
  mixture-prior verdict, the leakage-free protocol, the layered architecture, the single source
  of truth, the cache-cost curve, the champion reshuffle, the four-policy diagram, the
  pari-mutuel trade-off.
- `equations/` — 13 rendered equation PNGs embedded in the articles.
- `carousels/` — 35 Spanish carousel slides (5 × 7), 1080×1080, ready to post.

## Regenerating the visuals

Two helper scripts (run from `articles/`, needs `matplotlib`, `numpy`, `scipy`):

```bash
python _build_visuals.py          # regenerate all figures, equations, heroes, carousels
python _convert_inline_math.py    # convert any remaining inline $...$ to Unicode
```

## How to publish

- **Medium:** paste/import an `0N_*.md` article; images import from the relative `figures/` and
  `equations/` paths (host them or let Medium fetch on import). Equations and diagrams render as
  images, so nothing breaks.
- **LinkedIn:** paste the `0N_linkedin_post.md` text as the body; attach the 7 slides from the
  matching `0N_linkedin_carousel.md` (upload in order, or export to a PDF and post as a
  *document*).

## Note on figures

A few illustrative figures (e.g. the exact 200×–1500× span, the champion-rank reshuffle) encode
numbers reported in the project's own notes rather than a fresh pipeline run. They are faithful
to the documentation; re-run the pipeline if you want to refresh them before publishing.
