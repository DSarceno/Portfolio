# [STUB · idea] Scorelines, Not Just Winners: How Close Were the Exact-Score Predictions?

> Draft idea only — not yet written. Series: **live / results**. Originally idea "E".

**Subtitle (working):** An empirical look at the Poisson/Dixon-Coles scoreline grid against real
2026 results — and what the +2 exact-score bonus is really worth.

**Estimated length / tags (planned):** ~10 min · Statistics, Data Science, Sports Analytics,
Probability.

## Main idea

The system predicts not just H/D/A but the **most-likely exact scoreline** (top-3) via the Poisson
Dixon-Coles grid. Now that real matches exist, measure empirically: how often was the modal
scoreline exactly right; how often was the outcome right but the score wrong; the realised value of
the +2 exact-score bonus; and how the modal-draw tendency played out. A concrete, evidence-driven
companion to the theory.

## Planned sections / key points

- Recap (brief) of the scoreline grid: Poisson rates → Dixon-Coles low-score correction → grid.
- Exact-score hit rate on real 2026 matches; distribution of |predicted − actual| goals.
- Outcome-right-but-score-wrong rate; where the modal draw helped or hurt.
- Realised expected points from the exact-score bonus across the profiles.

## Non-redundancy

Complements article 01 (the *mathematics* of Skellam/Poisson/Dixon-Coles) with **live empirical
evidence**. **Overlaps with stub 08 (A)** on the exact-score-points angle — decide early whether to
merge into A's report card or keep as a focused deep-dive.

## Dependencies / feasibility

Needs pre-match scoreline predictions (snapshots), like A. Reuse existing
`outputs/predictions/scoreline_predictions.csv` captured per matchday.

## Suggested visuals

`hero10.png`; confusion-style heatmap of predicted vs actual scorelines; bar of exact-hit rate by
stage. Reuse `equations/poisson_pmf.png`, `equations/dixon_coles.png`, `equations/grid_sums.png`.
