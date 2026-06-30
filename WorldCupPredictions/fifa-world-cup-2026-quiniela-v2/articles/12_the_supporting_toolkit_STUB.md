# [STUB · idea] The Supporting Toolkit: Calibration, Ensembling & LASSO

> Draft idea only — not yet written. Series: **theory / foundations**. Originally idea "T3" (optional
> but recommended).

**Subtitle (working):** The mathematical "glue" that turns several model outputs into one honest,
parsimonious probability.

**Estimated length / tags (planned):** ~12 min · Machine Learning, Statistics, Data Science,
Mathematics.

## Main idea

The pieces that the headline articles only mention in passing, given a proper formal treatment:
**multinomial logistic regression** (the interpretable linear baseline), **LASSO / L1** feature
selection (sparsity, the soft-threshold), **probability calibration** (Platt/sigmoid + isotonic,
reliability and ECE), and the **probability ensemble / blending** (how the four models are pooled
into one calibrated vector). The thesis: the credibility of the whole system rests on this glue —
selection keeps the features honest, calibration keeps the probabilities honest, blending keeps any
single model from dominating.

## Planned sections / key points

- Multinomial logit: softmax model, the convex log-loss objective, why it's the interpretable floor.
- LASSO: `min ½||y−Xβ||² + α||β||₁`; the soft-thresholding / sparsity geometry; why it prunes
  collinear weak features.
- Calibration: Platt (sigmoid) vs isotonic (monotone, non-parametric); reliability diagram; ECE.
- Ensembling: the weighted blend (0.35/0.20/0.30/0.15) — linear vs log/geometric pooling; why
  calibration comes *after* blending.

## Non-redundancy

Poisson/Skellam/Dixon-Coles/Elo/Bayesian-shrinkage are **already covered by article 01** — do NOT
re-derive them here; cross-link instead. This article owns the *combination + calibration + selection*
layer specifically.

## Dependencies / feasibility

✅ Writable now (theory + real config: `model_params.yaml` weights and `calibration: isotonic +
sigmoid`). No data snapshots needed.

## Open decision

Confirms the Poisson question: recommended path (a) = no standalone Poisson article (covered by 01),
this toolkit piece fills the remaining theoretical gaps. Revisit only if a Poisson deep-dive with a
genuinely new angle is wanted.

## Suggested visuals

`hero12.png`; reliability diagram (before/after calibration); LASSO coefficient path; a blend
schematic. Reuse `equations/composite.png`, `equations/logloss.png`.
