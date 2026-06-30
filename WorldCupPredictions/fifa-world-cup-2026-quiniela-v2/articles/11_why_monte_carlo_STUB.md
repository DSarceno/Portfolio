# [STUB · idea] Why Monte Carlo? Simulating an Endogenous Tournament

> Draft idea only — not yet written. Series: **theory / foundations**. Originally idea "T1".

**Subtitle (working):** No closed form for a 48-team title race — the estimator, its variance, and
how many runs you actually need.

**Estimated length / tags (planned):** ~12 min · Statistics, Monte Carlo, Probability, Mathematics,
Sports Analytics.

## Main idea

Develop, formally, *why* Monte-Carlo is the right tool for tournament probabilities. The bracket is
**endogenous** — who you face in the round of 32 depends on group outcomes, which depend on the same
per-match probabilities you're propagating — so there is no closed form for P(champion). Define the
stochastic process, the Monte-Carlo estimator, its bias/variance, and the convergence rate; then
discuss how many simulations are needed for a target precision, plus reproducibility and
variance-reduction.

## Planned sections / key points

- Endogeneity: why the title probability has no analytic expression.
- The estimator: `p̂ = (1/N) Σ 1[event]`; unbiasedness; `Var(p̂) = p(1−p)/N`; SE ∝ 1/√N.
- Precision: runs needed for a given standard error / CI half-width; why N=10⁴ here.
- Reproducibility (per-run seeding) and variance reduction (common random numbers).
- Mapping to the code: group stage → best-thirds → R32 bracket → knockout, aggregated over runs.

## Non-redundancy

Distinct from article 04 (which is **performance** — caching + vectorization, the 200×–1500×
speedup). T1 is **why the estimator is correct and what it guarantees**. Cross-link the two: 04
explains how the variance-reducing precision (large N) is made affordable.

## Dependencies / feasibility

✅ Writable now (pure theory + existing simulator). Optionally add an empirical convergence plot
(p̂ vs N) — cheap to generate. Note the live caveat from memory
`wc2026-monte-carlo-sim-prematch-only` as honest "limitations" material.

## Suggested visuals

`hero11.png`; convergence plot (estimate vs N with ±SE band); endogenous-bracket schematic. Reuse
`equations/mc_estimator.png`, `equations/complexity.png`.
