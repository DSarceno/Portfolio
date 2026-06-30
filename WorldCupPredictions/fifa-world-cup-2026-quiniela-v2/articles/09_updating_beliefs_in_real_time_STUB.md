# [STUB · idea] Updating Beliefs in Real Time: How a Forecaster Changes Its Mind Every Matchday

> Draft idea only — not yet written. Series: **live / results**. Originally idea "B".

**Subtitle (working):** The retraining loop as Bayesian updating — watching ratings and tournament
probabilities move as results arrive.

**Estimated length / tags (planned):** ~12 min · Machine Learning, Data Science, Bayesian
Inference, Sports Analytics, MLOps.

## Main idea

Every matchday the system re-ingests results, refits Elo/PI/form ratings and the models, and
regenerates probabilities. Frame this loop as **belief updating**: show the concrete drift of a few
teams' qualification / round-reached / championship probabilities across matchdays, and connect the
mechanics (time-decayed Elo, warm-start retraining, feature refresh) to the intuition of a posterior
moving as evidence accumulates.

## Planned sections / key points

- The update loop, formally: prior (pre-matchday) → evidence (results) → posterior (new probs).
- Rating dynamics: how a single upset moves Elo (K-factor, goal-margin multiplier, time decay).
- Probability drift: trajectories of selected teams' P(advance) / P(champion) over matchdays.
- Stability: why shallow/shrunk models move smoothly rather than lurching (ties to article 07).

## Non-redundancy

Distinct from article 03 (architecture/reproducibility). This is the **dynamics of the belief**, not
the plumbing.

## Dependencies / feasibility

⚠️ The **P(champion) / round-reached** trajectories depend on the Monte-Carlo simulator, which is
currently pre-match-only and degrades/aborts mid-late tournament — see memory
`wc2026-monte-carlo-sim-prematch-only`. **Write this AFTER the post-group simulator fix.** The
ratings/per-match-probability drift can be shown earlier, but the marquee championship-drift chart
needs the fix first. Also benefits from matchday output snapshots.

## Suggested visuals

`hero09.png`; multi-line "probability drift" chart per matchday; an Elo-jump-after-upset diagram.
