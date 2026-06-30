# [STUB · idea] The Model's Report Card: Scoring a Live World Cup Forecast

> Draft idea only — not yet written. Series: **live / results**. Originally idea "A".

**Subtitle (working):** What a calibrated forecaster actually got right when the matches were played
— calibration, proper scores, and hit-rate by risk profile, on real 2026 results.

**Estimated length / tags (planned):** ~12 min · Data Science, Machine Learning, Model Evaluation,
Sports Analytics, Forecasting.

## Main idea

Take the model's **pre-match** predictions for the 2026 World Cup and score them against what
actually happened, using the quiniela scoring rule (+1 correct 1X2, +2 exact score, +1 upset bonus)
*and* proper scoring rules (log-loss, Brier). The headline: a live **report card** — reliability of
the live predictions (calibration in the wild), per-match log-loss vs the `ln 3 ≈ 1.0986` baseline,
and **hit-rate / points by risk profile** (safe vs balanced vs aggressive vs contrarian). Show that
the aggressive/contrarian profiles are built to score on exactly the events (upsets, exact scores)
that a plain accuracy number would penalise.

## Planned sections / key points

- Calibration in the wild: reliability diagram on real 2026 matches (not historical backtest).
- Proper scores live: realised log-loss / Brier per matchday; baseline comparison.
- Decision-layer scorecard: realised quiniela points per profile under the real scoring rule.
- "Accuracy vs the right metric": why two yardsticks (belief vs decision) are needed.

## Non-redundancy

Distinct from article 02 (backtest = *historical* methodology). This is the **live 2026 scorecard**.

## Dependencies / feasibility

Needs the predictions **as they were before each match** (snapshots). Prospectively cheap (capture
each matchday's outputs before kickoff); retroactively needs the deferred pre-tournament
reconstruction. See memory `wc2026-pre-tournament-comparison-feature`. Closely related to stub 10
(E); decide whether to merge the exact-score analysis here.

## Suggested visuals

`hero08.png`; reliability diagram (live); bar of realised points per profile; per-matchday log-loss
line. Reuse `equations/logloss.png`.
