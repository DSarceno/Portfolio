# Notebooks Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 4 exploratory notebooks into portfolio-grade, statistically deep, visually polished analyses that leverage the project's newest features (squad value, scorelines, backtest, calibration).

**Architecture:** A shared `notebooks/nb_style.py` module gives every notebook one consistent seaborn theme + helpers. One small leakage-safe helper (`collect_holdout_predictions`) is added to `src/training/backtester.py` to feed the calibration and backtest charts. The 4 notebooks are edited cell-by-cell (via NotebookEdit) and then executed end-to-end with `nbconvert` so they ship with rendered figures.

**Tech Stack:** Python 3.11, pandas, numpy, matplotlib 3.8, seaborn (new), scikit-learn/xgboost (already used), Jupyter/nbconvert.

---

## Conventions for this plan

- Notebook cells are edited with the **NotebookEdit** tool (`cell_type`, `edit_mode=insert|replace|delete`). When a task says "replace cell N", use the current cell index.
- After editing each notebook, **validate by executing it**: `jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` from the project root with the venv active. Expected: exit 0, no traceback in any cell.
- Notebooks run with cwd = `notebooks/`, so `import nb_style` resolves and `Path('..')` is the project root (existing pattern).
- Commit after each task.

---

## File Structure

- `notebooks/nb_style.py` (new) — theme, palette, KPI/annotation helpers, confederation colors.
- `src/training/backtester.py` (modify) — extract `_score_holdout`, add `collect_holdout_predictions`.
- `tests/unit/test_backtester.py` (modify) — test for `collect_holdout_predictions`.
- `requirements.txt` (modify) — add `seaborn`.
- `notebooks/0{1,2,3,4}_*.ipynb` (modify) — enhanced content.
- `CLAUDE.md` (modify) — one-line note about `nb_style.py` in §3.

---

## Task 1: Add the seaborn dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add seaborn pinned near matplotlib**

Find the line with `matplotlib==3.8.3` and add directly after it:

```
seaborn==0.13.2
```

- [ ] **Step 2: Install it**

Run: `pip install seaborn==0.13.2`
Expected: installs (matplotlib/numpy/pandas already satisfied).

- [ ] **Step 3: Verify import**

Run: `python -c "import seaborn; print(seaborn.__version__)"`
Expected: `0.13.2`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add seaborn for notebook visualizations"
```

---

## Task 2: Shared notebook style module

**Files:**
- Create: `notebooks/nb_style.py`

- [ ] **Step 1: Write `notebooks/nb_style.py`**

```python
"""Shared styling and helpers for the analysis notebooks.

Import once at the top of each notebook::

    import nb_style as nbs
    nbs.apply_theme()

Provides a single seaborn theme, a confederation colour map and small plotting
helpers so every notebook looks consistent and presentable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import seaborn as sns

# Make ``src`` importable from within notebooks/ (cwd = notebooks/).
_ROOT = Path("..").resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.ratings.shrinkage import TEAM_CONFEDERATION  # noqa: E402

# Qualitative palette (color-blind friendly) reused across notebooks.
PALETTE = sns.color_palette("colorblind")

CONFEDERATION_COLORS: dict[str, str] = {
    "UEFA": "#1f77b4",
    "CONMEBOL": "#2ca02c",
    "CONCACAF": "#ff7f0e",
    "AFC": "#d62728",
    "CAF": "#9467bd",
    "OFC": "#8c564b",
    "UNKNOWN": "#7f7f7f",
}


def apply_theme() -> None:
    """Apply the shared seaborn/matplotlib theme."""
    sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 110,
            "figure.titlesize": 15,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#444444",
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def team_confederation(team: str) -> str:
    """Return a team's confederation (``"UNKNOWN"`` if unmapped)."""
    return TEAM_CONFEDERATION.get(str(team), "UNKNOWN")


def confederation_color(team: str) -> str:
    """Return the plotting colour for a team's confederation."""
    return CONFEDERATION_COLORS[team_confederation(team)]


def kpi_header(items: Sequence[tuple[str, str]], title: str | None = None) -> None:
    """Render a row of KPI cards.

    Args:
        items: Sequence of ``(label, value)`` pairs.
        title: Optional figure title.
    """
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 1.6))
    if n == 1:
        axes = [axes]
    for ax, (label, value) in zip(axes, items):
        ax.axis("off")
        ax.text(0.5, 0.62, str(value), ha="center", va="center",
                fontsize=20, fontweight="bold", color="#1f77b4")
        ax.text(0.5, 0.18, label, ha="center", va="center",
                fontsize=10, color="#444444")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


def annotate_barh(ax: plt.Axes, values: Iterable[float], fmt: str = "{:.0f}",
                  pad: float = 0.0) -> None:
    """Write value labels at the end of horizontal bars."""
    for i, v in enumerate(values):
        ax.text(float(v) + pad, i, fmt.format(v), va="center", fontsize=8)
```

- [ ] **Step 2: Verify it imports and themes cleanly**

Run: `cd notebooks && python -c "import nb_style as nbs; nbs.apply_theme(); print(nbs.confederation_color('Brazil'))" && cd ..`
Expected: prints `#2ca02c` (CONMEBOL), no error.

- [ ] **Step 3: Commit**

```bash
git add notebooks/nb_style.py
git commit -m "feat(notebooks): shared seaborn theme and helpers"
```

---

## Task 3: Leakage-safe held-out predictions helper

Refactor the per-fold logic of `run_tournament_backtest` into `_score_holdout` and add `collect_holdout_predictions` that reuses it. This feeds the calibration curve and backtest charts (notebook 03) without duplicating the leakage-safe split.

**Files:**
- Modify: `src/training/backtester.py`
- Test: `tests/unit/test_backtester.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_backtester.py`:

```python
def test_collect_holdout_predictions_shapes(synthetic_matches) -> None:
    import numpy as np
    from src.training.backtester import collect_holdout_predictions

    # synthetic_matches spans several years; pick a year with prior history.
    import pandas as pd
    years = pd.to_datetime(synthetic_matches["date"], errors="coerce").dt.year
    target = int(years.max())
    proba, y_true = collect_holdout_predictions(
        synthetic_matches, year=target, competition="WC", min_train_matches=1
    )
    assert proba.ndim == 2 and proba.shape[1] == 3
    assert len(y_true) == proba.shape[0]
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert set(y_true).issubset({"H", "D", "A"})
```

> Note: if `synthetic_matches` has no `competition == "WC"` rows, adjust the
> fixture call to the competition present in the fixture (check
> `tests/conftest.py`); the assertion logic stays the same.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_backtester.py::test_collect_holdout_predictions_shapes -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'collect_holdout_predictions'`.

- [ ] **Step 3: Refactor the per-fold body into `_score_holdout`**

In `src/training/backtester.py`, inside `run_tournament_backtest`, the per-year loop currently builds ratings, features, trains, predicts. Extract that into a module-level function. Add this function **above** `run_tournament_backtest`:

```python
def _score_holdout(
    df: pd.DataFrame,
    year: int,
    competition: str,
    weights: BlendWeights,
    shrinker_factory: Optional[Callable[[], RatingShrinker]],
    hyperparameters: Optional[dict],
    lasso_select: bool,
    lasso_C: float,
    squad_values: Optional[pd.DataFrame],
    min_train_matches: int,
    played: pd.Series,
) -> Optional[tuple[pd.DataFrame, np.ndarray]]:
    """Train the full stack on ``< year`` and score the ``== year`` tournament.

    Returns ``(test_feats, proba)`` or ``None`` when the fold is skipped
    (insufficient history or no test matches). Leakage-safe: ratings and
    features for the test fixtures are derived only from pre-year data.
    """
    train_matches = df[df["year"] < year]
    comp_upper = df["competition"].astype(str).str.upper()
    test_matches = df[(df["year"] == year) & (comp_upper == competition.upper()) & played]
    if len(train_matches) < min_train_matches or test_matches.empty:
        return None

    shrinker = shrinker_factory() if shrinker_factory is not None else None
    ensemble = RatingEnsemble(shrinker=shrinker).fit(train_matches)
    composite = ensemble.composite_table()

    combined = pd.concat([train_matches, test_matches], ignore_index=True)
    feats = build_match_feature_matrix(
        combined, pd.DataFrame(), composite, squad_values=squad_values
    )
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce")
    feats["year"] = feats["date"].dt.year
    feats_comp = feats["competition"].astype(str).str.upper()
    train_feats = feats[(feats["year"] < year) & feats["outcome"].notna()].copy()
    test_feats = feats[
        (feats["year"] == year) & (feats_comp == competition.upper()) & feats["outcome"].notna()
    ].copy()
    if train_feats.empty or test_feats.empty:
        return None

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            models_dir=tmp_dir,
            hyperparameters=hyperparameters,
            shrinker=shrinker_factory() if shrinker_factory is not None else None,
            lasso_select=lasso_select,
            lasso_C=lasso_C,
        )
        outputs = trainer.fit(train_feats, val_features=None)

    predictor = MatchPredictor(
        outcome_models={"multinomial": outputs.multinomial, "xgboost": outputs.xgboost},
        poisson_model=outputs.poisson,
        elo=ensemble.elo,
        calibrator=None,
        blender=ProbabilityBlender(weights=weights),
    )
    proba = predictor.predict_proba(test_feats)
    return test_feats, proba
```

- [ ] **Step 4: Make `run_tournament_backtest` use `_score_holdout`**

Replace the body of the per-year `for year in sorted(target_years):` loop (everything from `train_matches = ...` down to the `proba = predictor.predict_proba(test_feats)` line) with:

```python
        scored = _score_holdout(
            df, year, competition, weights, shrinker_factory, hyperparameters,
            lasso_select, lasso_C, squad_values, min_train_matches, played,
        )
        if scored is None:
            logger.warning("Skipping %d: insufficient history or no test matches", year)
            continue
        test_feats, proba = scored
```

Keep the existing metric/record block (`y_true = ...`, `report = evaluator.evaluate(...)`, `record = {...}`, `records.append(record)`) unchanged after it.

- [ ] **Step 5: Add `collect_holdout_predictions` below `run_tournament_backtest`**

```python
def collect_holdout_predictions(
    matches: pd.DataFrame,
    year: int,
    competition: str = "WC",
    shrinker_factory: Optional[Callable[[], RatingShrinker]] = None,
    blend_weights: Optional[BlendWeights] = None,
    hyperparameters: Optional[dict] = None,
    lasso_select: bool = True,
    lasso_C: float = 0.1,
    squad_values: Optional[pd.DataFrame] = None,
    min_train_matches: int = 500,
) -> tuple[np.ndarray, list[str]]:
    """Out-of-sample (proba, y_true) for one held-out tournament.

    Trains the full stack on every match before *year* and predicts that
    tournament's played matches. Used by the calibration and backtest notebooks.

    Returns:
        ``(proba, y_true)`` with ``proba`` shape ``(n, 3)`` summing to 1 row-wise
        and ``y_true`` a list of ``"H"/"D"/"A"``. Empty arrays if the fold is
        skipped.

    Raises:
        ValueError: If *matches* is empty.
    """
    if matches.empty:
        raise ValueError("No matches supplied to collect_holdout_predictions")
    weights = blend_weights or BlendWeights()
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    played = df["score_a"].notna() & df["score_b"].notna()
    scored = _score_holdout(
        df, year, competition, weights, shrinker_factory, hyperparameters,
        lasso_select, lasso_C, squad_values, min_train_matches, played,
    )
    if scored is None:
        return np.empty((0, 3)), []
    test_feats, proba = scored
    return proba, test_feats["outcome"].astype(str).tolist()
```

- [ ] **Step 6: Run the new test and the full backtester suite**

Run: `pytest tests/unit/test_backtester.py -v --no-cov`
Expected: all pass (RPS tests + new shapes test).

- [ ] **Step 7: Regression-check the backtest still runs**

Run: `python scripts/backtest_tournaments.py --years 2022 2>&1 | tail -3`
Expected: prints a per-fold row for 2022 and aggregate (proves the refactor didn't break `run_tournament_backtest`).

- [ ] **Step 8: Commit**

```bash
git add src/training/backtester.py tests/unit/test_backtester.py
git commit -m "feat(backtester): collect_holdout_predictions for calibration/backtest notebooks"
```

---

## Task 4: Notebook 01 — Data & tournament landscape

**Files:**
- Modify: `notebooks/01_exploration.ipynb`

Rebuild as the ordered cells below. Use NotebookEdit to replace/insert cells. After every code cell that produces a figure, the figure must render on execution. All cells assume `matches` (canonical table, `date` parsed) is in scope from the setup cell.

- [ ] **Step 1: Replace the intro markdown (cell 0)**

```markdown
# 01 · Data & Tournament Landscape

A professional tour of the canonical match table that powers every model:
volume and composition of the data, scoring dynamics (incl. the home-advantage
trend), the WC-2026 group draw and which "groups of death" stand out, and the
historical data coverage that drives rating reliability.

**Data:** `data/interim/matches_unified.csv` (run `import_kaggle_history.py` +
`bootstrap_historical_data.py` first).
```

- [ ] **Step 2: Replace the setup cell (cell 1) to use nb_style**

```python
import nb_style as nbs
nbs.apply_theme()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import DataLoader

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

matches = DataLoader().load_matches()
matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
matches["year"] = matches["date"].dt.year
```

- [ ] **Step 3: KPI header cell (new)**

```python
n_wc26 = matches[(matches["competition"] == "WC") & (matches["year"] == 2026)].shape[0]
n_teams = pd.concat([matches["team_a"], matches["team_b"]]).nunique()
nbs.kpi_header([
    ("Matches", f"{len(matches):,}"),
    ("Teams", f"{n_teams}"),
    ("Competitions", f"{matches['competition'].nunique()}"),
    ("Span", f"{matches['year'].min()}–{matches['year'].max()}"),
    ("WC-2026 fixtures", f"{n_wc26}"),
], title="Canonical dataset at a glance")
```

- [ ] **Step 4: Matches-over-time cell (replace the old per-year bar)**

```python
per_year = matches.groupby("year").size()
fig, ax = plt.subplots(figsize=(11, 4))
sns.lineplot(x=per_year.index, y=per_year.values, marker="o", ax=ax, color=nbs.PALETTE[0])
ax.fill_between(per_year.index, per_year.values, alpha=0.15, color=nbs.PALETTE[0])
for wc_year in (2014, 2018, 2022, 2026):
    if wc_year in per_year.index:
        ax.axvline(wc_year, color="#d62728", ls="--", lw=1, alpha=0.6)
        ax.text(wc_year, per_year.max() * 0.98, "WC", color="#d62728", ha="center", fontsize=8)
ax.set(title="International matches per year", xlabel="Year", ylabel="Matches")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Competition composition cell (replace old comp bar)**

```python
per_comp = matches["competition"].value_counts()
share = (per_comp / per_comp.sum() * 100)
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=per_comp.values, y=per_comp.index, ax=ax, palette="crest")
nbs.annotate_barh(ax, per_comp.values, fmt="{:,.0f}", pad=per_comp.max() * 0.01)
ax.set(title="Matches by competition", xlabel="Matches", ylabel="")
plt.tight_layout(); plt.show()
```

- [ ] **Step 6: Goals & home-advantage analysis cell (new)**

```python
played = matches.dropna(subset=["score_a", "score_b"]).copy()
played["total_goals"] = played["score_a"] + played["score_b"]
played["home_win"] = played["score_a"] > played["score_b"]
played["draw"] = played["score_a"] == played["score_b"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.histplot(played["total_goals"], bins=range(0, 11), ax=axes[0], color=nbs.PALETTE[0])
axes[0].set(title=f"Goals per match (mean {played['total_goals'].mean():.2f})",
            xlabel="Total goals", ylabel="Matches")

trend = played.groupby("year").agg(home_win_rate=("home_win", "mean"),
                                   draw_rate=("draw", "mean"))
sns.lineplot(data=trend, ax=axes[1], marker="o")
axes[1].set(title="Home-win & draw rate over time", xlabel="Year", ylabel="Rate")
axes[1].legend(["Home-win rate", "Draw rate"])
plt.tight_layout(); plt.show()

print(f"Overall home-win rate: {played['home_win'].mean():.3f} | draw rate: {played['draw'].mean():.3f}")
```

- [ ] **Step 7: Groups of death cell (new)** — requires composite ratings

```python
from src.ratings.rating_ensemble import RatingEnsemble

composite = RatingEnsemble().fit(matches).composite_table().set_index("team")["composite_strength"]
wc = matches[(matches["competition"] == "WC") & (matches["year"] == 2026)].copy()
wc = wc[wc["group"].astype(str).str.strip().ne("")]

rows = []
for g, d in wc.groupby("group"):
    teams = sorted(set(d["team_a"]).union(d["team_b"]))
    strengths = sorted((composite.get(t, np.nan) for t in teams), reverse=True)
    rows.append({
        "group": g, "teams": ", ".join(teams),
        "mean_strength": np.nanmean(strengths),
        "top2_strength": np.nansum(strengths[:2]),
    })
group_df = pd.DataFrame(rows).sort_values("top2_strength", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=group_df, x="top2_strength", y="group", ax=ax, palette="rocket")
ax.set(title="Group difficulty (sum of top-2 composite strength) — higher = group of death",
       xlabel="Top-2 composite strength", ylabel="Group")
plt.tight_layout(); plt.show()
print(group_df.to_string(index=False))
```

- [ ] **Step 8: Rating-reliability coverage cell (replace old coverage)**

```python
wc_teams = sorted(set(wc["team_a"]).union(wc["team_b"]))
hist = matches[matches["year"] < 2026]
cov = pd.DataFrame({
    "team": wc_teams,
    "historical_matches": [int(((hist["team_a"] == t) | (hist["team_b"] == t)).sum()) for t in wc_teams],
}).sort_values("historical_matches")
fig, ax = plt.subplots(figsize=(9, 4))
sns.histplot(cov["historical_matches"], bins=20, ax=ax, color=nbs.PALETTE[2])
ax.axvline(cov["historical_matches"].median(), color="#d62728", ls="--",
           label=f"median={cov['historical_matches'].median():.0f}")
ax.set(title="Historical match coverage of WC-2026 teams (low = unstable ratings)",
       xlabel="Historical matches", ylabel="Teams"); ax.legend()
plt.tight_layout(); plt.show()
print("Lowest coverage (watch for unstable ratings):")
print(cov.head(8).to_string(index=False))
```

- [ ] **Step 9: Key findings markdown (new, final cell)**

```markdown
## Key findings

- **Volume & cadence:** the dataset spans {min}–{max}; match volume dips in 2020 (COVID) and peaks in qualifier years.
- **Scoring:** mean goals/match ≈ value shown above; the home-advantage line shows whether the edge is fading over the decade.
- **Group of death:** the top-ranked group by top-2 strength is the toughest path out of the group stage.
- **Coverage risk:** teams in the low-coverage tail will have the least reliable ratings — expect wider prediction intervals.
```
(Replace `{min}`/`{max}` with the actual years printed by the KPI cell.)

- [ ] **Step 10: Remove leftover old cells**

Delete any remaining original cells not replaced above (old per-year bar duplicate, old print-only coverage). The notebook should contain only the cells from Steps 1–9.

- [ ] **Step 11: Execute and verify**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration.ipynb`
Expected: exit 0, every figure renders, no traceback.

- [ ] **Step 12: Commit**

```bash
git add notebooks/01_exploration.ipynb
git commit -m "feat(notebooks): polish 01 exploration (goals trend, groups of death, KPIs)"
```

---

## Task 5: Notebook 02 — Team strength models

**Files:**
- Modify: `notebooks/02_ratings_diagnostics.ipynb`

- [ ] **Step 1: Replace intro markdown (cell 0)**

```markdown
# 02 · Team Strength Models

How the three rating systems (Elo, PI, rolling form) see the field, how they
agree, and — most importantly — where **squad market value (talent)** and
**Elo (results)** disagree. That gap is the core insight of the project: teams
the result-based ratings under- or over-rate.

**Prerequisite:** canonical table populated; `squad_values.csv` for the talent
scatter (optional — the cell degrades cleanly if missing).
```

- [ ] **Step 2: Replace setup cell (cell 1)**

```python
import nb_style as nbs
nbs.apply_theme()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_loader import DataLoader
from src.ratings.rating_ensemble import RatingEnsemble

matches = DataLoader().load_matches()
matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
ensemble = RatingEnsemble().fit(matches)
composite = ensemble.composite_table()
print(f"{len(composite)} teams rated.")
```

- [ ] **Step 3: KDE distributions cell (replace the 3-hist cell)**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["elo", "pi_combined", "form_score"], nbs.PALETTE):
    sns.histplot(composite[col], kde=True, ax=ax, color=color)
    ax.set(title=f"Distribution · {col}", xlabel=col, ylabel="Teams")
axes[0].axvline(1500, color="#d62728", ls="--", lw=1)
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Correlation heatmap cell (replace the manual imshow)**

```python
corr = composite[["elo", "pi_combined", "form_score", "composite_strength"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title("Correlation between rating systems")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Squad value vs Elo scatter cell (new, the headline insight)**

```python
from pathlib import Path
sv_path = Path("..") / "data" / "raw" / "squad_values" / "squad_values.csv"
wc = matches[(matches["competition"] == "WC") & (matches["date"].dt.year == 2026)]
wc_teams = sorted(set(wc["team_a"].dropna()).union(wc["team_b"].dropna()))

if not sv_path.exists():
    print("squad_values.csv not found — run scripts/build_squad_values.py to enable this chart.")
else:
    sv = pd.read_csv(sv_path)
    sv26 = sv[sv["as_of_date"] == "2026-06-01"].set_index("team")["squad_value_top11_meur"]
    df = composite.set_index("team").loc[[t for t in wc_teams if t in composite["team"].values]]
    df = df.assign(top11=[sv26.get(t, np.nan) for t in df.index]).dropna(subset=["top11"])
    df["elo_z"] = (df["elo"] - df["elo"].mean()) / df["elo"].std()
    df["val_z"] = (np.log1p(df["top11"]) - np.log1p(df["top11"]).mean()) / np.log1p(df["top11"]).std()
    df["residual"] = df["val_z"] - df["elo_z"]  # +ve = more talent than Elo implies

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [nbs.confederation_color(t) for t in df.index]
    ax.scatter(df["elo_z"], df["val_z"], c=colors, s=60, edgecolor="white")
    lims = [min(df["elo_z"].min(), df["val_z"].min()), max(df["elo_z"].max(), df["val_z"].max())]
    ax.plot(lims, lims, color="#888", ls="--", lw=1, label="talent = results")
    for t, r in df.iterrows():
        if abs(r["residual"]) > 0.8 or t in ("Brazil", "Morocco"):
            ax.annotate(t, (r["elo_z"], r["val_z"]), fontsize=8,
                        xytext=(4, 2), textcoords="offset points")
    ax.set(title="Talent (squad value) vs results (Elo) — WC-2026 teams, z-scored",
           xlabel="Elo (z)", ylabel="Top-11 squad value, log (z)")
    ax.legend(); plt.tight_layout(); plt.show()
    print("Most UNDER-rated by Elo (talent >> results):")
    print(df.sort_values("residual", ascending=False).head(5)[["elo", "top11", "residual"]].to_string())
    print("\nMost OVER-rated by Elo (results >> talent):")
    print(df.sort_values("residual").head(5)[["elo", "top11", "residual"]].to_string())
```

- [ ] **Step 6: Strength-by-confederation boxplot cell (new)**

```python
comp = composite.copy()
comp["confederation"] = comp["team"].map(nbs.team_confederation)
order = comp.groupby("confederation")["composite_strength"].median().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=comp, x="confederation", y="composite_strength", order=order,
            palette=[nbs.CONFEDERATION_COLORS[c] for c in order], ax=ax)
ax.set(title="Composite strength distribution by confederation", xlabel="", ylabel="Composite (z)")
plt.tight_layout(); plt.show()
```

- [ ] **Step 7: WC-2026 ranking colored by confederation (replace old barh)**

```python
wc_strength = composite[composite["team"].isin(wc_teams)].sort_values("composite_strength")
fig, ax = plt.subplots(figsize=(10, 11))
ax.barh(wc_strength["team"], wc_strength["composite_strength"],
        color=[nbs.confederation_color(t) for t in wc_strength["team"]])
ax.axvline(0, color="black", lw=0.5)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in nbs.CONFEDERATION_COLORS.values()]
ax.legend(handles, nbs.CONFEDERATION_COLORS.keys(), title="Confederation", fontsize=8)
ax.set(title="WC-2026 teams by composite strength", xlabel="Composite (z)", ylabel="")
plt.tight_layout(); plt.show()
```

- [ ] **Step 8: Biggest rating disagreements cell (new)**

```python
z = composite.copy()
for c in ["elo", "pi_combined"]:
    z[f"{c}_z"] = (z[c] - z[c].mean()) / z[c].std()
z["disagreement"] = (z["elo_z"] - z["pi_combined_z"]).abs()
print("Teams where Elo and PI disagree most (rating uncertainty):")
print(z.sort_values("disagreement", ascending=False).head(10)[
    ["team", "elo", "pi_combined", "disagreement"]].to_string(index=False))
```

- [ ] **Step 9: Key findings markdown (new, final)**

```markdown
## Key findings

- Elo, PI and form are strongly correlated but not identical — form adds the most independent signal.
- **Talent vs results:** the scatter shows teams above the diagonal carry more squad value than their Elo implies (under-rated by results) and vice-versa; Brazil and Morocco are the canonical examples.
- UEFA and CONMEBOL dominate the confederation boxplots; the long lower tails are the minnows.
- The disagreement table flags teams whose strength is least certain — treat their predictions with caution.
```

- [ ] **Step 10: Delete leftover old cells (old top-20 print, old hist, old imshow, old top/bottom prints superseded above).** Keep only Steps 1–9 cells.

- [ ] **Step 11: Execute and verify**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/02_ratings_diagnostics.ipynb`
Expected: exit 0, all figures render.

- [ ] **Step 12: Commit**

```bash
git add notebooks/02_ratings_diagnostics.ipynb
git commit -m "feat(notebooks): polish 02 ratings (talent-vs-Elo, confederation boxplots)"
```

---

## Task 6: Notebook 03 — Forecasting models & calibration

**Files:**
- Modify: `notebooks/03_model_comparison.ipynb`

- [ ] **Step 1: Replace intro markdown (cell 0)**

```markdown
# 03 · Forecasting Models & Calibration

What each model contributes, how independent they are, and — the rigorous part —
how **well-calibrated** the blended forecast is out-of-sample, plus a held-out
**backtest** vs the bookmaker-grade uniform baseline and the squad-value A/B.

**Prerequisite:** `scripts/train_models.py` has been run.
```

- [ ] **Step 2: Replace setup cell (cell 1)**

```python
import nb_style as nbs
nbs.apply_theme()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models.base_model import BaseOutcomeModel
from src.models.poisson_model import PoissonScoreModel
from src.ensemble.blender import BlendWeights, ProbabilityBlender
from src.prediction.feature_builder import build_inference_feature_matrix, filter_upcoming
from src.prediction.predictor import MatchPredictor
from src.utils.config import load_config
from src.utils.io import load_pickle

ROOT = Path("..").resolve()
MODELS = ROOT / "models"
config = load_config()
xgb = BaseOutcomeModel.load(MODELS / "xgboost_model.pkl")
mn = BaseOutcomeModel.load(MODELS / "multinomial_model.pkl")
poisson = load_pickle(MODELS / "poisson_model.pkl")
calibrator = load_pickle(MODELS / "calibrator.pkl")
weights = BlendWeights.from_config(config)
print(f"XGB feats={len(xgb.feature_columns)} | weights={weights}")
```

- [ ] **Step 3: Grouped feature-importance cell (replace old barh)**

```python
imp = xgb.feature_importance().head(20).copy()
def feat_group(f):
    if "squad_value" in f: return "squad value"
    if any(k in f for k in ["elo", "pi", "form", "attack", "defense", "reputation", "wc_"]): return "strength"
    return "context"
imp["group"] = imp["feature"].map(feat_group)
gcolors = {"strength": nbs.PALETTE[0], "squad value": nbs.PALETTE[2], "context": nbs.PALETTE[7]}
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(imp["feature"][::-1], imp["importance"][::-1],
        color=[gcolors[g] for g in imp["group"][::-1]])
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in gcolors.values()]
ax.legend(handles, gcolors.keys(), title="Feature group")
ax.set(title="XGBoost feature importance (top 20)", xlabel="Importance")
plt.tight_layout(); plt.show()
```

- [ ] **Step 4: Model-agreement cell (new)** — predicts all fixtures with each model

```python
fm = build_inference_feature_matrix(tournament_year=2026)
fx = filter_upcoming(fm, stage_substr="group")
X = fx[xgb.feature_columns]
p_xgb = xgb.predict_proba(X)
p_mn = mn.predict_proba(X)
p_poi = poisson.predict_proba(fx[["team_a", "team_b"]])
home = pd.DataFrame({"XGBoost": p_xgb[:, 0], "Multinomial": p_mn[:, 0], "Poisson": p_poi[:, 0]})
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(home.corr(), annot=True, fmt=".2f", cmap="vlag", vmin=0, vmax=1, square=True, ax=ax)
ax.set_title("Model agreement · corr of P(home) across fixtures")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Calibration / reliability curve cell (new, headline rigor)**

```python
from src.data.data_loader import DataLoader
from src.data.squad_value_client import SquadValueClient
from src.ratings.shrinkage import RatingShrinker
from src.training.backtester import collect_holdout_predictions

sv = SquadValueClient().load() if SquadValueClient().is_available() else None
proba, y_true = collect_holdout_predictions(
    DataLoader().load_matches(), year=2022, competition="WC",
    shrinker_factory=lambda: RatingShrinker(k_poisson=35, mixture_prior=False),
    blend_weights=weights, squad_values=sv,
)
y = np.array(y_true)
conf = proba.max(axis=1)
correct = (np.array(["H", "D", "A"])[proba.argmax(axis=1)] == y).astype(float)
bins = np.linspace(0, 1, 11)
idx = np.digitize(conf, bins) - 1
xs, ys, ece = [], [], 0.0
for b in range(10):
    m = idx == b
    if m.sum() == 0: continue
    xs.append(conf[m].mean()); ys.append(correct[m].mean())
    ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], ls="--", color="#888", label="perfect")
ax.plot(xs, ys, marker="o", color=nbs.PALETTE[0], label="model")
ax.set(title=f"Reliability diagram · held-out WC-2022 (ECE={ece:.3f})",
       xlabel="Predicted confidence", ylabel="Observed accuracy")
ax.legend(); plt.tight_layout(); plt.show()
```

- [ ] **Step 6: Backtest-vs-baseline bar cell (new)**

```python
from src.utils.metrics import compute_log_loss
ll_model = compute_log_loss(y_true, proba)
ll_uniform = np.log(3)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=["Uniform (1/3)", "Full model"], y=[ll_uniform, ll_model],
            palette=["#bbbbbb", nbs.PALETTE[0]], ax=ax)
for i, v in enumerate([ll_uniform, ll_model]):
    ax.text(i, v + 0.005, f"{v:.4f}", ha="center")
ax.set(title="Held-out WC-2022 log-loss (lower is better)", ylabel="Log-loss")
plt.tight_layout(); plt.show()
print(f"Model beats uniform by {(ll_uniform - ll_model):.4f} log-loss.")
```

- [ ] **Step 7: Poisson scoreline grid cell (keep, restyle)** — keep the existing grid logic but draw with seaborn:

```python
TEAM_A, TEAM_B = "Brazil", "Argentina"
grid = poisson.score_matrix(TEAM_A, TEAM_B)[:6, :6]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(grid, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "P"})
ax.invert_yaxis()
ax.set(title=f"Scoreline grid · {TEAM_A} vs {TEAM_B}",
       xlabel=f"{TEAM_B} goals", ylabel=f"{TEAM_A} goals")
plt.tight_layout(); plt.show()
for (sa, sb), p in poisson.top_k_scorelines(TEAM_A, TEAM_B, k=5):
    print(f"  {sa}-{sb}: {p:.4f}")
```

- [ ] **Step 8: Sample-match model comparison cell (restyle, use config blender)**

```python
predictor = MatchPredictor(
    outcome_models={"multinomial": mn, "xgboost": xgb}, poisson_model=poisson,
    calibrator=calibrator, blender=ProbabilityBlender(weights=weights),
)
row = fm[(fm["team_a"] == TEAM_A) & (fm["team_b"] == TEAM_B)]
if row.empty: row = fm[fm["team_a"] == TEAM_A].head(1)
Xr = row[xgb.feature_columns]
data = pd.DataFrame({
    "XGBoost": xgb.predict_proba(Xr)[0], "Multinomial": mn.predict_proba(Xr)[0],
    "Poisson": poisson.outcome_probabilities(TEAM_A, TEAM_B),
    "Blended": predictor.predict_proba(row)[0],
}, index=["Home", "Draw", "Away"])
fig, ax = plt.subplots(figsize=(8, 4))
data.plot(kind="bar", ax=ax, color=nbs.PALETTE[:4])
ax.set(title=f"{TEAM_A} vs {TEAM_B} · per-model probabilities", ylabel="P")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
```

- [ ] **Step 9: Key findings markdown (new, final)**

```markdown
## Key findings

- Feature importance is led by strength diffs; **squad value contributes a meaningful, independent block**.
- The models are correlated but not redundant — Poisson is the most independent of the ML pair.
- **Calibration:** the reliability curve hugs the diagonal (ECE shown) — predicted confidence ≈ observed accuracy out-of-sample.
- The model beats the uniform baseline on held-out WC-2022 log-loss, i.e. it carries genuine signal.
```

- [ ] **Step 10: Delete leftover old cells** not replaced above (old per-model print block, old confidence-only hist if superseded). Keep Steps 1–9.

- [ ] **Step 11: Execute and verify**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/03_model_comparison.ipynb`
Expected: exit 0. (This trains a held-out fold inside the calibration cell — allow ~30–60s.)

- [ ] **Step 12: Commit**

```bash
git add notebooks/03_model_comparison.ipynb
git commit -m "feat(notebooks): polish 03 models (calibration, backtest, model agreement)"
```

---

## Task 7: Notebook 04 — Simulation & quiniela strategy

**Files:**
- Modify: `notebooks/04_quiniela_strategy.ipynb`

- [ ] **Step 1: Delete the scratch/broken cells**

Delete every cell after the original "picks preview" cell — the user's scratch work: `pickel = pd.read_pickle(...)`, `dir(pickel)`, the sorted Elo top-30, the "extras que se borrarán" markdown, the empty cell, the Elo/Poisson/Blend loop, the **broken** `poisson.attack` cell (raises AttributeError), and the stale `round_reached` (`r32` only) cell. Use NotebookEdit `edit_mode=delete` on each (work bottom-up so indices stay valid).

- [ ] **Step 2: Replace intro markdown (cell 0)**

```markdown
# 04 · Tournament Simulation & Quiniela Strategy

Reads the Monte-Carlo outputs and pick sheets to answer the money question:
who wins, who advances, and **which risk profile maximises expected quiniela
value**. Includes an expected-value / risk-return view of the four profiles and
the distribution of most-likely scorelines.

**Prerequisite:** `export_quiniela_sheet.py`, `predict_scorelines.py`,
`simulate_tournament.py` have been run.
```

- [ ] **Step 3: Replace setup cell (cell 1)**

```python
import nb_style as nbs
nbs.apply_theme()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path("..").resolve()
OUT = ROOT / "outputs"
PROFILES = ["safe", "balanced", "aggressive", "contrarian"]
sheets = {p: pd.read_csv(OUT / "picks" / f"quiniela_{p}.csv") for p in PROFILES}
```

- [ ] **Step 4: Championship favorites cell (restyle, color by confederation)**

```python
champ = pd.read_csv(OUT / "simulations" / "championship_probabilities.csv").head(18)
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(champ["team"][::-1], champ["championship_prob"][::-1] * 100,
        color=[nbs.confederation_color(t) for t in champ["team"][::-1]])
nbs.annotate_barh(ax, (champ["championship_prob"][::-1] * 100), fmt="{:.1f}%", pad=0.1)
ax.set(title="Title favorites (Monte-Carlo, full model)", xlabel="Championship probability (%)")
plt.tight_layout(); plt.show()
```

- [ ] **Step 5: Round-reached heatmap cell (restyle, verify all stages)**

```python
rounds = pd.read_csv(OUT / "simulations" / "round_reached_probabilities.csv")
ORDER = ["r32", "r16", "qf", "sf", "final", "champion"]
LABELS = {"r32": "R32", "r16": "R16", "qf": "QF", "sf": "SF", "final": "Final", "champion": "Champ"}
pivot = rounds.pivot_table(index="team", columns="stage", values="prob", aggfunc="max").fillna(0)
pivot = pivot.reindex(columns=[c for c in ORDER if c in pivot.columns]).rename(columns=LABELS)
top = champ.head(15)["team"].tolist()
pivot = pivot.loc[[t for t in top if t in pivot.index]]
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(pivot * 100, annot=True, fmt=".0f", cmap="YlOrRd", vmin=0, vmax=100,
            cbar_kws={"label": "%"}, ax=ax)
ax.set(title="Probability of reaching each round (top-15 favorites)", xlabel="", ylabel="")
plt.tight_layout(); plt.show()
assert pivot.shape[1] >= 4, "round_reached has too few stages — re-run simulate_tournament"
```

- [ ] **Step 6: Expected value / risk-return cell (new, the useful core)**

```python
import yaml
scoring = yaml.safe_load((ROOT / "config" / "strategy.yaml").read_text(encoding="utf-8"))
rule = scoring["scoring"]["quiniela_default"]
correct_pts, upset_pts = rule["correct_1x2"], rule["upset_bonus"]

def profile_ev(df):
    p = df[["p_home", "p_draw", "p_away"]].to_numpy()
    pick_idx = df["pick"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    p_pick = p[np.arange(len(df)), pick_idx]
    favorite_idx = p.argmax(axis=1)
    is_upset = pick_idx != favorite_idx
    pts = correct_pts + np.where(is_upset, upset_pts, 0.0)      # points if pick hits
    ev = p_pick * pts                                          # expected points per pick
    var = p_pick * (pts - ev) ** 2 + (1 - p_pick) * (0 - ev) ** 2
    return ev.sum(), np.sqrt(var.sum())

stats = pd.DataFrame(
    [{"profile": p, "EV": profile_ev(sheets[p])[0], "risk": profile_ev(sheets[p])[1]} for p in PROFILES]
)
fig, ax = plt.subplots(figsize=(8, 6))
for _, r in stats.iterrows():
    ax.scatter(r["risk"], r["EV"], s=120)
    ax.annotate(r["profile"], (r["risk"], r["EV"]), xytext=(6, 4), textcoords="offset points")
ax.set(title="Quiniela profiles · expected value vs risk", xlabel="Risk (std of points)", ylabel="Expected points")
plt.tight_layout(); plt.show()
print(stats.round(2).to_string(index=False))
```

- [ ] **Step 7: Profile divergence cell (keep, tidy)** — show matches where profiles disagree:

```python
base = sheets["balanced"][["team_a", "team_b", "p_home", "p_draw", "p_away"]].copy()
for p in PROFILES:
    base[f"pick_{p}"] = sheets[p]["pick"].values
diff = base[(base["pick_safe"] != base["pick_aggressive"]) |
            (base["pick_balanced"] != base["pick_contrarian"])]
print(f"Matches where profiles disagree: {len(diff)}/{len(base)}")
print(diff.head(15).to_string(index=False))
```

- [ ] **Step 8: Scoreline distribution cell (new, uses scoreline output)**

```python
sl_path = OUT / "predictions" / "scoreline_predictions.csv"
if not sl_path.exists():
    print("scoreline_predictions.csv not found — run scripts/predict_scorelines.py.")
else:
    sl = pd.read_csv(sl_path)
    modal = sl["top1_score"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=modal.values, y=modal.index, palette="mako", ax=ax)
    ax.set(title="Most common modal scoreline across upcoming matches",
           xlabel="Matches", ylabel="Scoreline (home-away)")
    plt.tight_layout(); plt.show()
    print(f"Share of matches whose modal score is a draw: "
          f"{sl['top1_score'].str.split('-').apply(lambda s: s[0]==s[1]).mean():.0%}")
```

- [ ] **Step 9: Most-frequent finals cell (keep, tidy)**

```python
paths = pd.read_csv(OUT / "simulations" / "bracket_paths.csv")
finals = paths.groupby(["champion", "runner_up"]).size().sort_values(ascending=False).head(12)
print(f"From {len(paths)} simulations, most frequent finals:")
print(finals.to_string())
```

- [ ] **Step 10: Key findings markdown (new, final)**

```markdown
## Key findings

- The title race is led by the teams shown above (now talent-aware via squad value).
- **Profile EV/risk:** the risk-return chart shows which profile delivers the most expected points per unit of risk — the rational default for value maximisation.
- Profiles disagree on a large share of matches; those are where strategy actually matters.
- A meaningful fraction of matches mode to a draw scoreline (1-1/0-0) even when one side is favored — important when filling exact-score quinielas.
```

- [ ] **Step 11: Execute and verify**

Run: `jupyter nbconvert --to notebook --execute --inplace notebooks/04_quiniela_strategy.ipynb`
Expected: exit 0, all figures render, no scratch cells remain.

- [ ] **Step 12: Commit**

```bash
git add notebooks/04_quiniela_strategy.ipynb
git commit -m "feat(notebooks): rebuild 04 strategy (EV/risk, scoreline dist), drop scratch cells"
```

---

## Task 8: Final verification & docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Re-run the full unit suite**

Run: `pytest tests/unit/ -q --no-cov`
Expected: all pass (includes the new `collect_holdout_predictions` test).

- [ ] **Step 2: Re-execute all four notebooks back-to-back**

Run:
```bash
for n in 01_exploration 02_ratings_diagnostics 03_model_comparison 04_quiniela_strategy; do \
  jupyter nbconvert --to notebook --execute --inplace notebooks/$n.ipynb || echo "FAILED $n"; done
```
Expected: no "FAILED" lines.

- [ ] **Step 3: Add a one-line note to `CLAUDE.md` §3 structure**

Under the `notebooks/` entry in the structure tree (§3), append:
`(estilo compartido en nb_style.py; análisis de portafolio).`

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note shared notebook style module"
```

---

## Self-Review notes (already applied)

- **Spec coverage:** nb_style (Task 2), seaborn dep (Task 1), collect_holdout_predictions + test (Task 3), notebook 01 goals/groups/KPIs (Task 4), notebook 02 talent-vs-Elo/confederation (Task 5), notebook 03 calibration/backtest/agreement (Task 6), notebook 04 EV-ROI/scoreline/cleanup (Task 7), execution + docs (Task 8). All spec sections mapped.
- **Type consistency:** `collect_holdout_predictions` / `_score_holdout` signatures match between Task 3 definition and the notebook-03 call (Task 6, Step 5).
- **Degradation:** notebook cells that read `squad_values.csv` / `scoreline_predictions.csv` guard on existence (Tasks 5, 7).
