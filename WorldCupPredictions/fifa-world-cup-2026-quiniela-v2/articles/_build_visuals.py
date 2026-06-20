"""Generate all article/LinkedIn visuals: figures, equation images, hero banners,
and LinkedIn carousel slides. Self-contained PNGs so Medium/LinkedIn render correctly
(Medium does not render LaTeX or Mermaid). Run from the articles/ folder.

    python _build_visuals.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from scipy.stats import poisson, skellam

    HAVE_SCIPY = True
except Exception:  # pragma: no cover
    HAVE_SCIPY = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
EQ = os.path.join(HERE, "equations")
CAR = os.path.join(HERE, "carousels")
for d in (FIG, EQ, CAR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------- palette / style
NAVY = "#0E1B2E"
NAVY2 = "#16263B"
PANEL = "#1E3A57"
GRIDC = "#27415C"
GREEN = "#19C98C"
GOLD = "#F4C04E"
RED = "#FF6B6B"
BLUE = "#5BA8FF"
PURPLE = "#B58CFF"
TEXT = "#EAF2F9"
MUTED = "#90A6B8"

plt.rcParams.update(
    {
        "figure.facecolor": NAVY,
        "axes.facecolor": NAVY,
        "savefig.facecolor": NAVY,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRIDC,
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
    }
)

FIG_W, FIG_H, DPI = 9.2, 5.2, 150


def _accent_title(fig, title, sub=None):
    fig.text(0.06, 0.975, title, fontsize=17, fontweight="bold", color=TEXT, va="top")
    fig.add_artist(plt.Line2D([0.06, 0.17], [0.902, 0.902], color=GREEN, lw=3.2))
    if sub:
        fig.text(0.06, 0.872, sub, fontsize=11.5, color=MUTED, va="top")


def _brand(fig):
    fig.text(0.985, 0.012, "World Cup 2026 Quiniela Predictor V2", fontsize=8.5,
             color=MUTED, ha="right", va="bottom", alpha=0.8)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print("fig  ->", name)


# ============================================================ EQUATION IMAGES
def render_eq(name, lines, fontsize=26):
    """Render one or more mathtext lines stacked, on a white card."""
    if isinstance(lines, str):
        lines = [lines]
    n = len(lines)
    # Height must grow with the line count: positions are figure fractions, so a
    # fixed tiny canvas collapses multi-line stacks onto each other (the lines
    # overlap). One row per line at ~0.62 in keeps the absolute spacing legible.
    row = 0.62
    fig = plt.figure(figsize=(0.1, row * n), facecolor="white")
    ys = [(n - i - 0.5) / n for i in range(n)]
    for y, ln in zip(ys, lines):
        fig.text(0.5, y, ln, fontsize=fontsize, ha="center", va="center",
                 color="#10243A", math_fontfamily="dejavusans")
    path = os.path.join(EQ, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.22,
                facecolor="white")
    plt.close(fig)
    print("eq   ->", name)


def build_equations():
    render_eq("poisson_pmf.png", r"$P(X=k)=\frac{e^{-\lambda}\,\lambda^{k}}{k!}$")
    render_eq("lambda_rate.png", r"$\lambda_{ab}=\mu\cdot\alpha_a\cdot\beta_b\cdot h$")
    render_eq("dixon_coles.png", [
        r"$(0,0)\rightarrow 1-\lambda_a\lambda_b\rho \qquad (0,1)\rightarrow 1+\lambda_a\rho$",
        r"$(1,0)\rightarrow 1+\lambda_b\rho \qquad\quad (1,1)\rightarrow 1-\rho$",
    ], fontsize=22)
    render_eq("grid_sums.png",
              r"$P(\mathrm{H})=\sum_{i>j} M_{ij}\quad\ "
              r"P(\mathrm{D})=\sum_{i} M_{ii}\quad\ "
              r"P(\mathrm{A})=\sum_{i<j} M_{ij}$", fontsize=22)
    render_eq("skellam_pmf.png",
              r"$P(D=k)=e^{-(\lambda_a+\lambda_b)}"
              r"\left(\frac{\lambda_a}{\lambda_b}\right)^{k/2}"
              r"I_{|k|}\!\left(2\sqrt{\lambda_a\lambda_b}\right)$", fontsize=24)
    render_eq("skellam_outcomes.png",
              r"$P(\mathrm{H})=P(D>0)=1-F_D(0)\quad\ "
              r"P(\mathrm{D})=p_D(0)\quad\ P(\mathrm{A})=F_D(-1)$", fontsize=21)
    render_eq("elo.png",
              r"$E_a=\frac{1}{1+10^{-(r_a-r_b)/400}}"
              r"\qquad r_a\leftarrow r_a+K\,(S_a-E_a)$", fontsize=23)
    render_eq("composite.png",
              r"$\mathrm{composite}(a)="
              r"\frac{w_e\,z_e(a)+w_p\,z_p(a)+w_f\,z_f(a)}{w_e+w_p+w_f}$", fontsize=23)
    render_eq("shrinkage.png",
              r"$r^{*}=\frac{n}{n+K}\,\hat{r}+\frac{K}{n+K}\,r_0$", fontsize=26)
    render_eq("mc_estimator.png",
              r"$\hat{P}(X=\mathrm{champion})="
              r"\frac{1}{N}\sum_{r=1}^{N}\mathbf{1}\,[\,X\ \mathrm{wins\ run}\ r\,]$",
              fontsize=23)
    render_eq("logloss.png",
              r"$\mathrm{LogLoss}=-\sum_i \log\,\hat{p}_{\,i,\,y_i}$", fontsize=25)
    render_eq("complexity.png",
              r"$O\!\left(N\cdot m\cdot \mathrm{cost}\right)\ \longrightarrow\ "
              r"O\!\left(n^{2}\cdot \mathrm{cost}+N\cdot m\right)$", fontsize=23)
    render_eq("argmax_utility.png",
              r"$\mathrm{pick}^{\star}=\mathrm{argmax}_{a}\ "
              r"E_{P}\!\left[\,U(a,\,\mathrm{outcome})\,\right]$", fontsize=24)


# ============================================================ ARTICLE FIGURES
def fig1_skellam_vs_grid():
    la, lb = 1.6, 1.05
    kmax = 6
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.07, right=0.97, wspace=0.32)

    if HAVE_SCIPY:
        pa = poisson.pmf(np.arange(kmax + 1), la)
        pb = poisson.pmf(np.arange(kmax + 1), lb)
    else:
        def pois(k, l):
            from math import exp, factorial
            return exp(-l) * l ** k / factorial(k)
        pa = np.array([pois(k, la) for k in range(kmax + 1)])
        pb = np.array([pois(k, lb) for k in range(kmax + 1)])
    M = np.outer(pa, pb)  # M[i,j] = P(a=i, b=j)

    im = axL.imshow(M, origin="lower", cmap="mako" if "mako" in plt.colormaps() else "viridis")
    axL.set_xticks(range(kmax + 1)); axL.set_yticks(range(kmax + 1))
    axL.set_xlabel("Team B goals (j)"); axL.set_ylabel("Team A goals (i)")
    axL.set_title("Score grid:  M[i,j] = P(A=i, B=j)", color=TEXT, fontsize=12.5, pad=8)
    # outline triangles
    for i in range(kmax + 1):
        for j in range(kmax + 1):
            if i > j:
                axL.add_patch(mpatches.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                              edgecolor=GREEN, lw=0.8, alpha=.55))
            elif i == j:
                axL.add_patch(mpatches.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                              edgecolor=GOLD, lw=1.1))
    axL.text(0.5, 5.3, "i > j  → Home win", color=GREEN, fontsize=10, fontweight="bold")
    axL.text(2.7, 1.0, "i < j → Away win", color=BLUE, fontsize=10, fontweight="bold")
    axL.text(3.2, 3.2, "draws", color=GOLD, fontsize=9.5, rotation=45)

    # Skellam pmf of difference
    ks = np.arange(-5, 6)
    if HAVE_SCIPY:
        pmf = skellam.pmf(ks, la, lb)
    else:
        pmf = np.array([sum(M[i, j] for i in range(kmax + 1) for j in range(kmax + 1)
                            if i - j == k) for k in ks])
    colors = [GREEN if k > 0 else (GOLD if k == 0 else BLUE) for k in ks]
    axR.bar(ks, pmf, color=colors, edgecolor=NAVY, width=0.8)
    axR.set_xlabel("Goal difference  D = G_A - G_B")
    axR.set_ylabel("P(D = k)")
    axR.set_title("Skellam: one closed form for H / D / A", color=TEXT, fontsize=12.5, pad=8)
    axR.set_xticks(ks)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    pH, pD, pA = pmf[ks > 0].sum(), pmf[ks == 0].sum(), pmf[ks < 0].sum()
    axR.text(0.02, 0.95, f"P(H)={pH:.2f}  P(D)={pD:.2f}  P(A)={pA:.2f}",
             transform=axR.transAxes, color=MUTED, fontsize=10, va="top")

    _accent_title(fig, "Don't integrate a grid — use Skellam",
                  "The same two Poisson rates give H/D/A directly from the goal-difference distribution")
    _brand(fig)
    save(fig, "fig01_skellam_vs_grid.png")


def fig2_time_decay():
    days = np.linspace(0, 1500, 400)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.10, right=0.96)
    for xi, c, lab, hl in [(0.0015, GREEN, "Elo / PI  (xi = 0.0015)", 462),
                           (0.0020, GOLD, "Poisson  (xi = 0.0020)", 347)]:
        w = np.exp(-xi * days)
        ax.plot(days, w, color=c, lw=3, label=lab)
        ax.axvline(hl, color=c, ls=":", lw=1.4, alpha=.8)
        ax.scatter([hl], [0.5], color=c, zorder=5, s=45, edgecolor=NAVY)
        ax.annotate(f"half-life ~{hl}d", (hl, 0.5), (hl + 40, 0.62 if hl > 400 else 0.74),
                    color=c, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.8, alpha=.5)
    ax.set_xlabel("Days before the most recent match")
    ax.set_ylabel("Weight  w = exp(-xi · days)")
    ax.set_ylim(0, 1.02); ax.set_xlim(0, 1500)
    leg = ax.legend(facecolor=NAVY2, edgecolor=GRIDC, fontsize=10.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TEXT)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _accent_title(fig, "Recent matches count more (Dixon-Coles 1997)",
                  "Exponential time decay applied to Elo, PI and Poisson updates")
    _brand(fig)
    save(fig, "fig02_time_decay.png")


def fig3_shrinkage():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 5.0))
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.08, right=0.97, wspace=0.28)
    confs = ["UEFA /\nCONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"]
    priors = [1620, 1430, 1420, 1400, 1200]
    cols = [GREEN, BLUE, GOLD, PURPLE, RED]
    axL.bar(confs, priors, color=cols, edgecolor=NAVY)
    axL.set_ylim(1100, 1700)
    axL.set_ylabel("Elo prior  r_0")
    axL.set_title("Confederation shrinkage priors", color=TEXT, fontsize=12.5, pad=8)
    for x, v in zip(range(5), priors):
        axL.text(x, v + 8, str(v), ha="center", color=TEXT, fontsize=10, fontweight="bold")
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)
    axL.tick_params(axis="x", labelsize=9)

    n = np.linspace(0, 80, 300)
    for K, c in [(30, GREEN), (35, GOLD)]:
        axR.plot(n, n / (n + K), color=c, lw=3, label=f"K = {K}")
    axR.set_xlabel("Matches observed  (n)")
    axR.set_ylabel("Weight on empirical rating  n/(n+K)")
    axR.set_ylim(0, 1.02)
    axR.set_title("Small samples are pulled to the prior", color=TEXT, fontsize=12.5, pad=8)
    leg = axR.legend(facecolor=NAVY2, edgecolor=GRIDC, fontsize=10.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(TEXT)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    _accent_title(fig, "James-Stein shrinkage tames minnows",
                  "Few matches → rating pulled toward the confederation mean")
    _brand(fig)
    save(fig, "fig03_shrinkage.png")


def fig4_squad_value():
    metrics = ["Log-loss", "Brier", "Ordinal RPS"]
    before = [0.9994, 0.1976, 0.2117]
    after = [0.9905, 0.1955, 0.2082]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.10, right=0.96)
    ax.bar(x - w / 2, before, w, color=MUTED, label="Without squad value")
    ax.bar(x + w / 2, after, w, color=GREEN, label="With squad value (A.2)")
    for xi, b, a in zip(x, before, after):
        ax.text(xi - w / 2, b + .004, f"{b:.4f}", ha="center", color=TEXT, fontsize=9)
        ax.text(xi + w / 2, a + .004, f"{a:.4f}", ha="center", color=GREEN, fontsize=9,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Held-out score (lower = better)")
    leg = ax.legend(facecolor=NAVY2, edgecolor=GRIDC, fontsize=10.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(TEXT)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _accent_title(fig, "Squad value: ACCEPTED by the backtest",
                  "Leakage-free A/B on WC 2018 + 2022 — every proper score improves")
    _brand(fig)
    save(fig, "fig04_squad_value.png")


def fig5_mixture_verdict():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.9),
                                   gridspec_kw={"width_ratios": [0.85, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.02, right=0.96, wspace=0.18)
    axL.axis("off")
    axL.text(0.5, 0.78, "Champion list", ha="center", color=MUTED, fontsize=12)
    axL.text(0.5, 0.60, "LOOKS better", ha="center", color=GOLD, fontsize=18,
             fontweight="bold")
    axL.text(0.5, 0.45, "Spain → #1  ✓", ha="center", color=TEXT, fontsize=13)
    axL.text(0.5, 0.22, "(face validity ≠ evidence)", ha="center", color=MUTED,
             fontsize=10.5, style="italic")
    axL.add_patch(mpatches.FancyArrowPatch((0.86, 0.5), (1.02, 0.5),
                  transform=axL.transAxes, arrowstyle="-|>", mutation_scale=22,
                  color=RED, lw=2))

    bars = ["Log-loss", "Accuracy"]
    deltas = [2.3, -2.4]
    cols = [RED, RED]
    axR.bar(bars, deltas, color=cols, edgecolor=NAVY, width=0.5)
    axR.axhline(0, color=MUTED, lw=1)
    axR.set_ylabel("Change vs baseline  (%, pts)")
    axR.set_title("...but the metric got WORSE", color=TEXT, fontsize=12.5, pad=8)
    axR.text(0, 2.5, "+2.3%\nlog-loss", ha="center", color=RED, fontsize=10.5,
             fontweight="bold")
    axR.text(1, -3.0, "-2.4 pts\naccuracy", ha="center", color=RED, fontsize=10.5,
             fontweight="bold", va="top")
    axR.set_ylim(-4, 4)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    _accent_title(fig, "Mixture prior (A.4): REJECTED",
                  "It improved the thing that doesn't matter and degraded the one that does")
    _brand(fig)
    save(fig, "fig05_mixture_verdict.png")


def fig6_leakage_free():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H - 0.4))
    fig.subplots_adjust(top=0.74, bottom=0.10, left=0.05, right=0.97)
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    # timeline
    ax.annotate("", xy=(9.6, 1.2), xytext=(0.4, 1.2),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    ax.text(9.6, 0.7, "time", color=MUTED, fontsize=10, ha="right")
    # train block
    ax.add_patch(mpatches.FancyBboxPatch((0.6, 1.6), 6.0, 1.5,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=PANEL, ec=GREEN, lw=2))
    ax.text(3.6, 2.7, "TRAIN", color=GREEN, fontsize=14, fontweight="bold", ha="center")
    ax.text(3.6, 2.15, "all matches with  year < Y", color=TEXT, fontsize=11, ha="center")
    ax.text(3.6, 1.78, "ratings • LASSO • XGB • multinomial • Poisson", color=MUTED,
            fontsize=9, ha="center")
    # test block
    ax.add_patch(mpatches.FancyBboxPatch((7.0, 1.6), 2.4, 1.5,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=PANEL, ec=GOLD, lw=2))
    ax.text(8.2, 2.7, "TEST", color=GOLD, fontsize=14, fontweight="bold", ha="center")
    ax.text(8.2, 2.15, "tournament == Y", color=TEXT, fontsize=10.5, ha="center")
    ax.text(8.2, 1.78, "(held out)", color=MUTED, fontsize=9, ha="center")
    # as-of arrow
    ax.annotate("", xy=(7.05, 3.45), xytext=(6.55, 3.45),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.text(5.0, 3.85, "as-of join: test features use ONLY pre-Y data  →  no leakage",
            color=BLUE, fontsize=10.5, ha="center", fontweight="bold")
    ax.text(8.2, 4.55, "metrics: log-loss · Brier · RPS · ECE", color=TEXT, fontsize=10,
            ha="center")
    _accent_title(fig, "The arbiter: leakage-free cross-tournament backtest",
                  "Every modelling change must improve held-out log-loss here")
    _brand(fig)
    save(fig, "fig06_leakage_free.png")


def _layer_box(ax, x, y, w, h, label, color, sub=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10", fc=PANEL, ec=color, lw=2))
    ax.text(x + w / 2, y + h / 2 + (0.06 if sub else 0), label, ha="center",
            va="center", color=TEXT, fontsize=11, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.18, sub, ha="center", va="center",
                color=MUTED, fontsize=8.2)


def fig7_layers():
    layers = [
        ("utils", MUTED), ("data", BLUE), ("ratings", GREEN), ("features", GREEN),
        ("models", GOLD), ("ensemble", GOLD),
        ("simulation / prediction / training", PURPLE), ("api", RED),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H + 0.4))
    fig.subplots_adjust(top=0.80, bottom=0.05, left=0.05, right=0.97)
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(layers) + 0.5)
    for i, (name, c) in enumerate(layers):
        y = len(layers) - 1 - i
        _layer_box(ax, 2.2, y + 0.12, 5.6, 0.74, name, c)
        if i < len(layers) - 1:
            ax.annotate("", xy=(5.0, y + 0.12), xytext=(5.0, y + 0.86),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    ax.text(8.2, len(layers) / 2, "imports point\nDOWN only\n(strict DAG)",
            color=GREEN, fontsize=10.5, ha="center", va="center", fontweight="bold")
    _accent_title(fig, "Strictly unidirectional layer dependencies",
                  "Any backward import is a code smell — the graph stays a DAG")
    _brand(fig)
    save(fig, "fig07_layers.png")


def fig8_source_of_truth():
    fig, ax = plt.subplots(figsize=(FIG_W + 0.4, FIG_H))
    fig.subplots_adjust(top=0.80, bottom=0.05, left=0.03, right=0.97)
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    sources = ["football-data.org", "Kaggle history", "StatsBomb", "FIFA rankings",
               "Transfermarkt\n(squad value)"]
    ys = np.linspace(6.2, 0.8, len(sources))
    for s, y in zip(sources, ys):
        ax.add_patch(mpatches.FancyBboxPatch((0.2, y - 0.4), 2.7, 0.8,
                     boxstyle="round,pad=0.02,rounding_size=0.08", fc=NAVY2, ec=BLUE, lw=1.6))
        ax.text(1.55, y, s, ha="center", va="center", color=TEXT, fontsize=8.6)
        ax.annotate("", xy=(4.6, 3.5), xytext=(2.95, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, alpha=.7))
    ax.add_patch(mpatches.FancyBboxPatch((4.6, 2.6), 2.9, 1.8,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=PANEL, ec=GREEN, lw=2.6))
    ax.text(6.05, 3.75, "matches_unified.csv", ha="center", color=GREEN, fontsize=11,
            fontweight="bold")
    ax.text(6.05, 3.30, "canonical table", ha="center", color=TEXT, fontsize=9.5)
    ax.text(6.05, 2.95, "validated before write", ha="center", color=MUTED, fontsize=8.5,
            style="italic")
    outs = ["ratings", "features", "models", "simulation", "picks / API"]
    yo = np.linspace(6.2, 0.8, len(outs))
    for o, y in zip(outs, yo):
        ax.add_patch(mpatches.FancyBboxPatch((9.1, y - 0.4), 2.6, 0.8,
                     boxstyle="round,pad=0.02,rounding_size=0.08", fc=NAVY2, ec=GOLD, lw=1.6))
        ax.text(10.4, y, o, ha="center", va="center", color=TEXT, fontsize=9)
        ax.annotate("", xy=(9.1, y), xytext=(7.55, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2, alpha=.7))
    _accent_title(fig, "One source of truth, everything derived",
                  "Heterogeneous sources → one validated canonical table → all downstream")
    _brand(fig)
    save(fig, "fig08_source_of_truth.png")


def fig9_cache_cost():
    N = np.linspace(1, 4000, 400)
    n2 = 48 * 47
    naive = 8 * N
    cached = n2 + 0 * N
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.95)
    ax.plot(N, naive, color=RED, lw=3, label="No cache:  ~8N model calls")
    ax.plot(N, cached, color=GREEN, lw=3, label=f"Cache:  n(n-1) = {n2} calls (flat)")
    ax.axvline(2000, color=MUTED, ls=":", lw=1.2)
    ax.text(2000, 30000, "default\nN = 2000", color=MUTED, fontsize=9, ha="center")
    ax.set_xlabel("Monte-Carlo runs  (N)")
    ax.set_ylabel("Expensive model evaluations")
    leg = ax.legend(facecolor=NAVY2, edgecolor=GRIDC, fontsize=10.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(TEXT)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.annotate("200x - 1500x\nfewer calls", (3000, 8 * 3000), (1500, 18000),
                color=GOLD, fontsize=11, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4))
    _accent_title(fig, "Memoization collapses a factorial workload",
                  "Expensive calls become a one-time precompute; the loop does O(1) lookups")
    _brand(fig)
    save(fig, "fig09_cache_cost.png")


def fig10_champion_reshuffle():
    teams = ["Brazil", "Morocco", "Spain"]
    degraded = [15, 2, 1]   # rank under ratings+Poisson only (illustrative, from notes)
    full = [5, 10, 1]       # rank under full blended model
    cols = [GREEN, RED, GOLD]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(top=0.78, bottom=0.12, left=0.13, right=0.84)
    for t, d, f, c in zip(teams, degraded, full, cols):
        ax.plot([0, 1], [d, f], color=c, lw=3, marker="o", ms=10, mec=NAVY)
        ax.text(-0.04, d, f"{t}  #{d}", ha="right", va="center", color=c, fontsize=11,
                fontweight="bold")
        ax.text(1.04, f, f"#{f}", ha="left", va="center", color=c, fontsize=11,
                fontweight="bold")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(16, 0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Degraded\n(ratings + Poisson)", "Full blended\n(+ squad value)"])
    ax.set_ylabel("Championship rank")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    _accent_title(fig, "Full-model scoring fixes the title race",
                  "Performance bought correctness: every pair scored with the full model")
    _brand(fig)
    save(fig, "fig10_champion_reshuffle.png")


def fig11_one_forecast_four_policies():
    fig, ax = plt.subplots(figsize=(FIG_W + 0.6, FIG_H))
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.03, right=0.97)
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 7)
    ax.add_patch(mpatches.FancyBboxPatch((0.4, 2.6), 3.1, 1.8,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc=PANEL, ec=GREEN, lw=2.6))
    ax.text(1.95, 3.85, "Calibrated forecast", ha="center", color=GREEN, fontsize=11.5,
            fontweight="bold")
    ax.text(1.95, 3.42, "(p_H, p_D, p_A)", ha="center", color=TEXT, fontsize=11)
    ax.text(1.95, 3.0, "one validated belief", ha="center", color=MUTED, fontsize=8.6,
            style="italic")
    profiles = [
        ("safe", "thr 0.58 · draw 0.90", GREEN, 5.8),
        ("balanced", "thr 0.50 · draw 1.00", BLUE, 4.25),
        ("aggressive", "thr 0.43 · upset 0.18", GOLD, 2.7),
        ("contrarian", "thr 0.38 · fade bias", RED, 1.15),
    ]
    for name, knobs, c, y in profiles:
        ax.add_patch(mpatches.FancyBboxPatch((7.4, y - 0.55), 4.2, 1.05,
                     boxstyle="round,pad=0.02,rounding_size=0.10", fc=NAVY2, ec=c, lw=2))
        ax.text(7.65, y + 0.12, name, ha="left", color=c, fontsize=11.5, fontweight="bold")
        ax.text(7.65, y - 0.28, knobs, ha="left", color=MUTED, fontsize=9)
        ax.annotate("", xy=(7.4, y), xytext=(3.5, 3.5),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7, alpha=.85))
    ax.text(5.4, 5.9, "decision layer", color=MUTED, fontsize=9.5, style="italic")
    _accent_title(fig, "One forecast → four decision policies",
                  "Belief is shared; only the utility-shaped policy changes (Strategy pattern)")
    _brand(fig)
    save(fig, "fig11_four_policies.png")


def fig12_parimutuel():
    p = np.linspace(0.08, 0.92, 200)
    payoff = 1 / p  # inverse: crowded favorite pays little, lonely underdog pays a lot
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(top=0.78, bottom=0.14, left=0.11, right=0.95)
    ax.plot(p, payoff, color=MUTED, lw=2.5)
    pts = [(0.72, "safe", GREEN), (0.55, "balanced", BLUE),
           (0.34, "aggressive", GOLD), (0.22, "contrarian", RED)]
    for pp, lab, c in pts:
        ax.scatter([pp], [1 / pp], color=c, s=120, zorder=5, edgecolor=NAVY)
        ax.annotate(lab, (pp, 1 / pp), (pp + 0.02, 1 / pp + 0.6),
                    color=c, fontsize=11, fontweight="bold")
    ax.set_xlabel("Probability of the pick  (model belief)")
    ax.set_ylabel("Payoff if correct  (pool share)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(0.62, 7.2, "lonely underdog:\nlow prob, high payoff", color=RED, fontsize=9.5)
    ax.text(0.55, 1.4, "crowded favorite: high prob, low payoff", color=GREEN, fontsize=9.5)
    _accent_title(fig, "Why argmax(p) is not the optimal bet",
                  "Pari-mutuel payoffs are competitive — the best action depends on risk appetite")
    _brand(fig)
    save(fig, "fig12_parimutuel.png")


# ============================================================ HERO BANNERS
def hero(name, kicker, title, sub, accent=GREEN):
    fig = plt.figure(figsize=(12, 5), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 5)
    # subtle motif: faint pitch arcs
    for r, a in [(2.2, 0.05), (3.0, 0.04), (3.8, 0.03)]:
        ax.add_patch(mpatches.Circle((11.8, 2.5), r, fill=False, ec=accent, lw=1.2,
                                     alpha=a))
    ax.add_patch(mpatches.Rectangle((0, 0), 0.16, 5, color=accent))
    ax.text(0.6, 4.1, kicker, color=accent, fontsize=14, fontweight="bold")
    # wrap title
    ax.text(0.6, 2.75, title, color=TEXT, fontsize=33, fontweight="bold", va="center")
    ax.text(0.6, 1.15, sub, color=MUTED, fontsize=14.5, va="center")
    ax.text(0.6, 0.35, "FIFA World Cup 2026 Quiniela Predictor V2", color=accent,
            fontsize=10.5, alpha=0.85)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=120, facecolor=NAVY)
    plt.close(fig)
    print("hero ->", name)


def build_heroes():
    hero("hero01.png", "MATHEMATICAL FOUNDATIONS",
         "Why a Skellam Distribution\nBeats a Score Grid",
         "Poisson goals, closed-form outcomes, and the bug hiding in a 5x5 matrix", GREEN)
    hero("hero02.png", "EXPERIMENTAL METHODOLOGY",
         "The Champion List Is a Liar",
         "A leakage-free backtest as the sole arbiter of every model change", GOLD)
    hero("hero03.png", "ARCHITECTURE & REPRODUCIBILITY",
         "One Source of Truth,\nEight Layers, Zero Surprises",
         "Layered dependencies, graceful degradation, and silent failures", BLUE)
    hero("hero04.png", "PERFORMANCE ENGINEERING",
         "200x - 1500x",
         "Caching and vectorizing a Monte-Carlo tournament simulator", PURPLE)
    hero("hero05.png", "DECISION THEORY",
         "Probability Is Not\na Decision",
         "Expected value, four risk profiles, and pari-mutuel game theory", RED)


# ============================================================ LINKEDIN CAROUSELS
def slide(path, kicker, title, body, footer, idx, total, accent=GREEN, big=False):
    fig = plt.figure(figsize=(8, 8), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.add_patch(mpatches.Rectangle((0, 0), 0.22, 10, color=accent))
    ax.add_patch(mpatches.Circle((9.6, 9.6), 1.8, fill=False, ec=accent, lw=1.2, alpha=.06))
    if kicker:
        ax.text(0.7, 9.2, kicker, color=accent, fontsize=15, fontweight="bold")
    tsize = 33 if big else 25
    ax.text(0.7, 7.7 if big else 8.1, title, color=TEXT, fontsize=tsize,
            fontweight="bold", va="top")
    y = 5.7 if big else 6.2
    for line in body:
        bullet = line.startswith("- ")
        # A non-bullet line that starts with whitespace (e.g. a lone "↓") is a
        # continuation marker: align it to the bullet-text column, not the far-
        # left muted column, so it reads as part of the bullet flow.
        indent = (not bullet) and line[:1] == " " and line.strip() != ""
        txt = line[2:] if bullet else line
        if bullet:
            ax.add_patch(mpatches.Circle((0.95, y + 0.12), 0.07, color=accent))
            ax.text(1.3, y, txt, color=TEXT, fontsize=15.5, va="center")
        elif indent:
            ax.text(1.3, y, line.strip(), color=TEXT, fontsize=15.5, va="center")
        else:
            ax.text(0.7, y, txt, color=MUTED, fontsize=15, va="center")
        y -= 1.02
    ax.text(0.7, 0.55, footer, color=MUTED, fontsize=11, va="center")
    ax.text(9.3, 0.55, f"{idx}/{total}", color=accent, fontsize=12, fontweight="bold",
            ha="right", va="center")
    fig.savefig(path, dpi=135, facecolor=NAVY)
    plt.close(fig)


def build_carousel(prefix, accent, slides):
    total = len(slides)
    for i, s in enumerate(slides, 1):
        p = os.path.join(CAR, f"{prefix}_slide{i:02d}.png")
        slide(p, s.get("kicker", ""), s["title"], s.get("body", []),
              s.get("footer", "@DSarceno · Predictor Mundial 2026"), i, total, accent,
              big=s.get("big", False))
    print(f"carousel {prefix}: {total} slides")


def build_carousels():
    build_carousel("c1", GREEN, [
        {"kicker": "PROBABILIDADES 1X2", "title": "No integres una\nmatriz. Usa\nSkellam.", "big": True,
         "body": ["", "Un modelo de fútbol", "de muestra pequeña."]},
        {"title": "El problema", "body": [
            "- ~10 partidos oficiales / año",
            "- ~10⁴ partidos útiles",
            "- El deep learning queda fuera", "- Entran los modelos estructurales"]},
        {"title": "El insight", "body": [
            "- Goles ~ Poisson", "- Diferencia D = G_a - G_b",
            "- D sigue una Skellam", "- P(L)=P(D>0), P(E)=P(D=0)",
            "- 3 llamadas a scipy.stats.skellam"]},
        {"title": "Por qué gana", "body": [
            "- Sin truncar la matriz", "- Sin renormalizar",
            "- Empates mejor calibrados", "- Sin bug de orientación"]},
        {"title": "El bug que mata", "body": [
            "- Triángulos invertidos",
            "- Poisson ~50% del blend",
            "- Paraguay > España (campeón)",
            "- El bug probabilístico no truena"]},
        {"title": "Para llevar", "body": [
            "- Forma cerrada > matriz", "- Hibridiza por pregunta",
            "- Ensemble = ortogonalidad", "- Shrinkage en muestra pequeña"]},
        {"title": "¿Tú cómo sacas\nel 1X2?", "big": True,
         "body": ["", "¿Matriz o Skellam?", "Hablemos ↓"]},
    ])
    build_carousel("c2", GOLD, [
        {"kicker": "EVALUACIÓN DE MODELOS", "title": "La lista de\ncampeones\nmiente.", "big": True,
         "body": ["", "La validez aparente", "no es evidencia."]},
        {"title": "La trampa", "body": [
            "- Ajustas un prior → se ve mejor",
            "- España #1, Brasil top 5", "- Se siente como progreso",
            "- La métrica real empeoró"]},
        {"title": "El árbitro", "body": [
            "- Entrena con año < Y", "- Evalúa el torneo == Y",
            "- As-of join → sin fuga", "- Decide por log-loss held-out"]},
        {"title": "Veredicto 1: ACEPTADO", "body": [
            "- Valor de plantilla (Transfermarkt)",
            "- Log-loss 0.9994 → 0.9905",
            "- Brier y RPS mejoran",
            "- Accuracy igual (la habría matado)"]},
        {"title": "Veredicto 2: RECHAZADO", "body": [
            "- Mixture prior élite/regular",
            "- Hacía la lista MÁS bonita",
            "- Log-loss +2.3%, acc -2.4 pts",
            "- Quedó desactivado"]},
        {"title": "Para llevar", "body": [
            "- Fija la métrica primero",
            "- La accuracy oculta mejoras",
            "- Prevenir fuga es mecánico",
            "- Construye el árbitro primero"]},
        {"title": "¿Cómo separas\n'se ve bien' de\n'está mejor'?", "big": True,
         "body": ["", "Comenta abajo ↓"]},
    ])
    build_carousel("c3", BLUE, [
        {"kicker": "MLOps", "title": "Un config que\nno hacía nada\ndurante meses.", "big": True,
         "body": ["", "La arquitectura mata más", "sistemas que los modelos."]},
        {"title": "Cuatro pilares", "body": [
            "- Capas unidireccionales (DAG)",
            "- Una fuente de verdad (CSV)",
            "- Etapas aisladas por archivos",
            "- Degradación elegante"]},
        {"title": "El config muerto", "body": [
            "- ensemble.weights no se leía",
            "- Config.get() leía otro archivo",
            "- Editar el YAML → nada cambiaba",
            "- Un knob muerto finge confianza"]},
        {"title": "Bug silencioso de pandas", "body": [
            "- out[c]=raw[c] tras dropna",
            "- sin reset_index(drop=True)",
            "- alinea índice → casi todo NaN",
            "- sin excepción alguna"]},
        {"title": "Estado no serializable", "body": [
            "- defaultdict(lambda: ...)",
            "- rompe joblib.dump",
            "- fix: __getstate__/__setstate__",
            "- si no, falla en silencio"]},
        {"title": "Para llevar", "body": [
            "- Concentra el riesgo y protégelo",
            "- El config muerto es un pasivo",
            "- Casi todo bug ML es silencioso",
            "- Vuelve ruidoso lo silencioso"]},
        {"title": "¿Tu bug\nSILENCIOSO más\ncaro?", "big": True,
         "body": ["", "¿Config muerto? ¿NaN?", "¿Pickle roto? ↓"]},
    ])
    build_carousel("c4", PURPLE, [
        {"kicker": "RENDIMIENTO", "title": "200x - 1500x\nmás rápido.", "big": True,
         "body": ["", "Memoización + NumPy en un", "simulador Monte-Carlo."]},
        {"title": "El problema", "body": [
            "- 'Campeón' no tiene forma cerrada",
            "- Cuadro endógeno → Monte-Carlo",
            "- El error cae como O(1/√N)",
            "- No basta con muestrear más"]},
        {"title": "El insight", "body": [
            "- Solo 48 equipos",
            "- n(n-1) = 2256 enfrentamientos",
            "- La misma pregunta, miles de veces",
            "- Función pura → memoiza"]},
        {"title": "La transformación", "body": [
            "- O(N·m·cost)",
            "      ↓",
            "- O(n²·cost + N·m)",
            "- Loop interno: lookup O(1)"]},
        {"title": "El verdadero premio", "body": [
            "- Barato → modelo COMPLETO",
            "- Brasil #15 → #5",
            "- España → #1 al 12%",
            "- La velocidad compró correctitud"]},
        {"title": "Para llevar", "body": [
            "- Busca la pregunta repetida",
            "- Vectoriza el bookkeeping",
            "- Amortiza warmups reusando",
            "- Rápido + mal sigue siendo mal"]},
        {"title": "¿Memoizar o\nparalelizar\nprimero?", "big": True,
         "body": ["", "¿Cuál es tu default? ↓"]},
    ])
    build_carousel("c5", RED, [
        {"kicker": "TEORÍA DE DECISIÓN", "title": "La probabilidad\nno es una\ndecisión.", "big": True,
         "body": ["", "Un forecast.", "Cuatro apuestas distintas."]},
        {"title": "La confusión", "body": [
            "- 'Produce probabilidades'",
            "- Es solo media solución",
            "- Creencia ≠ acción",
            "- argmax(p) asume utilidad 0/1"]},
        {"title": "El juego", "body": [
            "- Pari-mutuel: se reparte el pozo",
            "- Favorito lleno paga poco",
            "- Sorpresa sola paga mucho",
            "- Gana la precisión diferenciada"]},
        {"title": "Cuatro políticas", "body": [
            "- safe: favoritos estrictos",
            "- balanced: fallback a empate",
            "- aggressive: sorpresas con criterio",
            "- contrarian: fade a la masa"]},
        {"title": "Por qué desacoplar", "body": [
            "- Un solo forecast validado",
            "- Barato añadir un 5º perfil",
            "- La calibración sigue honesta",
            "- Strategy = utilidad en código"]},
        {"title": "Para llevar", "body": [
            "- Una probabilidad no es decisión",
            "- Pagos competitivos rompen argmax",
            "- Mide cada capa por separado",
            "- ¿Qué utilidad gobierna de verdad?"]},
        {"title": "¿Qué utilidad sirve\nel output de TU\nmodelo?", "big": True,
         "body": ["", "Casi nunca la 0/1 ↓"]},
    ])


if __name__ == "__main__":
    build_equations()
    fig1_skellam_vs_grid()
    fig2_time_decay()
    fig3_shrinkage()
    fig4_squad_value()
    fig5_mixture_verdict()
    fig6_leakage_free()
    fig7_layers()
    fig8_source_of_truth()
    fig9_cache_cost()
    fig10_champion_reshuffle()
    fig11_one_forecast_four_policies()
    fig12_parimutuel()
    build_heroes()
    build_carousels()
    print("\nDONE.")
