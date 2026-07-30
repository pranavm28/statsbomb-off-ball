"""
Figures for the project book. Every chart is generated from the real outputs in
this repository -- nothing is illustrative-only unless the caption says so.

Run:  python docs/book_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# book palette: light background, print-friendly
BG = "#ffffff"
INK = "#1b1f27"
MUTED = "#6b7280"
NAVY = "#1f3b63"
TEAL = "#12836a"
RED = "#b23a48"
AMBER = "#c98a2e"
GRID = "#e5e7eb"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#c9cdd4", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
})


def save(fig, name):
    fig.savefig(FIG / name, dpi=170, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- fig 1
def fig_markov_chain():
    """How value iteration propagates goal probability backwards across cells."""
    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.axis("off")
    xs = [0.06, 0.3, 0.54, 0.78]
    labels = ["Own half", "Middle", "Final third", "Penalty area"]
    vals = ["xT ≈ 0.004", "xT ≈ 0.012", "xT ≈ 0.05", "xT ≈ 0.33"]
    for i, (x, lab, v) in enumerate(zip(xs, labels, vals)):
        shade = ["#eef2f7", "#dfe8f2", "#c3d6ea", "#9dbfe0"][i]
        ax.add_patch(Rectangle((x, 0.42), 0.16, 0.26, facecolor=shade,
                               edgecolor=NAVY, lw=1.1, transform=ax.transAxes))
        ax.text(x + 0.08, 0.60, lab, ha="center", fontsize=8.5, fontweight="bold",
                transform=ax.transAxes)
        ax.text(x + 0.08, 0.49, v, ha="center", fontsize=8, color=NAVY,
                transform=ax.transAxes)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 0.165, 0.55), (xs[i + 1] - 0.005, 0.55),
                                         arrowstyle="-|>", mutation_scale=13,
                                         color=MUTED, lw=1.2, transform=ax.transAxes))
            ax.text((x + 0.165 + xs[i + 1]) / 2, 0.585, "move", ha="center",
                    fontsize=7.2, color=MUTED, transform=ax.transAxes)
        ax.add_patch(FancyArrowPatch((x + 0.08, 0.41), (x + 0.08, 0.24),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=RED, lw=1.0, transform=ax.transAxes))
        ax.text(x + 0.095, 0.30, "shoot", fontsize=7, color=RED, transform=ax.transAxes)
    ax.text(0.5, 0.11, "Value flows BACKWARDS: the penalty area is valuable because shots "
                       "score;\nthe middle is valuable because it leads to the final third.",
            ha="center", fontsize=8.4, color=MUTED, transform=ax.transAxes)
    ax.text(0.5, 0.86, "A Markov chain over pitch cells",
            ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
    save(fig, "fig01_markov.png")


# ---------------------------------------------------------------- fig 2
def fig_xt_surface():
    """The fitted threat surface actually used in this project."""
    grid = np.load(config.DATA_PROC / "threat_grid.npy")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # A linear scale is dominated by the single hottest cell and washes the rest of
    # the pitch to a flat pale yellow. A power norm (gamma < 1) compresses the top and
    # opens up the low end, so the actual gradient across the pitch is visible.
    from matplotlib.colors import PowerNorm
    im = ax.imshow(grid.T, origin="lower", extent=[0, 120, 0, 80],
                   cmap="YlOrRd", aspect="equal",
                   norm=PowerNorm(gamma=0.45, vmin=grid.min(), vmax=grid.max()))
    for x in (0, 60, 120):
        ax.axvline(x, color="#5b6472", lw=0.8)
    ax.add_patch(Rectangle((102, 22), 18, 36, fill=False, edgecolor="#5b6472", lw=0.8))
    ax.add_patch(Rectangle((0, 22), 18, 36, fill=False, edgecolor="#5b6472", lw=0.8))
    ax.set_xlabel("attacking direction  →")
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02,
                      label="xT (value of holding the ball)")
    cb.set_ticks([0.06, 0.10, 0.18, 0.30, 0.47])
    ax.set_title("The fitted threat surface", fontsize=11, fontweight="bold", pad=9)
    save(fig, "fig02_xt_surface.png")


# ---------------------------------------------------------------- fig 3
def fig_two_frames():
    """The reconstruction: release frame + receipt frame = the run."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3))
    titles = ["Frame A — pass struck", "Frame B — ball received", "The difference = the run"]
    rng = np.random.default_rng(4)
    mates = np.array([[35, 30], [48, 55], [62, 22], [70, 45], [55, 38]], float)
    defs = np.array([[72, 32], [80, 50], [66, 60], [88, 40], [58, 18]], float)
    runner_a = np.array([62, 22]); runner_b = np.array([86, 28]); ball = np.array([48, 55])
    for k, ax in enumerate(axes):
        ax.set_xlim(20, 110); ax.set_ylim(5, 75); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((20, 5), 90, 70, fill=False, edgecolor="#aeb4bd", lw=1))
        ax.add_patch(Rectangle((92, 22), 18, 36, fill=False, edgecolor="#aeb4bd", lw=1))
        ax.scatter(defs[:, 0], defs[:, 1], s=52, c=RED, zorder=3)
        show_mates = mates if k == 0 else np.vstack([mates[:2], mates[3:]])
        ax.scatter(show_mates[:, 0], show_mates[:, 1], s=52, c=NAVY, zorder=3)
        if k == 0:
            ax.scatter(*runner_a, s=90, facecolors="none", edgecolors=TEAL, lw=2, zorder=4)
            ax.annotate("which dot is\nthe receiver?", runner_a + [-24, -14], fontsize=7.4,
                        color=TEAL, ha="center")
            ax.scatter(*ball, s=40, c=AMBER, zorder=5, marker="o")
        elif k == 1:
            ax.scatter(*runner_b, s=95, c=TEAL, zorder=5)
            ax.annotate("named + exact\n(tagged 'actor')", runner_b + [-4, 12], fontsize=7.4,
                        color=TEAL, ha="center")
        else:
            ax.scatter(*runner_a, s=80, facecolors="none", edgecolors=TEAL, lw=1.6, zorder=4)
            ax.scatter(*runner_b, s=95, c=TEAL, zorder=5)
            ax.annotate("", xy=runner_b, xytext=runner_a,
                        arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.1, ls=(0, (2, 1.6))))
            ax.annotate("", xy=runner_b, xytext=ball,
                        arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.6))
            ax.text(64, 12, "gold = the pass    green = the run", fontsize=7.6, color=MUTED)
        ax.set_title(titles[k], fontsize=9.4, fontweight="bold", pad=6)
    save(fig, "fig03_two_frames.png")


# ---------------------------------------------------------------- fig 4
def fig_auc_explained():
    """What AUC means, with the project's own score marked."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.5))
    rng = np.random.default_rng(1)
    neg = rng.beta(2, 9, 4000); pos = rng.beta(5, 4, 900)
    a1.hist(neg, bins=40, alpha=.75, color="#9fb3c8", label="no shot followed")
    a1.hist(pos, bins=40, alpha=.75, color=RED, label="shot followed")
    a1.set_xlabel("model's predicted probability"); a1.set_yticks([])
    a1.legend(fontsize=7.6, frameon=False)
    a1.set_title("AUC asks: are the red bars to the RIGHT of the blue?",
                 fontsize=9, fontweight="bold")
    ths = np.linspace(0, 1, 200)
    tpr = [(pos > t).mean() for t in ths]; fpr = [(neg > t).mean() for t in ths]
    a2.plot(fpr, tpr, color=NAVY, lw=2)
    a2.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    a2.fill_between(fpr, tpr, alpha=.10, color=NAVY)
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.text(.52, .22, "AUC = area\nunder this curve", fontsize=8.4, color=NAVY)
    a2.text(.52, .08, "0.5 = coin flip", fontsize=7.8, color=MUTED)
    a2.set_title("The ROC curve", fontsize=9, fontweight="bold")
    for ax in (a1, a2):
        ax.grid(alpha=.25, color=GRID)
    save(fig, "fig04_auc.png")


# ---------------------------------------------------------------- fig 5
def fig_calibration():
    """Real calibration of the possession-value model."""
    rep = json.load(open(config.OUTPUTS / "value_report.json"))
    cal = rep["calibration"]["plus_360"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, .3], [0, .3], ls="--", color=MUTED, lw=1, label="perfect")
    ax.plot(cal["mean_pred"], cal["frac_pos"], "o-", color=TEAL, lw=1.8, ms=5,
            label="our model")
    ax.set_xlabel("what the model predicted"); ax.set_ylabel("what actually happened")
    ax.set_title("Calibration: are the probabilities honest?", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.25, color=GRID)
    save(fig, "fig05_calibration.png")


# ---------------------------------------------------------------- fig 6
def fig_groupkfold():
    """Why a random split leaks and a match-grouped split does not."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.4, 3.0))
    rng = np.random.default_rng(3)
    n = 60
    match_id = np.repeat(np.arange(6), 10)
    rand_fold = rng.integers(0, 2, n)
    for ax, folds, title, note in [
        (a1, rand_fold, "Random split — LEAKS",
         "passes from the SAME move end up on both sides"),
        (a2, (match_id % 2), "Split by match — safe",
         "a whole match is either training or testing, never split")]:
        for i in range(n):
            ax.add_patch(Rectangle((i, 0), .86, 1,
                                   color=NAVY if folds[i] == 0 else "#d9a441"))
        for m in range(1, 6):
            ax.axvline(m * 10 - .07, color=INK, lw=1.1)
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(f"{title}   —   {note}", fontsize=8.8, fontweight="bold", loc="left")
    fig.text(.5, -.04, "each block = one action   |   vertical lines = match boundaries   |   "
                       "navy = train, gold = test", ha="center", fontsize=7.8, color=MUTED)
    save(fig, "fig06_groupkfold.png")


# ---------------------------------------------------------------- fig 7
def fig_ablation():
    """The four information tiers."""
    rep = json.load(open(config.OUTPUTS / "ablation_report.json"))
    tiers = rep["tiers"]
    names = ["1. Event data\nonly", "2. + where the\nrunner is",
             "3. + the defence\naround him", "4. + how\nsurrounded"]
    aucs = [t["auc"] for t in tiers]
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    cols = ["#9fb3c8", TEAL, TEAL, "#c9cdd4"]
    bars = ax.bar(names, aucs, color=cols, width=.62)
    ax.set_ylim(0.915, 0.949)
    ax.set_ylabel("AUC")
    for i, (b, t) in enumerate(zip(bars, tiers)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .0007,
                f"{t['auc']:.4f}", ha="center", fontsize=8.6, fontweight="bold")
        if t["gain_vs_previous"]:
            ax.text(b.get_x() + b.get_width() / 2, 0.9175,
                    f"+{t['gain_vs_previous']:.4f}", ha="center", fontsize=8,
                    color=TEAL if t["gain_vs_previous"] > .003 else MUTED)
    ax.set_title("What each layer of information is worth\n"
                 f"(total gain from the camera data: +{rep['total_gain_from_360']:.4f})",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=.25, color=GRID); ax.set_axisbelow(True)
    save(fig, "fig07_ablation.png")


# ---------------------------------------------------------------- fig 8
def fig_encirclement():
    """What encirclement means, geometrically."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    setups = [
        ("Clear escape side  (encirclement ≈ 0.1)",
         np.array([[.2, .78], [.32, .68], [.15, .58]])),
        ("Surrounded  (encirclement ≈ 0.8)",
         np.array([[.25, .8], [.78, .72], [.8, .28], [.22, .25], [.5, .88]])),
    ]
    for ax, (title, defs) in zip(axes, setups):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
        c = Circle((.5, .5), .34, fill=False, ls=(0, (3, 3)), edgecolor="#c9cdd4")
        ax.add_patch(c)
        ax.scatter(.5, .5, s=140, c=TEAL, zorder=4)
        ax.scatter(defs[:, 0], defs[:, 1], s=85, c=RED, zorder=4)
        for d in defs:
            ax.plot([.5, d[0]], [.5, d[1]], color="#d8dce2", lw=1, zorder=1)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    fig.text(.5, -.02, "Encirclement measures the LARGEST ANGULAR GAP between nearby "
                       "defenders.\nA big gap means somewhere to escape to; no gap means "
                       "bodies on every side.",
             ha="center", fontsize=8.2, color=MUTED)
    save(fig, "fig08_encirclement.png")


# ---------------------------------------------------------------- fig 9
def fig_block_spread_trap():
    """The feature that looks important and is really a depth proxy."""
    r = pd.read_parquet(config.DATA_PROC / "runs.parquet")
    u = r[(r["usable"] == 1) & (r["is_run"] == 1)].dropna(subset=["block_spread"])
    ft = u[u["receipt_x"] >= 80].copy()
    ft["q"] = pd.qcut(ft["block_spread"], 4, labels=["most\ncompact", "Q2", "Q3",
                                                     "most\nstretched"])
    g = ft.groupby("q", observed=True).agg(shot=("shot_5s", "mean"),
                                           depth=("receipt_x", "mean")).reset_index()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a1.bar(g["q"].astype(str), g["shot"], color=[RED, "#d08a94", "#c9cdd4", "#9fb3c8"], width=.6)
    a1.set_ylabel("shot within 5s"); a1.set_title("Looks like a finding…", fontsize=9.4,
                                                  fontweight="bold")
    for i, v in enumerate(g["shot"]):
        a1.text(i, v + .006, f"{v:.3f}", ha="center", fontsize=8.4, fontweight="bold")
    a2.bar(g["q"].astype(str), g["depth"], color="#9fb3c8", width=.6)
    a2.set_ylim(85, 100); a2.set_ylabel("mean distance up the pitch")
    a2.set_title("…until you look at this", fontsize=9.4, fontweight="bold")
    for i, v in enumerate(g["depth"]):
        a2.text(i, v + .3, f"{v:.1f}", ha="center", fontsize=8.4, fontweight="bold")
    for ax in (a1, a2):
        ax.grid(axis="y", alpha=.25, color=GRID); ax.set_axisbelow(True)
    fig.text(.5, -.04, "A defence is compact BECAUSE the ball is deep in its own box. "
                       "The feature is mostly a restatement of position.",
             ha="center", fontsize=8.2, color=MUTED)
    save(fig, "fig09_block_spread_trap.png")


# ---------------------------------------------------------------- fig 10
def fig_journey():
    """The decision timeline: what we built, what the result was, where we pivoted."""
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    ax.axis("off")
    steps = [
        ("Build 1", "xT + line-breaking passes", "rejected — too generic", RED),
        ("Build 2", "space model (pitch control × threat)", "heuristic, not fitted", RED),
        ("Build 3", "reception value, properly fitted", "NULL: 360 adds +0.0026", AMBER),
        ("Build 4", "off-ball runs from two frames", "run features add +0.0197", TEAL),
        ("Refine", "isolate run from pass, final third", "leaderboard reads true", TEAL),
        ("Refine", "compactness / encirclement", "split: null in V, helps runs", AMBER),
    ]
    y = 0.92
    for i, (tag, what, result, col) in enumerate(steps):
        ax.text(0.02, y, tag, fontsize=8.6, fontweight="bold", color=MUTED,
                transform=ax.transAxes)
        ax.add_patch(Rectangle((0.13, y - .045), 0.44, .078, facecolor="#f4f6f9",
                               edgecolor="#dfe3e9", transform=ax.transAxes))
        ax.text(0.15, y - .012, what, fontsize=9, transform=ax.transAxes)
        ax.add_patch(Circle((0.60, y - .006), .011, color=col, transform=ax.transAxes))
        ax.text(0.635, y - .012, result, fontsize=8.8, color=col, fontweight="bold",
                transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.35, y - .05), (0.35, y - .10),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#c9cdd4", lw=1.1, transform=ax.transAxes))
        y -= 0.155
    ax.text(0.02, 0.99, "The path: each result is what triggered the next decision",
            fontsize=10.5, fontweight="bold", transform=ax.transAxes)
    save(fig, "fig10_journey.png")


if __name__ == "__main__":
    print("generating book figures ...")
    fig_markov_chain(); fig_xt_surface(); fig_two_frames(); fig_auc_explained()
    fig_calibration(); fig_groupkfold(); fig_ablation(); fig_encirclement()
    fig_block_spread_trap(); fig_journey()
    print("done ->", FIG)
