"""Plots for the Diabetes dataset. Each plot is saved as a PNG file."""

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, TARGET_COLUMN
from src.explore import get_target_correlations


def _save(fig, filename):
    """Save a figure into outputs/figures and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def plot_target_histogram(df):
    """Histogram: how disease progression values are distributed."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET_COLUMN], bins=30, color="steelblue", edgecolor="black")
    ax.set_title("Distribution of Disease Progression (Target)")
    ax.set_xlabel("Disease progression one year after baseline")
    ax.set_ylabel("Number of patients")
    _save(fig, "01_target_histogram.png")


def plot_feature_correlations(df):
    """Bar chart: correlation of each feature with the target."""
    correlations = get_target_correlations(df)
    colors = ["seagreen" if value > 0 else "indianred" for value in correlations]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(correlations.index, correlations.values, color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Correlation of Each Feature with Disease Progression")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Correlation coefficient (r)")
    _save(fig, "02_feature_correlations.png")


def plot_bmi_vs_target(df):
    """Scatter plot: BMI (strongest feature) against the target."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["bmi"], df[TARGET_COLUMN], alpha=0.6, color="darkorange", edgecolor="black")
    ax.set_title("Body Mass Index vs Disease Progression")
    ax.set_xlabel("Body mass index (BMI)")
    ax.set_ylabel("Disease progression one year after baseline")
    _save(fig, "03_bmi_vs_target.png")


def plot_predicted_vs_actual(y_test, y_pred):
    """Scatter plot: model predictions against the true values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.7, color="mediumpurple", edgecolor="black")

    # Dashed line = perfect predictions (predicted == actual)
    low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([low, high], [low, high], "k--", label="Perfect prediction")

    ax.set_title("Linear Regression: Predicted vs Actual (Test Set)")
    ax.set_xlabel("Actual disease progression")
    ax.set_ylabel("Predicted disease progression")
    ax.legend()
    _save(fig, "04_predicted_vs_actual.png")


# ---------------------------------------------------------------------------
# Model comparison figures
# The selected model is blue, a significantly worse model is red, and all
# other models stay a quiet gray.
# ---------------------------------------------------------------------------
BEST_COLOR = "#2a78d6"
WORST_COLOR = "#e34948"
OTHER_COLOR = "#c3c2b7"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"


def _style_axes(ax):
    """Light, recessive axes so the data carries the message."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(OTHER_COLOR)
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK)
    ax.tick_params(axis="x", colors=MUTED_INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _model_colors(names, selected, tied):
    return [BEST_COLOR if n == selected else OTHER_COLOR if n in tied else WORST_COLOR
            for n in names]


def _join(names):
    """'A', 'A and B' or 'A, B and C'."""
    names = list(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _save_with_footer(fig, filename, footer):
    """Save a figure with a small note under the plot."""
    fig.text(0.02, 0.015, footer, ha="left", va="bottom", fontsize=10, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def _verdict(results, selected, tied):
    """One sentence that states the statistical result in plain words."""
    worse = [m for m in results["Model"] if m not in tied]
    text = f"{_join(tied)} are statistically tied"
    if worse:
        text += f"; {_join(worse)} {'is' if len(worse) == 1 else 'are'} significantly worse"
    return text + f".\nSelected: {selected} (the simplest of the tied models)."


def plot_model_r2(results, fold_r2, selected, tied):
    """Box plot: R2 of every model on each of the cross-validation rounds."""
    names = results["Model"].tolist()  # best mean R2 first
    colors = _model_colors(names, selected, tied)
    positions = list(range(len(names)))[::-1]
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for name, pos, color in zip(names, positions, colors):
        scores = fold_r2[name].to_numpy()
        ax.boxplot(scores, positions=[pos], vert=False, widths=0.55, showfliers=False,
                   patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                   boxprops={"facecolor": color, "edgecolor": "none", "alpha": 0.55},
                   whiskerprops={"color": MUTED_INK}, capprops={"color": MUTED_INK})
        # every single round as a dot, so nothing is hidden behind the box
        ax.scatter(scores, pos + rng.uniform(-0.16, 0.16, len(scores)), s=14,
                   color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(1.01, pos, f"mean {scores.mean():.3f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=11, color=INK,
                fontweight="bold" if name == selected else "normal")

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{n}  (selected)" if n == selected else n for n in names])
    _style_axes(ax)
    ax.set_xlabel("R² on unseen data (1.0 = perfect prediction)", color=MUTED_INK)
    fig.suptitle("Which model predicts one-year disease progression best?",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    ax.set_title(f"Each dot is one of {len(fold_r2)} cross-validation rounds; every model was "
                 "tuned and tested on the same rounds.\nBoxes that overlap mean there is no "
                 "real difference between the models.",
                 loc="left", fontsize=10, color=MUTED_INK, pad=10)
    _save_with_footer(fig, "05_model_comparison_r2.png", _verdict(results, selected, tied))


def plot_model_summary(results, selected, tied):
    """Bar chart: mean accuracy (R2) and mean error (MAE) side by side.

    Models are in the same order in both panels (best R2 at the top).
    """
    names = results["Model"].tolist()
    worst = names[-1]
    colors = _model_colors(names, selected, tied)
    positions = list(range(len(names)))[::-1]

    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    panels = [
        (ax_r2, results["CV R2 mean"].to_numpy(), "{:.3f}",
         "Accuracy (R²): higher is better", "R² (1.0 = perfect prediction)"),
        (ax_mae, results["CV MAE mean"].to_numpy(), "{:.1f}",
         "Average error (MAE): lower is better", "Average error in progression units"),
    ]
    for ax, values, fmt, title, xlabel in panels:
        ax.barh(positions, values, height=0.6, color=colors)
        ax.axvline(0, color=MUTED_INK, linewidth=1)
        span = max(values.max(), 0) - min(values.min(), 0)
        for name, pos, value in zip(names, positions, values):
            tag = "  SELECTED" if name == selected else "  WORST" if name == worst else ""
            ax.text(max(value, 0) + 0.02 * span, pos, fmt.format(value) + tag,
                    va="center", ha="left", fontsize=11, color=INK,
                    fontweight="bold" if tag else "normal")
        ax.set_xlim(min(values.min(), 0) - 0.05 * span, max(values.max(), 0) + 0.42 * span)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel(xlabel, color=MUTED_INK)

    ax_r2.set_yticks(positions)
    ax_r2.set_yticklabels(names)
    fig.suptitle("Model comparison: average scores over 50 cross-validation rounds",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    _save_with_footer(fig, "06_model_summary.png", _verdict(results, selected, tied))


def plot_selected_predicted_vs_actual(name, y_test, y_pred, r2):
    """Scatter plot: the selected model's test-set predictions against true values."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, s=40, alpha=0.75, color=BEST_COLOR,
               edgecolor="white", linewidth=0.8)

    low, high = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([low, high], [low, high], "--", color=MUTED_INK, linewidth=1.5,
            label="Perfect prediction")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{name} (selected model): predicted vs actual\n"
                 f"Held-out test set, R² = {r2:.3f}",
                 fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel("Actual disease progression after one year")
    ax.set_ylabel("Predicted disease progression after one year")
    ax.legend(frameon=False)
    _save(fig, "07_selected_model_predicted_vs_actual.png")
