"""Plots for the Diabetes dataset. Each plot is saved as a PNG file."""

import matplotlib

matplotlib.use("Agg")  # save plots to files without opening a window
import matplotlib.pyplot as plt

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


# Colors for the model comparison figures: the best model is highlighted,
# all other models stay a quiet gray.
BEST_COLOR = "#2a78d6"
OTHER_COLOR = "#c3c2b7"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"


def _style_axes(ax):
    """Light, recessive axes so the bars carry the message."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(OTHER_COLOR)
    ax.tick_params(axis="y", length=0, labelsize=11, labelcolor=INK)
    ax.tick_params(axis="x", colors=MUTED_INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _comparison_bars(results, value_col, std_col, best_name, title, subtitle,
                     xlabel, filename, fmt, best_tag="  (best)"):
    """Horizontal bar chart with one bar per model, best model highlighted."""
    names = results["Model"].tolist()
    values = results[value_col].to_numpy()
    errors = results[std_col].to_numpy() if std_col else [0.0] * len(values)
    colors = [BEST_COLOR if name == best_name else OTHER_COLOR for name in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = list(range(len(names)))[::-1]  # first row of the table at the top
    ax.barh(positions, values, height=0.6, color=colors,
            xerr=errors if std_col else None,
            error_kw={"ecolor": MUTED_INK, "capsize": 4, "linewidth": 1})
    ax.set_yticks(positions)
    ax.set_yticklabels([f"{n}{best_tag}" if n == best_name else n for n in names])
    ax.axvline(0, color=MUTED_INK, linewidth=1)

    # Value label just right of the bar's error bar (or of zero, whichever
    # is further right), so a negative bar's label never runs into the
    # model names on the left or hides behind its own error bar.
    span = max(values.max(), 0) - min(values.min(), 0)
    for name, pos, value, err in zip(names, positions, values, errors):
        x = max(value + err, 0) + 0.015 * span
        ax.text(x, pos, fmt.format(value), va="center", ha="left", fontsize=11,
                color=INK, fontweight="bold" if name == best_name else "normal")

    low = min((values - errors).min(), 0)
    high = (values + errors).max()
    ax.set_xlim(low - 0.05 * span, high + 0.15 * span)

    _style_axes(ax)
    ax.set_xlabel(xlabel, color=MUTED_INK)
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    ax.set_title(subtitle, loc="left", fontsize=10, color=MUTED_INK, pad=10)
    _save(fig, filename)


def plot_model_r2(results):
    """Cross-validated R2 for every model (higher is better)."""
    best = results.sort_values("CV R2 mean", ascending=False).iloc[0]
    ordered = results.sort_values("CV R2 mean", ascending=False)
    _comparison_bars(
        ordered, "CV R2 mean", "CV R2 std", best["Model"],
        title=f"{best['Model']} predicts disease progression best (R² = {best['CV R2 mean']:.3f})",
        subtitle="Mean R² over 5-fold cross-validation; error bars = ±1 std across folds. "
                 "Higher is better.\nLasso, Ridge and Linear Regression are within 0.002 "
                 "of each other: a near tie.",
        xlabel="R² (share of variation in one-year disease progression explained)",
        filename="05_model_comparison_r2.png", fmt="{:.3f}",
    )


def plot_model_mae(results):
    """Cross-validated MAE for every model (lower is better).

    The highlighted bar is the best model by R2, so each model keeps the
    same color in every comparison figure.
    """
    best_by_r2 = results.sort_values("CV R2 mean", ascending=False).iloc[0]["Model"]
    ordered = results.sort_values("CV MAE mean", ascending=True)
    _comparison_bars(
        ordered, "CV MAE mean", None, best_by_r2,
        title="Linear models make the smallest errors (about 44 units)",
        subtitle="Mean absolute error over 5-fold cross-validation. Lower is better.\n"
                 f"Blue = best model by R² ({best_by_r2}). The top three differ by "
                 "less than 0.3 units.",
        xlabel="MAE (average error in disease-progression units)",
        filename="06_model_comparison_mae.png", fmt="{:.1f}", best_tag="",
    )


WORST_COLOR = "#e34948"


def plot_model_summary(results):
    """One simple figure: accuracy (R2) and error (MAE) side by side.

    Models are in the same order in both panels (best R2 at the top), the
    best model is blue, the worst is red, and both are labelled in words.
    """
    ordered = results.sort_values("CV R2 mean", ascending=False)
    names = ordered["Model"].tolist()
    best, worst = names[0], names[-1]
    colors = [BEST_COLOR if n == best else WORST_COLOR if n == worst else OTHER_COLOR
              for n in names]
    positions = list(range(len(names)))[::-1]

    fig, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # The BEST tag goes only on the R2 panel: on MAE the top three models
    # are within 0.3 units, so no single one is clearly best there.
    panels = [
        (ax_r2, ordered["CV R2 mean"].to_numpy(), "{:.3f}", True,
         "Accuracy (R²): higher is better", "R² (1.0 = perfect prediction)"),
        (ax_mae, ordered["CV MAE mean"].to_numpy(), "{:.1f}", False,
         "Average error (MAE): lower is better", "Average error in progression units"),
    ]
    for ax, values, fmt, tag_best, title, xlabel in panels:
        ax.barh(positions, values, height=0.6, color=colors)
        ax.axvline(0, color=MUTED_INK, linewidth=1)
        span = max(values.max(), 0) - min(values.min(), 0)
        for name, pos, value in zip(names, positions, values):
            tag = ("  BEST" if name == best and tag_best
                   else "  WORST" if name == worst else "")
            ax.text(max(value, 0) + 0.02 * span, pos, fmt.format(value) + tag,
                    va="center", ha="left", fontsize=11, color=INK,
                    fontweight="bold" if tag else "normal")
        ax.set_xlim(min(values.min(), 0) - 0.05 * span, max(values.max(), 0) + 0.35 * span)
        _style_axes(ax)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel(xlabel, color=MUTED_INK)

    ax_r2.set_yticks(positions)
    ax_r2.set_yticklabels(names)
    fig.suptitle(f"Which model predicts disease progression after one year best?  "
                 f"{best} is best, {worst} is worst",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK)
    top3 = ordered.iloc[:3]
    fig.text(0.02, 0.01,
             f"Note: {', '.join(top3['Model'])} are almost tied "
             f"(R² {top3['CV R2 mean'].min():.3f}-{top3['CV R2 mean'].max():.3f}, "
             f"error {top3['CV MAE mean'].min():.1f}-{top3['CV MAE mean'].max():.1f}). "
             "Scores are averages over 5-fold cross-validation.",
             ha="left", fontsize=10, color=MUTED_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "08_model_summary.png", dpi=120)
    plt.close(fig)


def plot_best_predicted_vs_actual(name, y_test, y_pred, r2):
    """Scatter plot: best model's test-set predictions against true values."""
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
    ax.set_title(f"Best model ({name}): Predicted vs Actual, test set (R² = {r2:.3f})",
                 fontsize=12, fontweight="bold", color=INK)
    ax.set_xlabel("Actual disease progression after one year")
    ax.set_ylabel("Predicted disease progression after one year")
    ax.legend(frameon=False)
    _save(fig, "07_best_model_predicted_vs_actual.png")
