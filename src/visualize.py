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
