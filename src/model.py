"""Train and evaluate a basic Linear Regression model."""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, RESULTS_DIR, TEST_SIZE


def split_data(X, y):
    """Split into training data (to learn from) and test data (to evaluate on)."""
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_model(X_train, y_train):
    """Fit a Linear Regression model on the training data.

    Linear Regression learns one weight (coefficient) per feature so that
    prediction = intercept + sum(coefficient * feature value).
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_test, y_pred):
    """Compare predictions to the true values using three regression metrics."""
    return {
        "MSE": mean_squared_error(y_test, y_pred),   # average squared error
        "MAE": mean_absolute_error(y_test, y_pred),  # average error in target units
        "R2": r2_score(y_test, y_pred),              # share of variation explained (1.0 = perfect)
    }


def report_results(model, feature_names, metrics, train_size, test_size):
    """Print the metrics and coefficients and save them to a text file."""
    coefficients = pd.Series(model.coef_, index=feature_names).sort_values(
        key=abs, ascending=False
    )

    lines = [
        "=" * 60,
        "LINEAR REGRESSION RESULTS",
        "=" * 60,
        f"Training samples: {train_size}",
        f"Test samples:     {test_size}",
        "",
        f"Mean Squared Error (MSE):  {metrics['MSE']:.2f}",
        f"Mean Absolute Error (MAE): {metrics['MAE']:.2f}",
        f"R-squared (R2):            {metrics['R2']:.3f}",
        "",
        f"Intercept: {model.intercept_:.2f}",
        "Coefficients (largest effect first; features are in different units):",
        coefficients.round(3).to_string(),
    ]
    report = "\n".join(lines)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.txt").write_text(report, encoding="utf-8")
