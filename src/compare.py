"""Compare several regression models under the same conditions.

Every model gets:
    - the same data (10 features, original units)
    - the same 80/20 train/test split (random_state=42)
    - the same 5-fold cross-validation folds (shuffled, random_state=42)

Linear models (Linear, Ridge, Lasso) are wrapped in a Pipeline with
StandardScaler, so scaling is learned only from the training part of each
split and the coefficients become comparable. Tree models do not need
scaling, so they use the raw features.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE, RESULTS_DIR
from src.model import evaluate_model

CV_FOLDS = 5


def get_models():
    """Return the models to compare, all with default settings."""
    return {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=1.0, random_state=RANDOM_STATE)),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    }


def compare_models(X, y, X_train, X_test, y_train, y_test):
    """Evaluate every model on the test split and with cross-validation.

    Returns:
        results: DataFrame with one row per model, best CV R2 first.
        fitted: dict of model name -> model fitted on the training split.
    """
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"}

    rows, fitted = [], {}
    for name, model in get_models().items():
        # 1. Same single split as the Lab 2 baseline
        model.fit(X_train, y_train)
        test = evaluate_model(y_test, model.predict(X_test))
        fitted[name] = model

        # 2. Cross-validation on all 442 samples (more reliable)
        cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring)

        rows.append({
            "Model": name,
            "Test R2": test["R2"],
            "Test MAE": test["MAE"],
            "Test MSE": test["MSE"],
            "CV R2 mean": cv_scores["test_r2"].mean(),
            "CV R2 std": cv_scores["test_r2"].std(),
            "CV MAE mean": -cv_scores["test_mae"].mean(),
            "CV MSE mean": -cv_scores["test_mse"].mean(),
        })

    results = pd.DataFrame(rows).sort_values("CV R2 mean", ascending=False)
    return results.reset_index(drop=True), fitted


def report_comparison(results):
    """Print the comparison table and save it to a text file."""
    best = results.iloc[0]
    lines = [
        "=" * 78,
        "MODEL COMPARISON (same data, same split, same CV folds)",
        "=" * 78,
        f"Single split: 80% train / 20% test, random_state={RANDOM_STATE}",
        f"Cross-validation: {CV_FOLDS}-fold KFold, shuffle=True, random_state={RANDOM_STATE}",
        "Ranked by mean cross-validated R2 (higher is better).",
        "",
        results.round(3).to_string(index=False),
        "",
        f"Best model: {best['Model']} "
        f"(CV R2 = {best['CV R2 mean']:.3f} +/- {best['CV R2 std']:.3f}, "
        f"CV MAE = {best['CV MAE mean']:.2f})",
    ]
    report = "\n".join(lines)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_comparison.txt").write_text(report, encoding="utf-8")
