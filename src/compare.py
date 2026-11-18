"""Compare several regression models fairly and pick the best one.

Protocol (every model gets exactly the same treatment):

1. Tuning. Each model's settings (hyperparameters) are chosen by a grid
   search with 5-fold cross-validation. Comparing untuned models would be
   unfair, e.g. a Decision Tree with no depth limit simply memorises the
   training data.

2. Repeated nested cross-validation on all 442 samples. The data is split
   into 5 folds, 10 times with different shuffles (50 train/test rounds).
   In each round the grid search runs on the training part only, and the
   tuned model is scored on the part it has never seen. All models use the
   same 50 splits, so their scores can be compared pair by pair.

3. Significance test. The corrected resampled t-test (Nadeau & Bengio,
   2003) checks whether the difference between a model and the top model
   is real or within the noise of the splits.

4. Selection rule (fixed before looking at the results). Take the model
   with the highest mean R2. Every model that is NOT significantly worse
   (p >= 0.05) counts as tied with it. Among tied models, choose the
   simplest one.

5. Confirmation. Each model is also tuned on the Lab 2 training split
   (80%) and scored once on the held-out test split (20%).

Linear models are wrapped in a Pipeline with StandardScaler, so scaling is
learned from training data only. Tree models do not need scaling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import RANDOM_STATE, RESULTS_DIR
from src.model import evaluate_model

CV_FOLDS = 5
CV_REPEATS = 10
SIGNIFICANCE_LEVEL = 0.05

# From simplest to most complex. Used only to break statistical ties.
SIMPLICITY_ORDER = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]


def get_models():
    """Return {name: (estimator, hyperparameter grid or None)}."""
    return {
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), None),
        "Ridge": (
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": np.logspace(-2, 3, 26)},
        ),
        "Lasso": (
            make_pipeline(StandardScaler(), Lasso(max_iter=10000)),
            {"lasso__alpha": np.logspace(-2, 1.5, 22)},
        ),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=RANDOM_STATE),
            {"max_depth": [2, 3, 4, 5, 6, None], "min_samples_leaf": [1, 5, 10, 20, 40]},
        ),
        "Random Forest": (
            RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
            {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5, 10], "max_features": [0.33, 1.0]},
        ),
    }


def make_tuned(estimator, grid):
    """Wrap a model in a grid search (or return it unchanged if nothing to tune)."""
    if grid is None:
        return clone(estimator)
    inner_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(clone(estimator), grid, cv=inner_cv, scoring="r2")


def corrected_ttest(diff, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio) for repeated k-fold scores.

    A normal paired t-test is too optimistic here because the training
    sets of different folds overlap. The correction widens the variance.
    Returns the two-sided p-value for "mean difference = 0".
    """
    n = len(diff)
    variance = np.var(diff, ddof=1)
    if variance == 0:
        return 1.0
    t = diff.mean() / np.sqrt(variance * (1 / n + n_test / n_train))
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def compare_models(X, y, X_train, X_test, y_train, y_test):
    """Run the full protocol.

    Returns:
        results: one row per model, best mean CV R2 first.
        fold_r2: DataFrame of the 50 R2 scores per model (same splits).
        fitted: {name: model tuned and fitted on the training split}.
    """
    outer_cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    rows, fold_r2, fitted = [], {}, {}
    for name, (estimator, grid) in get_models().items():
        print(f"  evaluating {name} ...", flush=True)

        # Nested CV: tuning happens inside each of the 50 training parts
        scores = cross_validate(make_tuned(estimator, grid), X, y,
                                cv=outer_cv, scoring=scoring, n_jobs=-1)
        fold_r2[name] = scores["test_r2"]

        # Confirmation on the held-out test split (tuned on train only)
        model = make_tuned(estimator, grid).fit(X_train, y_train)
        fitted[name] = model
        test = evaluate_model(y_test, model.predict(X_test))
        params = getattr(model, "best_params_", {})

        rows.append({
            "Model": name,
            "CV R2 mean": scores["test_r2"].mean(),
            "CV R2 std": scores["test_r2"].std(ddof=1),
            "CV MAE mean": -scores["test_mae"].mean(),
            "CV RMSE mean": -scores["test_rmse"].mean(),
            "Test R2": test["R2"],
            "Test MAE": test["MAE"],
            "Test RMSE": np.sqrt(test["MSE"]),
            "Tuned settings": ", ".join(
                f"{k.split('__')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('__')[-1]}={v}"
                for k, v in params.items()) or "(nothing to tune)",
        })

    results = pd.DataFrame(rows).sort_values("CV R2 mean", ascending=False).reset_index(drop=True)
    fold_r2 = pd.DataFrame(fold_r2)

    # Compare every model with the top one on the same 50 splits
    top = results.loc[0, "Model"]
    n_test = len(X) // CV_FOLDS
    n_train = len(X) - n_test
    p_values, win_rates = [], []
    for name in results["Model"]:
        if name == top:
            p_values.append(np.nan)
            win_rates.append(np.nan)
            continue
        diff = fold_r2[top] - fold_r2[name]
        p_values.append(corrected_ttest(diff.to_numpy(), n_train, n_test))
        win_rates.append(100 * (diff > 0).mean())
    results["p vs top"] = p_values
    results["Top wins %"] = win_rates
    return results, fold_r2, fitted


def select_model(results):
    """Apply the selection rule. Returns (selected name, list of tied names)."""
    tied = results.loc[
        results["p vs top"].isna() | (results["p vs top"] >= SIGNIFICANCE_LEVEL), "Model"
    ].tolist()
    selected = min(tied, key=SIMPLICITY_ORDER.index)
    return selected, tied


def report_comparison(results, fold_r2, selected, tied):
    """Print the comparison and save it (plus the raw fold scores) to files."""
    top = results.loc[0, "Model"]
    worse = [m for m in results["Model"] if m not in tied]
    n_rounds = CV_FOLDS * CV_REPEATS

    cv_table = results[["Model", "CV R2 mean", "CV R2 std", "CV MAE mean",
                        "CV RMSE mean", "p vs top", "Top wins %"]].copy()
    cv_table["p vs top"] = cv_table["p vs top"].map(
        lambda p: "(top)" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}"))
    cv_table["Top wins %"] = cv_table["Top wins %"].map(
        lambda w: "-" if pd.isna(w) else f"{w:.0f}%")
    test_table = results[["Model", "Test R2", "Test MAE", "Test RMSE", "Tuned settings"]]

    lines = [
        "=" * 92,
        "MODEL COMPARISON (tuned models, same data, same splits)",
        "=" * 92,
        f"1) Repeated nested cross-validation: {CV_FOLDS} folds x {CV_REPEATS} repeats "
        f"= {n_rounds} rounds, all 442 samples, random_state={RANDOM_STATE}",
        "   Each model is tuned inside every round, then scored on unseen data.",
        "   R2: higher is better.  MAE / RMSE: lower is better.",
        f"   'p vs top'   = corrected resampled t-test against {top} "
        f"(p < {SIGNIFICANCE_LEVEL} means really worse).",
        f"   'Top wins %' = share of the {n_rounds} rounds where {top} beat this model.",
        "",
        cv_table.round(3).to_string(index=False),
        "",
        "2) Confirmation on the held-out test split (80% train / 20% test, "
        f"random_state={RANDOM_STATE})",
        "",
        test_table.round(3).to_string(index=False),
        "",
        "=" * 92,
        "CONCLUSION",
        "=" * 92,
        f"Highest mean R2: {top} ({results.loc[0, 'CV R2 mean']:.3f}).",
        f"Statistically tied with it (p >= {SIGNIFICANCE_LEVEL}): {', '.join(tied)}.",
        f"Significantly worse (p < {SIGNIFICANCE_LEVEL}): {', '.join(worse) if worse else 'none'}.",
        f"Selected model (simplest of the tied models): {selected}.",
    ]
    report = "\n".join(lines)
    print(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_comparison.txt").write_text(report, encoding="utf-8")
    fold_r2.round(4).to_csv(RESULTS_DIR / "cv_fold_r2_scores.csv", index_label="round")
    results.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
