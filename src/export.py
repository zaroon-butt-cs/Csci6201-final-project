"""Export the selected model so the web app can make the same predictions.

The web app runs in the browser, so it cannot call scikit-learn. A scaled
linear model is only a formula, so we save its numbers to a JSON file:

    z_i        = (value_i - mean_i) / scale_i          (StandardScaler)
    prediction = intercept + sum(coefficient_i * z_i)  (linear model)

The file also holds dataset statistics the app uses for its input ranges
and for putting a prediction into context.
"""

import json

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, PROJECT_ROOT, RANDOM_STATE

MODEL_DIR = OUTPUT_DIR / "model"
WEB_DATA_DIR = PROJECT_ROOT / "web" / "src" / "data"


def _comparison_rows(results, selected, tied):
    """The comparison table as plain values, for the app's project-info panel."""
    rows = []
    for _, r in results.iterrows():
        rows.append({
            "model": r["Model"],
            "cv_r2": float(r["CV R2 mean"]),
            "cv_r2_std": float(r["CV R2 std"]),
            "cv_mae": float(r["CV MAE mean"]),
            "cv_rmse": float(r["CV RMSE mean"]),
            "p_vs_top": None if pd.isna(r["p vs top"]) else float(r["p vs top"]),
            "test_r2": float(r["Test R2"]),
            "test_mae": float(r["Test MAE"]),
            "test_rmse": float(r["Test RMSE"]),
            "settings": r["Tuned settings"],
            "tied": r["Model"] in tied,
            "selected": r["Model"] == selected,
        })
    return rows


def export_model(name, tuned_model, X, y, results, tied, protocol):
    """Fit the selected model on ALL samples and save it as JSON.

    Model selection is finished at this point, so the final model is
    trained on every available patient (standard practice).

    Args:
        name: model name, e.g. "Linear Regression".
        tuned_model: unfitted pipeline or GridSearchCV for that model.
        X, y: the full dataset.
        results: the comparison table (one row per model).
        tied: names of the models statistically tied with the top model.
        protocol: how the comparison was run (folds, repeats, split sizes).

    Returns:
        The fitted pipeline and the list of files written.
    """
    cv_row = results.set_index("Model").loc[name]
    tuned_model.fit(X, y)
    pipeline = getattr(tuned_model, "best_estimator_", tuned_model)
    if not hasattr(pipeline, "steps") or not hasattr(pipeline[-1], "coef_"):
        raise NotImplementedError(
            f"{name} is not a scaled linear model, so it cannot be exported as a formula."
        )
    scaler, linear = pipeline[0], pipeline[-1]
    predictions = pipeline.predict(X)

    # Three real patients (low / typical / high prediction) as example inputs
    examples = []
    for label, q in [("Low", 10), ("Typical", 50), ("High", 90)]:
        row = int(np.abs(predictions - np.percentile(predictions, q)).argmin())
        examples.append({
            "label": label,
            "values": {col: float(X.iloc[row][col]) for col in X.columns},
            "actual": float(y.iloc[row]),
        })

    payload = {
        "model": name,
        "trained_on": int(len(X)),
        "random_state": RANDOM_STATE,
        "settings": getattr(tuned_model, "best_params_", {}),
        "intercept": float(linear.intercept_),
        "features": [
            {
                "key": col,
                "mean": float(scaler.mean_[i]),
                "scale": float(scaler.scale_[i]),
                "coefficient": float(linear.coef_[i]),
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "median": float(X[col].median()),
            }
            for i, col in enumerate(X.columns)
        ],
        "performance": {
            "cv_r2": float(cv_row["CV R2 mean"]),
            "cv_mae": float(cv_row["CV MAE mean"]),
            "cv_rmse": float(cv_row["CV RMSE mean"]),
        },
        "target": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            # value at the 0th, 1st, ..., 100th percentile of the 442 patients
            "percentiles": [float(v) for v in np.percentile(y, range(101))],
        },
        "examples": examples,
        "protocol": protocol,
        "comparison": _comparison_rows(results, name, tied),
    }
    text = json.dumps(payload, indent=2)

    written = []
    for folder in (MODEL_DIR, WEB_DATA_DIR):
        if folder is WEB_DATA_DIR and not folder.parent.exists():
            continue  # the web app has not been created yet
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model.json").write_text(text, encoding="utf-8")
        written.append(folder / "model.json")
    return pipeline, written
