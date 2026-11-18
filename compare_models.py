"""Final Project: compare Linear, Ridge, Lasso, Decision Tree and Random Forest,
select the best model and export it for the web app.

Usage:
    python compare_models.py        (takes about 5 minutes)
"""

from src.compare import (
    CV_FOLDS,
    CV_REPEATS,
    SIGNIFICANCE_LEVEL,
    compare_models,
    get_models,
    make_tuned,
    report_comparison,
    select_model,
)
from src.config import PROJECT_ROOT, RANDOM_STATE, TEST_SIZE
from src.data_loader import load_data, split_features_target
from src.export import export_model
from src.model import evaluate_model, split_data
from src.visualize import (
    plot_model_r2,
    plot_model_summary,
    plot_selected_predicted_vs_actual,
)


def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 1. Evaluate every model under the same conditions
    print("Comparing models (tuning + 50 cross-validation rounds each):")
    results, fold_r2, fitted = compare_models(X, y, X_train, X_test, y_train, y_test)

    # 2. Apply the selection rule and report
    selected, tied = select_model(results)
    print()
    report_comparison(results, fold_r2, selected, tied)

    # 3. Figures
    plot_model_r2(results, fold_r2, selected, tied)
    plot_model_summary(results, selected, tied)
    y_pred = fitted[selected].predict(X_test)
    r2 = evaluate_model(y_test, y_pred)["R2"]
    plot_selected_predicted_vs_actual(selected, y_test, y_pred, r2)

    # 4. Train the selected model on all data and export it for the web app
    protocol = {
        "cv_folds": CV_FOLDS,
        "cv_repeats": CV_REPEATS,
        "significance_level": SIGNIFICANCE_LEVEL,
        "test_size": TEST_SIZE,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "random_state": RANDOM_STATE,
    }
    _, written = export_model(selected, make_tuned(*get_models()[selected]), X, y,
                              results, tied, protocol)

    print("\nFigures saved to outputs/figures/ (05, 06, 07)")
    for path in written:
        print(f"Model exported to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
