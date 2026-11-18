"""Final Project step: compare Linear, Ridge, Lasso, Decision Tree and Random Forest.

Usage:
    python compare_models.py
"""

from src.compare import compare_models, report_comparison
from src.data_loader import load_data, split_features_target
from src.model import evaluate_model, split_data
from src.visualize import (
    plot_best_predicted_vs_actual,
    plot_model_mae,
    plot_model_r2,
    plot_model_summary,
)


def main():
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results, fitted = compare_models(X, y, X_train, X_test, y_train, y_test)
    report_comparison(results)

    # Figures: R2 and MAE per model, then the best model's predictions
    plot_model_r2(results)
    plot_model_mae(results)
    plot_model_summary(results)

    best_name = results.iloc[0]["Model"]
    y_pred = fitted[best_name].predict(X_test)
    r2 = evaluate_model(y_test, y_pred)["R2"]
    plot_best_predicted_vs_actual(best_name, y_test, y_pred, r2)

    print("\nFigures saved to outputs/figures/ (05-08)")


if __name__ == "__main__":
    main()
