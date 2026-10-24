"""CSCI 6201 Lab 2 - Diabetes Disease Progression (Regression).

Runs the full workflow:
    data -> features and target -> model -> predictions -> evaluation

Usage:
    python analysis.py
"""

from src.data_loader import load_data, split_features_target
from src.explore import explore_data
from src.model import evaluate_model, report_results, split_data, train_model
from src.visualize import (
    plot_bmi_vs_target,
    plot_feature_correlations,
    plot_predicted_vs_actual,
    plot_target_histogram,
)


def main():
    # 1. Load the data
    df = load_data()

    # 2. Explore the data
    explore_data(df)

    # 3. Visualize the data
    plot_target_histogram(df)
    plot_feature_correlations(df)
    plot_bmi_vs_target(df)

    # 4. Separate features (X) and target (y), then split into train/test
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Train the model and make predictions on unseen test data
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    # 6. Evaluate the predictions
    metrics = evaluate_model(y_test, y_pred)
    print()
    report_results(model, X.columns, metrics, len(X_train), len(X_test))
    plot_predicted_vs_actual(y_test, y_pred)

    print("\nDone. Figures saved to outputs/figures/, results saved to outputs/results/")


if __name__ == "__main__":
    main()
