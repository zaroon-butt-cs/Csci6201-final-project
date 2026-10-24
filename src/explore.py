"""Exploratory data analysis (EDA) for the Diabetes dataset."""

import pandas as pd

from src.config import RESULTS_DIR, TARGET_COLUMN

# Short descriptions of each feature (from the scikit-learn documentation)
FEATURE_DESCRIPTIONS = {
    "age": "age in years",
    "sex": "sex (coded as 1 or 2)",
    "bmi": "body mass index",
    "bp": "average blood pressure",
    "s1": "tc, total serum cholesterol",
    "s2": "ldl, low-density lipoproteins",
    "s3": "hdl, high-density lipoproteins",
    "s4": "tch, total cholesterol / HDL",
    "s5": "ltg, log of serum triglycerides level",
    "s6": "glu, blood sugar level",
}


def explore_data(df):
    """Print basic information about the dataset and save it to a text file."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    lines = []  # everything we print is also collected here for the report

    def report(text=""):
        print(text)
        lines.append(str(text))

    report("=" * 60)
    report("DATASET OVERVIEW")
    report("=" * 60)
    report(f"Shape of the data (rows, columns): {df.shape}")
    report(f"Number of samples:  {X.shape[0]}")
    report(f"Number of features: {X.shape[1]}")
    report("Target variable: 'target' = disease progression one year after baseline")
    report("Task type: REGRESSION (the target is a continuous number, not a class)")

    report("\nFeature names:")
    for name in X.columns:
        report(f"  {name:<4} - {FEATURE_DESCRIPTIONS.get(name, '')}")

    report("\n" + "=" * 60)
    report("DESCRIPTIVE STATISTICS")
    report("=" * 60)
    report(df.describe().round(2).to_string())

    report("\nMinimum and maximum values:")
    report(pd.DataFrame({"min": df.min(), "max": df.max()}).round(2).to_string())

    report("\nAverages (mean):")
    report(df.mean().round(2).to_string())

    report("\n" + "=" * 60)
    report("MISSING VALUES")
    report("=" * 60)
    report(df.isnull().sum().to_string())
    report(f"Total missing values: {df.isnull().sum().sum()}")

    # "Class counts" are for classification. Our target is continuous, so
    # instead we group it into 3 equal-sized ranges to see how it is spread.
    report("\n" + "=" * 60)
    report("TARGET DISTRIBUTION (grouped into 3 ranges)")
    report("=" * 60)
    target_groups = pd.qcut(y, q=3, labels=["low", "medium", "high"])
    report(target_groups.value_counts().sort_index().to_string())

    report("\n" + "=" * 60)
    report("CORRELATION OF EACH FEATURE WITH THE TARGET")
    report("=" * 60)
    report(get_target_correlations(df).round(3).to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def get_target_correlations(df):
    """Return each feature's correlation with the target, strongest first."""
    correlations = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    return correlations.sort_values(ascending=False)
