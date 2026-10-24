"""Load the scikit-learn Diabetes dataset."""

from sklearn.datasets import load_diabetes

from src.config import TARGET_COLUMN


def load_data():
    """Load the Diabetes dataset as a pandas DataFrame.

    scaled=False gives the original units (age in years, BMI, blood
    pressure, ...) instead of the mean-centered/scaled version, which
    makes the numbers much easier to read and explain.

    Returns:
        df: DataFrame with the 10 features plus a 'target' column.
    """
    dataset = load_diabetes(as_frame=True, scaled=False)
    df = dataset.frame  # features + target in one table
    return df


def split_features_target(df):
    """Split the DataFrame into X (features) and y (target).

    X = the 10 baseline measurements of each patient (inputs).
    y = disease progression one year after baseline (what we predict).
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
