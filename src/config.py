"""Project-wide settings so every module uses the same values."""

from pathlib import Path

# Folder paths (relative to the project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# Fixed seed so the train/test split (and results) are the same every run
RANDOM_STATE = 42

# 20% of the data is held back for testing the model
TEST_SIZE = 0.2

# Name of the target column in our DataFrame
TARGET_COLUMN = "target"
