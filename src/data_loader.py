
from pathlib import Path

import pandas as pd


# Locate the project directory independently of the terminal location
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"


def load_student_data():
    """Load the student dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    return df


def validate_student_data(df):
    """Perform basic dataset validation."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    if "Target" not in df.columns:
        raise ValueError("Target column is missing.")

    expected_targets = {"Graduate", "Dropout", "Enrolled"}

    actual_targets = set(df["Target"].dropna().unique())

    if not actual_targets.issubset(expected_targets):
        raise ValueError(
            f"Unexpected target categories: {actual_targets}"
        )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df["Target"].value_counts().to_dict()
    }


if __name__ == "__main__":

    data = load_student_data()

    report = validate_student_data(data)

    print("EDUREASON AI - DATA VALIDATION")
    print("-" * 40)

    for key, value in report.items():
        print(f"{key}: {value}")
