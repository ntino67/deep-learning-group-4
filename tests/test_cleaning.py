import numpy as np
import pandas as pd
import pytest

from src.cleaning import cleaning_dataset

BINARY_COLS = [
    "Diabetes",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]
INT_COLS = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]


def make_df(n=100):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Diabetes_012": np.random.choice([0.0, 1.0, 2.0], n),
            "HighBP": np.random.choice([0.0, 1.0], n),
            "HighChol": np.random.choice([0.0, 1.0], n),
            "CholCheck": np.random.choice([0.0, 1.0], n),
            "BMI": np.random.uniform(12, 98, n),
            "Smoker": np.random.choice([0.0, 1.0], n),
            "Stroke": np.random.choice([0.0, 1.0], n),
            "HeartDiseaseorAttack": np.random.choice([0.0, 1.0], n),
            "PhysActivity": np.random.choice([0.0, 1.0], n),
            "Fruits": np.random.choice([0.0, 1.0], n),
            "Veggies": np.random.choice([0.0, 1.0], n),
            "HvyAlcoholConsump": np.random.choice([0.0, 1.0], n),
            "AnyHealthcare": np.random.choice([0.0, 1.0], n),
            "NoDocbcCost": np.random.choice([0.0, 1.0], n),
            "GenHlth": np.random.randint(1, 6, n).astype(float),
            "MentHlth": np.random.randint(0, 31, n).astype(float),
            "PhysHlth": np.random.randint(0, 31, n).astype(float),
            "DiffWalk": np.random.choice([0.0, 1.0], n),
            "Sex": np.random.choice([0.0, 1.0], n),
            "Age": np.random.randint(1, 14, n).astype(float),
            "Education": np.random.randint(1, 7, n).astype(float),
            "Income": np.random.randint(1, 9, n).astype(float),
        }
    )
    return df


def call_cleaning(df):
    return cleaning_dataset(
        df, "Diabetes_012", "Diabetes", 2.0, BINARY_COLS, INT_COLS, ["BMI"]
    )


def test_source_column_dropped():
    result = call_cleaning(make_df())
    assert "Diabetes_012" not in result.columns


def test_target_column_created():
    result = call_cleaning(make_df())
    assert "Diabetes" in result.columns


def test_binary_cols_are_bool():
    result = call_cleaning(make_df())
    for col in BINARY_COLS:
        assert result[col].dtype == bool, f"{col} is not bool"


def test_int_cols_are_int():
    result = call_cleaning(make_df())
    for col in INT_COLS:
        assert result[col].dtype == int, f"{col} is not int"


def test_no_duplicates():
    df = make_df()
    df = pd.concat([df, df.iloc[:10]])  # add duplicates
    result = call_cleaning(df)
    assert result.duplicated().sum() == 0


def test_no_null_values():
    df = make_df()
    df.loc[0, "BMI"] = np.nan
    result = call_cleaning(df)
    assert result.isnull().sum().sum() == 0


def test_bmi_outliers_removed():
    result = call_cleaning(make_df())
    q1, q3 = np.percentile(result["BMI"], [25, 75])
    iqr = q3 - q1
    assert result["BMI"].min() >= q1 - 1.5 * iqr
    assert result["BMI"].max() <= q3 + 1.5 * iqr


def test_original_dataset_not_mutated():
    df = make_df()
    original = df.copy()
    call_cleaning(df)
    pd.testing.assert_frame_equal(df, original)
