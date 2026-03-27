import pandas as pd
import pytest

from src.cleaning import binarize_target


def make_df(values):
    return pd.DataFrame({"Diabetes_012": values, "OtherCol": [1.0] * len(values)})


def test_positive_value_becomes_one():
    df = make_df([2.0, 0.0, 1.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert list(result["Diabetes"]) == [1.0, 0.0, 0.0]


def test_source_column_is_dropped():
    df = make_df([2.0, 0.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert "Diabetes_012" not in result.columns


def test_target_column_is_created():
    df = make_df([2.0, 0.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert "Diabetes" in result.columns


def test_other_columns_unchanged():
    df = make_df([2.0, 0.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert list(result["OtherCol"]) == [1.0, 1.0]


def test_all_negative_values():
    df = make_df([0.0, 1.0, 0.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert list(result["Diabetes"]) == [0.0, 0.0, 0.0]


def test_all_positive_values():
    df = make_df([2.0, 2.0, 2.0])
    result = binarize_target(df, "Diabetes_012", "Diabetes", 2.0)
    assert list(result["Diabetes"]) == [1.0, 1.0, 1.0]
