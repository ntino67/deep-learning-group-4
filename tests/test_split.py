import numpy as np
import pandas as pd
import pytest

from src.preprocessing import split_dataset


def make_df(n=1000):
    np.random.seed(42)
    return pd.DataFrame({
        "Feature1": np.random.randn(n),
        "Feature2": np.random.randn(n),
        "Diabetes": np.random.choice([0.0, 1.0], n),
    })


def test_returns_six_splits():
    result = split_dataset(make_df(), "Diabetes")
    assert len(result) == 6


def test_correct_split_sizes():
    df = make_df(1000)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, "Diabetes")
    assert X_train.shape[0] == 800
    assert X_val.shape[0] == 100
    assert X_test.shape[0] == 100


def test_no_overlap_between_splits():
    df = make_df(1000)
    X_train, X_val, X_test, _, _, _ = split_dataset(df, "Diabetes")
    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


def test_target_column_not_in_X():
    X_train, X_val, X_test, _, _, _ = split_dataset(make_df(), "Diabetes")
    assert "Diabetes" not in X_train.columns
    assert "Diabetes" not in X_val.columns
    assert "Diabetes" not in X_test.columns


def test_X_and_y_same_length():
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(make_df(), "Diabetes")
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]


def test_reproducible_with_random_state():
    df = make_df()
    result1 = split_dataset(df, "Diabetes")
    result2 = split_dataset(df, "Diabetes")
    pd.testing.assert_frame_equal(result1[0], result2[0])
