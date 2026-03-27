import numpy as np
import pandas as pd
import pytest

from src.preprocessing import scale_features


def make_splits(n=1000):
    np.random.seed(42)
    train = pd.DataFrame(np.random.randn(n, 3) * 10 + 5, columns=["A", "B", "C"])
    val = pd.DataFrame(np.random.randn(100, 3) * 10 + 5, columns=["A", "B", "C"])
    test = pd.DataFrame(np.random.randn(100, 3) * 10 + 5, columns=["A", "B", "C"])
    return train, val, test


def test_returns_three_arrays():
    result = scale_features(*make_splits())
    assert len(result) == 3


def test_output_is_numpy():
    X_train, X_val, X_test = scale_features(*make_splits())
    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_val, np.ndarray)
    assert isinstance(X_test, np.ndarray)


def test_train_mean_near_zero():
    X_train, _, _ = scale_features(*make_splits())
    assert np.abs(X_train.mean()) < 0.01


def test_train_std_near_one():
    X_train, _, _ = scale_features(*make_splits())
    assert np.abs(X_train.std() - 1.0) < 0.01


def test_val_not_refitted():
    train, val, test = make_splits()
    _, X_val, _ = scale_features(train, val, test)
    # Val mean should NOT be near zero since scaler was fit on train only
    assert np.abs(X_val.mean()) > 0.0


def test_shapes_preserved():
    train, val, test = make_splits()
    X_train, X_val, X_test = scale_features(train, val, test)
    assert X_train.shape == train.shape
    assert X_val.shape == val.shape
    assert X_test.shape == test.shape
