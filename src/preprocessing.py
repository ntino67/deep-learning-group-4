from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.cleaning import cleaning_dataset
from src.eda import display_info, display_visualization, visualize_outliers
from src.utils import _validate


def preprocessing(
    dataset: pd.DataFrame,
    source: str,
    target: str,
    positive_value: float,
    binary_cols: List[str],
    int_cols: List[str],
    outlier_cols: List[str],
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    print("Cleaning the dataset...")
    visualize_outliers(dataset)
    df = cleaning_dataset(
        dataset, source, target, positive_value, binary_cols, int_cols, outlier_cols
    )
    visualize_outliers(df)
    display_info(df, target)
    display_visualization(df, target)

    df.to_csv("./data/cleaned_dataset.csv", index=False)
    print("Saved the cleaned dataset.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, target)

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    print("\nFinished the preprocessing.")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def split_dataset(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, ...]:
    _validate(dataset=df, target=target)

    X = df.drop(columns=[target])
    y = df[target]

    # 80-10-10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)
    X_val = scaler.transform(val)
    X_test = scaler.transform(test)

    print(f"Training set size: {X_train.shape[0]} samples.")
    print(f"Validation set size: {X_val.shape[0]} samples.")
    print(f"Test set size: {X_test.shape[0]} samples.")

    return X_train, X_val, X_test
