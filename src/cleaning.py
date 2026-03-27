from typing import List

import numpy as np
import pandas as pd

from src.utils import _validate


def cleaning_dataset(
    dataset: pd.DataFrame,
    source: str,
    target: str,
    positive_value: float,
    binary_cols: List[str],
    int_cols: List[str],
    outlier_cols: List[str],
) -> pd.DataFrame:
    _validate(
        dataset=dataset,
        source=source,
        positive_value=positive_value,
        int_cols=int_cols,
        outlier_cols=outlier_cols,
    )

    df = binarize_target(dataset, source, target, positive_value)

    _validate(dataset=df, target=target, binary_cols=binary_cols)

    # Two sum because the first sum return a Series (count of null per column), the second one sums up all the nulls per column
    df_null: int = df.isnull().sum().sum()
    print(f"Null values in the dataset: {df_null}")
    if df_null > 0:
        df.dropna(inplace=True)
        print("Dropped rows with missing values.")

    df_dup: int = df.duplicated().sum()
    print(f"Duplicated values in the dataset: {df_dup}")
    if df_dup > 0:
        df.drop_duplicates(inplace=True)
        print("Dropped the duplicated rows.")

    df[binary_cols] = df[binary_cols].astype(bool)
    df[int_cols] = df[int_cols].astype(int)

    print("Removing the outliers...")
    mask = pd.Series(True, index=df.index)
    for outlier in outlier_cols:
        q1, q3 = np.percentile(df[outlier], [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= (df[outlier] >= lower) & (df[outlier] <= upper)
    df = df[mask]

    print("Cleaned dataset.")
    return df


def binarize_target(
    df: pd.DataFrame, source_col: str, target_col: str, positive_value: float
) -> pd.DataFrame:
    _validate(dataset=df, source=source_col)

    df = df.copy()
    df[target_col] = df[source_col].map(lambda x: 1.0 if x == positive_value else 0.0)
    df.drop(columns=[source_col], inplace=True)
    return df
