from typing import List

import pandas as pd


def _validate(
    dataset: pd.DataFrame = None,
    source: str = None,
    target: str = None,
    positive_value: float = None,
    binary_cols: List[str] = None,
    int_cols: List[str] = None,
    outlier_cols: List[str] = None,
) -> None:
    if dataset is not None and dataset.empty:
        raise ValueError("Dataset is empty.")
    if source is not None and source not in dataset.columns:
        raise ValueError(f"Source column '{source}' not found in dataset.")
    if target is not None and target not in dataset.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    if positive_value is not None and positive_value not in dataset[source].unique():
        raise ValueError(
            f"Positive value '{positive_value}' not found in column '{source}'."
        )
    for col in (binary_cols or []) + (int_cols or []) + (outlier_cols or []):
        if col not in dataset.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")
