import os
import shutil

import kagglehub
import pandas as pd

from src import preprocessing
from src.config import (
    BINARY_COLS,
    INT_COLS,
    OUTLIER_COL,
    POSITIVE_VALUE,
    SOURCE,
    TARGET,
)


def main() -> None:
    os.makedirs("./data", exist_ok=True)
    file_path: str = "./data/diabetes_012_health_indicators_BRFSS2015.csv"
    if not os.path.exists(file_path):
        path: str = kagglehub.dataset_download(
            "alexteboul/diabetes-health-indicators-dataset"
        )
        src: str = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
        shutil.move(src, file_path)
    df: pd.DataFrame = pd.read_csv(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.preprocessing(
        df, SOURCE, TARGET, POSITIVE_VALUE, BINARY_COLS, INT_COLS, OUTLIER_COL
    )


if __name__ == "__main__":
    main()
