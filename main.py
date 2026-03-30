import os
import shutil

import kagglehub
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from src import preprocessing
from src.config import (
    BINARY_COLS,
    INT_COLS,
    OUTLIER_COLS,
    POSITIVE_VALUE,
    SOURCE,
    TARGET,
)
from src.model import create_model


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
        df, SOURCE, TARGET, POSITIVE_VALUE, BINARY_COLS, INT_COLS, OUTLIER_COLS
    )

    # Here a CLI part that allows us to choose the model type, dropout_rate or l2_lambda

    create_model(
        X_train,
        X_val,
        y_train,
        y_val,
        model_type,
        "best-model.keras",
        dropout_rate=dropout_rate,
        l2_lambda=l2_lambda,
    )


if __name__ == "__main__":
    main()
