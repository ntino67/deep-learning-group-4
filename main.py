import argparse
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
from src.final_eval import evaluate
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

    args = parse_args()
    model, history = create_model(
        X_train,
        X_val,
        y_train,
        y_val,
        args.model_type,
        "best-model.keras",
        dropout_rate=args.dropout_rate,
        l2_lambda=args.l2_lambda,
    )

    if args.evaluate:
        evaluate(X_test, y_test, threshold=args.threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Prediction Model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "dropout", "regularization", "complete"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate (used with dropout and complete models)",
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.001,
        help="L2 regularization lambda (used with regularization and complete models)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        defautl=0.35,
        help="Decision threshold of the classification",
    )
    parser.add_argument(
        "--evaluate",
        type=bool,
        default=False,
        help="Decide if you display the evaluation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
