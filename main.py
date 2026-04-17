import argparse
import os
import shutil

import kagglehub
import pandas as pd
import wandb

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from src import load_or_preprocess
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
    args = parse_args()

    os.makedirs("./data", exist_ok=True)
    file_path: str = "./data/diabetes_012_health_indicators_BRFSS2015.csv"
    if not os.path.exists(file_path):
        path: str = kagglehub.dataset_download(
            "alexteboul/diabetes-health-indicators-dataset"
        )
        src: str = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
        shutil.move(src, file_path)
    df: pd.DataFrame = pd.read_csv(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = (
        load_or_preprocess.load_or_preprocess(
            df, SOURCE, TARGET, POSITIVE_VALUE, BINARY_COLS, INT_COLS, OUTLIER_COLS
        )
    )

    run = wandb.init(
        entity="matteo-heidelberger-cesi",
        project="diabetes-prediction",
        config={
            "model_type": args.model_type,
            "dropout_rate": args.dropout_rate,
            "l2_lambda": args.l2_lambda,
            "imbalance_method": args.imbalance_method,
            "batch_size": 64,
        },
    )

    model, history = create_model(
        X_train,
        X_val,
        y_train,
        y_val,
        args.model_type,
        "best-model.keras",
        args.imbalance_method,
        dropout_rate=args.dropout_rate,
        l2_lambda=args.l2_lambda,
    )

    run.log(
        {
            "val_loss": min(history.history["val_loss"]),
            "val_recall": max(history.history["val_recall"]),
            "val_precision": max(history.history["val_precision"]),
            "val_auc": max(history.history["val_auc"]),
        }
    )

    run.finish()

    if args.evaluate:
        evaluate(X_test, y_test, threshold=args.threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Diabetes Prediction Model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="base",
        choices=["base", "dropout", "regularization", "complete", "advanced"],
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
        default=0.35,
        help="Decision threshold of the classification",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training",
    )
    parser.add_argument(
        "--imbalance-method",
        type=str,
        default="class_weight",
        choices=["class_weight", "smote"],
        help="Choose between class_weight and SMOTE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
