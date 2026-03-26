import os
import shutil

import kagglehub
import pandas as pd

from src import preprocessing


def main() -> None:
    os.makedirs("./data", exist_ok=True)
    file_path = "./data/diabetes_012_health_indicators_BRFSS2015.csv"
    if not os.path.exists(file_path):
        path = kagglehub.dataset_download(
            "alexteboul/diabetes-health-indicators-dataset"
        )
        src = os.path.join(path, "diabetes_012_health_indicators_BRFSS2015.csv")
        shutil.move(src, file_path)
    df = pd.read_csv(file_path)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.preprocessing(df)


if __name__ == "__main__":
    main()
