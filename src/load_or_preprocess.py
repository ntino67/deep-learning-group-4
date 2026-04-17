import os

import numpy as np
import pandas as pd

from src import preprocessing

CACHE_PATH = "./data/preprocessed"


def load_or_preprocess(
    df, source, target, positive_value, binary_cols, int_cols, outlier_cols
):
    if os.path.exists(f"{CACHE_PATH}/X_train.npy"):
        print("Loading cached preprocessed data...")
        X_train = np.load(f"{CACHE_PATH}/X_train.npy")
        X_val = np.load(f"{CACHE_PATH}/X_val.npy")
        X_test = np.load(f"{CACHE_PATH}/X_test.npy")
        y_train = pd.read_csv(f"{CACHE_PATH}/y_train.csv").squeeze()
        y_val = pd.read_csv(f"{CACHE_PATH}/y_val.csv").squeeze()
        y_test = pd.read_csv(f"{CACHE_PATH}/y_test.csv").squeeze()
    else:
        os.makedirs(CACHE_PATH)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.preprocessing(
            df, source, target, positive_value, binary_cols, int_cols, outlier_cols
        )
        np.save(f"{CACHE_PATH}/X_train.npy", X_train)
        np.save(f"{CACHE_PATH}/X_val.npy", X_val)
        np.save(f"{CACHE_PATH}/X_test.npy", X_test)
        y_train.to_csv(f"{CACHE_PATH}/y_train.csv", index=False)
        y_val.to_csv(f"{CACHE_PATH}/y_val.csv", index=False)
        y_test.to_csv(f"{CACHE_PATH}/y_test.csv", index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test
