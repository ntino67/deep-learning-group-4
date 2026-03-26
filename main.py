import pandas as pd

from src import preprocessing

file_path = "./diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.preprocessing(df)


if __name__ == "__main__":
    main()
