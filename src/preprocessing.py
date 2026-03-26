from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def preprocessing(dataset: pd.DataFrame) -> None:
    bin_diabetes = dataset["Diabetes_012"].map(lambda x: 1.0 if x == 2.0 else 0.0)
    dataset["Diabetes"] = bin_diabetes
    dataset.drop(columns=["Diabetes_012"], inplace=True)
    display_info(dataset)


def display_info(dataset: pd.DataFrame) -> None:
    print("=" * 25 + " Table " + "=" * 25)
    print(dataset.head())
    print("=" * 25 + " Summary " + "=" * 25)
    print(dataset.info())
    print("=" * 25 + " Null values " + "=" * 25)
    print(dataset.isnull().sum())
    print("=" * 25 + " Statistical Summary" + "=" * 25)
    print(dataset.describe())
    print("=" * 25 + " Visualize Outliers " + "=" * 25)
    visualize_outliers(dataset)
    print("(Plot)")


def visualize_outliers(dataset: pd.DataFrame) -> None:
    numerical: List[str] = ["BMI", "Age"]

    fig, axs = plt.subplots(
        len(numerical), 1, figsize=(7, len(numerical) * 1.5), dpi=95
    )
    axs = axs.flatten()

    for i, col in enumerate(numerical):
        if col in numerical:
            axs[i].boxplot(dataset[col], vert=False)
            axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()
