from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def preprocessing(dataset: pd.DataFrame) -> None:
    df = cleaning_dataset(dataset)
    display_info(df)

    X = df.drop(columns=["Diabetes"])
    y = df["Diabetes"]


def cleaning_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    bin_diabetes = df["Diabetes_012"].map(lambda x: 1.0 if x == 2.0 else 0.0)
    df["Diabetes"] = bin_diabetes
    df.drop(columns=["Diabetes_012"], inplace=True)

    q1, q3 = np.percentile(df["BMI"], [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df["BMI"] >= lower) & (df["BMI"] <= upper)]


def display_info(dataset: pd.DataFrame) -> None:
    print("=" * 25 + " Table " + "=" * 25)
    print(dataset.head())
    print("=" * 25 + " Summary " + "=" * 25)
    print(dataset.info())
    print("=" * 25 + " Null values " + "=" * 25)
    print(dataset.isnull().sum())
    print("=" * 25 + " Statistical Summary" + "=" * 25)
    print(dataset.describe())


def display_visualization(dataset: pd.DataFrame) -> None:
    print("=" * 25 + " Visualize Outliers " + "=" * 25)
    visualize_outliers(dataset)
    print("(Plot)")
    print("=" * 25 + " Correlation Heatmap " + "=" * 25)
    correlation(dataset)
    print("=" * 25 + " Target Variable Distribution " + "=" * 25)

    plt.pie(
        dataset["Diabetes"].value_counts(),
        labels=["Diabetes", "Not Diabetes"],
        autopct="%.f%%",
        shadow=True,
    )
    plt.title("Outcome Proportionality")
    plt.show()
    print(
        "The target variable is not balanced as you can see with this pie chart. This will affect the model training and evaluation"
    )


def visualize_outliers(dataset: pd.DataFrame) -> None:
    numerical: List[str] = ["BMI", "Age"]

    fig, axs = plt.subplots(
        len(numerical), 1, figsize=(7, len(numerical) * 1.5), dpi=95
    )
    axs = axs.flatten()

    for i, col in enumerate(numerical):
        axs[i].boxplot(dataset[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def correlation(dataset: pd.DataFrame) -> None:
    corr = dataset.corr()
    plt.figure(dpi=130)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

    print(corr["Diabetes"].sort_values(ascending=False))
