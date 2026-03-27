from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def display_info(dataset: pd.DataFrame, target: str) -> None:
    print("=" * 25 + " Table " + "=" * 25)
    print(dataset.head())
    print("=" * 25 + " Summary " + "=" * 25)
    print(dataset.info())
    print("=" * 25 + " Statistical Summary" + "=" * 25)
    print(dataset.describe())
    print("=" * 25 + f" Correlation with {target} (Ranked) " + "=" * 25)
    print_target_correlation(dataset, target)


def display_visualization(dataset: pd.DataFrame, target: str) -> None:
    print("=" * 25 + " Correlation Heatmap " + "=" * 25)
    plot_correlation_heatmap(dataset)
    print("=" * 25 + " Target Variable Distribution " + "=" * 25)
    plot_target_distribution(dataset, target)


def plot_correlation_heatmap(dataset: pd.DataFrame) -> None:
    corr = dataset.corr()
    plt.figure(dpi=130)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()


def print_target_correlation(dataset: pd.DataFrame, target: str) -> None:
    corr = dataset.corr()
    print(corr[target].sort_values(ascending=False))


def plot_target_distribution(dataset: pd.DataFrame, target: str) -> None:
    plt.pie(
        dataset[target].value_counts(),
        labels=[f"{target}", f"Not {target}"],
        autopct="%.f%%",
        shadow=True,
    )
    plt.title("Outcome Proportionality")
    plt.show()
    print(
        "The target variable is not balanced as you can see with this pie chart. This will affect the model training and evaluation."
    )


def visualize_outliers(dataset: pd.DataFrame) -> None:
    numerical: List[str] = ["BMI", "Age"]

    _, axs = plt.subplots(len(numerical), 1, figsize=(7, len(numerical) * 1.5), dpi=95)
    axs = axs.flatten()

    for i, col in enumerate(numerical):
        axs[i].boxplot(dataset[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()
