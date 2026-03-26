from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocessing(
    dataset: pd.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    print("Cleaning the dataset...")
    df = cleaning_dataset(dataset)
    display_info(df)
    display_visualization(df)

    df.to_csv("./data/cleaned_dataset.csv", index=False)
    print("Saved the cleaned dataset.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)

    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    print("\nFinished the preprocessing.")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def cleaning_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    bin_diabetes = df["Diabetes_012"].map(lambda x: 1.0 if x == 2.0 else 0.0)
    df["Diabetes"] = bin_diabetes
    df.drop(columns=["Diabetes_012"], inplace=True)

    # Two sum because the first sum return a Series (count of null per column), the second one sums up all the nulls per column
    df_null: int = df.isnull().sum().sum()
    print(f"Null values in the dataset: {df_null}")
    if df_null > 0:
        df.dropna(inplace=True)
        print("Dropped rows with missing values.")

    df_dup: int = df.duplicated().sum()
    print(f"Duplicated values in the dataset: {df_dup}")
    if df_dup > 0:
        df.drop_duplicates(inplace=True)
        print("Dropped the duplicated rows.")

    binary_cols = [
        "Diabetes",
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    ]
    df[binary_cols] = df[binary_cols].astype(bool)
    int_cols = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    df[int_cols] = df[int_cols].astype(int)

    visualize_outliers(df)
    print("Removing the outliers...")
    q1, q3 = np.percentile(df["BMI"], [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clean_df: pd.DataFrame = df[(df["BMI"] >= lower) & (df["BMI"] <= upper)]
    print("Cleaned dataset.")
    visualize_outliers(clean_df)
    return clean_df


def display_info(dataset: pd.DataFrame) -> None:
    print("=" * 25 + " Table " + "=" * 25)
    print(dataset.head())
    print("=" * 25 + " Summary " + "=" * 25)
    print(dataset.info())
    print("=" * 25 + " Statistical Summary" + "=" * 25)
    print(dataset.describe())
    print("=" * 25 + " Correlation with Diabetes " + "=" * 25)
    print_target_correlation(dataset, "Diabetes")


def display_visualization(dataset: pd.DataFrame) -> None:
    print("=" * 25 + " Correlation Heatmap " + "=" * 25)
    plot_correlation_heatmap(dataset)
    print("=" * 25 + " Target Variable Distribution " + "=" * 25)
    plot_target_distribution(dataset)


def visualize_outliers(dataset: pd.DataFrame) -> None:
    numerical: List[str] = ["BMI", "Age"]

    _, axs = plt.subplots(len(numerical), 1, figsize=(7, len(numerical) * 1.5), dpi=95)
    axs = axs.flatten()

    for i, col in enumerate(numerical):
        axs[i].boxplot(dataset[col], vert=False)
        axs[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(dataset: pd.DataFrame) -> None:
    corr = dataset.corr()
    plt.figure(dpi=130)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()
    print(
        "The correlation heatmap doesn't allow us to remove any column. The correlation between the different parameters are too low."
    )


def print_target_correlation(dataset: pd.DataFrame, target: str) -> None:
    corr = dataset.corr()
    print(corr[target].sort_values(ascending=False))


def plot_target_distribution(dataset: pd.DataFrame) -> None:
    plt.pie(
        dataset["Diabetes"].value_counts(),
        labels=["Diabetes", "Not Diabetes"],
        autopct="%.f%%",
        shadow=True,
    )
    plt.title("Outcome Proportionality")
    plt.show()
    print(
        "The target variable is not balanced as you can see with this pie chart. This will affect the model training and evaluation."
    )


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, ...]:
    X = df.drop(columns=["Diabetes"])
    y = df["Diabetes"]

    # 80-10-10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, ...]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train)
    X_val = scaler.transform(val)
    X_test = scaler.transform(test)

    print(f"Training set: {X_train.shape[0]} samples.")
    print(f"Validation set: {X_val.shape[0]} samples.")
    print(f"Test set: {X_test.shape[0]} samples.")

    return X_train, X_val, X_test
