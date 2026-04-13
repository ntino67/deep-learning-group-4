import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from src import preprocessing
from src.config import *


def evaluate(X_test_scaled: np.ndarray,
             y_test: pd.Series,
             threshold: float = 0.35)
    # mvl stands for model version list. we put the names of the three saved keras files into this list so the script can just cycle through them automatically. this is way better than writing the same code three times.
    models = ["model_base.keras", "model_dropout.keras", "model_complete.keras"]

    # rge stands for running global evaluation. this is the main loop. if it finds the file it uses the keras load model function to bring the neural network back to life. then it runs the predict function which gives us a bunch of decimals between zero and one. these decimals represent the probability of someone having diabetes.
    print("group 4 final evaluation results\n")
    plt.figure(figsize=(10, 8))

    for m_name in models:
        if os.path.exists(m_name):
            # loading the model takes some time because tensorflow has to reconstruct the layers and the weights.
            current_model = tf.keras.models.load_model(m_name)
            # we use verbose zero here so the screen doesnt get flooded with progress bars while it predicts.
            y_probs = current_model.predict(X_test_scaled, verbose=0)

            # auc stands for area under the curve. a score of 0.5 means 50/50 and 1.0 means it is perfect.
            auc_score = roc_auc_score(y_test, y_probs)
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            plt.plot(fpr, tpr, label=f"{m_name} (auc = {auc_score:.4f})")
            print(f"\narchitecture variant: {m_name}")
            print(f"final test roc-auc: {auc_score:.4f}")

            # cr stands for classification report. we still want to see the precision and recall numbers. we are using a standard threshold of point five to turn the probabilities into hard zero or one categories.
            y_pred = (y_probs > threshold).astype(int)
            print(
                classification_report(
                    y_test, y_pred, target_names=["no diabetes", "diabetes"]
                )
            )
        else:
            print(
                f"\nmissing file error: could not find the file named {m_name} in this folder"
            )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("sprint 2 - model comparison (roc curves)")
    plt.legend()
    plt.savefig("sprint2_roc_comparison.png")
    print("\nsaved as 'sprint2_roc_comparison.png'")
