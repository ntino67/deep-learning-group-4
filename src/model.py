import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input
from keras.metrics import Precision, Recall


def create_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    nb_input = X_train.shape[1]

    model = Sequential(
        [
            Input(shape=(nb_input,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", Recall(), Precision()],
    )

    # Stop early if val_loss doesn't improve for 10 consecutive epochs
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        "best-model.keras", monitor="val_loss", save_best_only=True, verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
    )
