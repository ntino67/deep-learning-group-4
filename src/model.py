import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Input
from keras.metrics import Precision, Recall
from sklearn.utils.class_weight import compute_class_weight


def create_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
) -> None:
    nb_input = X_train.shape[1]

    model = Sequential(
        [
            Input(shape=(nb_input,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.3),
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

    # compute class weight due in case the target feauture is imbalanced
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight,
    )
