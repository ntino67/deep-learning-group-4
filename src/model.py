from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from keras import Model, Sequential, callbacks, regularizers
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, Dropout, Input
from keras.metrics import Precision, Recall, AUC
from sklearn.utils.class_weight import compute_class_weight


def create_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    model_type: str,
    checkpoint_path: str,
    dropout_rate: float = 0.3,
    l2_lambda: float = 0.001,
) -> Tuple[Model, History]:
    input_dim = X_train.shape[1]

    match model_type:
        case "base":
            model = build_base_model(input_dim)
        case "regularization":
            model = build_regularized_model(input_dim, l2_lambda)
        case "dropout":
            model = build_dropout_model(input_dim, dropout_rate)
        case "complete":
            model = build_complete_model(input_dim, dropout_rate, l2_lambda)
        case _:
            raise ValueError(
                f"Invalid model type: {model_type}: Please choose from option 1 to 4."
            )

    model = compile_model(model)

    model_callbacks = build_callbacks(checkpoint_path)

    # compute class weight due in case the target feauture is imbalanced
    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {i: weights[i] for i in range(len(weights))}

    history = train_model(
        model, X_train, X_val, y_train, y_val, model_callbacks, class_weight
    )

    return model, history


def build_base_model(input_dim: int) -> Model:
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )


def build_dropout_model(input_dim: int, dropout_rate: float = 0.3) -> Model:
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )


def build_regularized_model(input_dim: int, l2_lambda: float = 0.001) -> Model:
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            BatchNormalization(),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            BatchNormalization(),
            Dense(1, activation="sigmoid"),
        ]
    )


def build_complete_model(
    input_dim: int, dropout_rate: float = 0.3, l2_lambda: float = 0.001
) -> Model:
    return Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_lambda)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(1, activation="sigmoid"),
        ]
    )


def compile_model(model: Model) -> Model:
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            Recall(name="recall"),
            Precision(name="precision"),
            AUC(name="auc"),
        ],
    )
    return model


def build_callbacks(checkpoint_path: str) -> List[callbacks.Callback]:
    # Stop early if val_loss doesn't improve for 10 consecutive epochs
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    return [early_stop, checkpoint]


def train_model(
    model: Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    model_callbacks: List[callbacks.Callback],
    class_weight: Dict,
) -> History:
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=model_callbacks,
        class_weight=class_weight,
    )
    return history
