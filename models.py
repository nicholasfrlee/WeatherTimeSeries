import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd


def build_model1() -> Sequential:
    """Build and return the first model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, "relu"))
    model.add(Dense(1, "linear"))
    return model


def build_model2() -> Sequential:
    """Build and return the second model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, "silu"))
    model.add(Dense(1, "linear"))
    return model


def build_model3() -> Sequential:
    """Build and return the third model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(8))
    model.add(Dense(4, "relu"))
    model.add(Dense(1, "linear"))
    return model


def compile_and_train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> Sequential:
    """Compile and train a Keras Sequential model."""
    cp = ModelCheckpoint("model/", save_best_only=True)
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[RootMeanSquaredError()],
    )
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[cp]
    )
    return model


def load_or_train_model(
    model_name, build_model_func, X_train, y_train, X_val, y_val, learning_rate, epochs
):
    try:
        trained_model = load_model(model_name)
        print(f"Pre-trained {model_name} model loaded successfully.")
    except OSError:
        print(f"Pre-existing {model_name} model not found. Training model.")
        model = build_model_func()
        trained_model = compile_and_train_model(
            model, X_train, y_train, X_val, y_val, learning_rate, epochs
        )
        trained_model.save(model_name)
        print(f"{model_name} model trained and saved.")
    return trained_model