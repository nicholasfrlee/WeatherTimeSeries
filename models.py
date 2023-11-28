from pathlib import Path
import os
from typing import Tuple, Dict
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd

from utilities import Dataset


def build_lstm_model_1() -> Sequential:
    """Build and return the first model architecture."""
    model = Sequential()
    model.add(InputLayer(input_shape=(5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model


def build_lstm_model_2() -> Sequential:
    """Build and return the second model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, "silu"))
    model.add(Dense(1, "linear"))
    return model


def build_lstm_model_3() -> Sequential:
    """Build and return a model with softmax activation."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="softmax"))
    model.add(Dense(1, "linear"))
    return model


def build_lstm_model_4() -> Sequential:
    """Build and return the fourth model architecture."""
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


def build_gru_model1() -> Sequential:
    """Build and return a GRU model."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(GRU(64))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model


def compile_and_train_model(
    model: Sequential,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float,
    epochs: int,
) -> Sequential:
    """Compile and train a Keras Sequential model."""
    cp = ModelCheckpoint(f"saved_models/{model_name}", save_best_only=True)
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
    save_path = Path("saved_models") / model_name
    try:
        trained_model = load_model(save_path)
        print(f"Pre-trained {model_name} model loaded successfully.")
    except OSError:
        print(f"Pre-existing {model_name} model not found. Training model.")
        model = build_model_func()
        trained_model = compile_and_train_model(
            model, model_name, X_train, y_train, X_val, y_val, learning_rate, epochs
        )
        trained_model.save(save_path)
        print(f"{model_name} model trained and saved.")
    finally:
        assert save_path.is_dir()
    return trained_model


def get_prediction_results(
    trained_model: Sequential, X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Get predictions and results for a trained model."""
    predictions = trained_model.predict(X).flatten()
    results = pd.DataFrame(data={"Train Predictions": predictions, "Actual": y})
    return predictions, results


def evaluate_model(
    trained_model: Sequential, X: np.ndarray, y: np.ndarray
) -> Tuple[pd.DataFrame, float]:
    """Evaluate a trained model and return results."""
    predictions = trained_model.predict(X).flatten()
    results = pd.DataFrame(data={"Predictions": predictions, "Actuals": y})
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return results, rmse


def train_evaluate_models(models: Dict, dataset: Dataset, LEARNING_RATE: int):
    model_results = []
    for i, model in enumerate(models.keys()):
        model_name = models[model]
        trained_model = load_or_train_model(
            f"{model_name}/",
            model,
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            LEARNING_RATE,
            epochs=10,
        )
        model_train_results, model_train_rmse = evaluate_model(
            trained_model, dataset.X_train, dataset.y_train
        )
        model_val_results, model_val_rmse = evaluate_model(
            trained_model, dataset.X_val, dataset.y_val
        )
        model_test_results, model_test_rmse = evaluate_model(
            trained_model, dataset.X_test, dataset.y_test
        )
        model_results.append(
            {
                "model name": model_name,
                "train": model_train_rmse,
                "val": model_val_rmse,
                "test": model_test_rmse,
            }
        )
    return model_results
