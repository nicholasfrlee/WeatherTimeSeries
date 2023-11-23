from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import mean_squared_error


def df_to_X_y(df: pd.DataFrame, window_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DataFrame to input features (X) and labels (y) for time series prediction."""
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [[a] for a in df_as_np[i : i + window_size]]
        X.append(row)
        label = df_as_np[i + window_size]
        y.append(label)
    return np.array(X), np.array(y)


def build_model1() -> Sequential:
    """Build and return the first model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, "relu"))
    model.add(Dense(1, "linear"))
    return model


def build_model2() -> Sequential:
    """Build and return the first model architecture."""
    model = Sequential()
    model.add(InputLayer((5, 1)))
    model.add(LSTM(64))
    model.add(Dense(8, "silu"))
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
