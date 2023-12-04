from pathlib import Path
import pickle
from typing import Tuple, Dict, List
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer, GRU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

import tensorflow as tf
import numpy as np
import pandas as pd

from utilities import Dataset


class Model:
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        if self.random_seed is not None:
            # Set random seed for reproducibility
            np.random.seed(self.random_seed)
            tf.random.set_seed(self.random_seed)

    def compile_and_train_model():
        pass

    def load_or_train_model(
        model_name: str,
        self,
        X_train: List[float],
        y_train: List[float],
        X_val: List[float],
        y_val: List[float],
        learning_rate: float,
        epochs: int,
        train_override: bool,
    ):
        save_path = Path("saved_models") / model_name
        try:
            if train_override:
                raise OSError
            trained_model = load_model(save_path)

            print(f"Pre-trained {model_name} model loaded successfully.")
        except OSError:
            print(f"Pre-existing {model_name} model not found. Training model.")
            trained_model = self.compile_and_train_model(
                self, model_name, X_train, y_train, X_val, y_val, learning_rate, epochs
            )
            trained_model.save(save_path)
            print(f"{model_name} model trained and saved.")
        finally:
            pickle_save_path = save_path / "history.pickle"
            with open(pickle_save_path, "rb") as handle:
                trained_model_history = pickle.load(handle)

        return trained_model, trained_model_history


def evaluate_model(
    trained_model: Sequential, X: np.ndarray, y: np.ndarray
) -> Tuple[pd.DataFrame, float]:
    """Evaluate a trained model and return results."""
    predictions = trained_model.predict(X).flatten()
    results = pd.DataFrame(data={"Predictions": predictions, "Actuals": y})
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return results, rmse


def train_evaluate_models(
    models: Dict,
    dataset: Dataset,
    LEARNING_RATE: float,
    epochs: int,
    train_override: bool,
) -> pd.DataFrame:
    """
    Trains all models specified in a function : model_name dictionary and outputs a dataframe with the results over epochs
    """
    model_results = []
    for i, model in enumerate(models.keys()):
        model_name = models[model]
        trained_model, trained_model_history = Model.load_or_train_model(
            f"{model_name}/",
            model,
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            LEARNING_RATE,
            epochs,
            train_override,
        )

        # Evaluate on train, val, and test sets after training completion
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
                "model_name": model_name,
                "train_rmse": model_train_rmse,
                "val_rmse": model_val_rmse,
                "test_rmse": model_test_rmse,
                "losses": trained_model_history.history["loss"],
                "rmses": trained_model_history.history["root_mean_squared_error"],
                "val_losses": trained_model_history.history["val_loss"],
                "val_rmses": trained_model_history.history[
                    "val_root_mean_squared_error"
                ],
                "epochs_trained": len(trained_model_history.history["loss"]),
            }
        )

    return pd.DataFrame(model_results)


class TensorFlowModel(Model, Sequential):
    def __init__(self, random_seed=None):
        super(TensorFlowModel, self).__init__(random_seed=random_seed)

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
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[cp],
        )
        pickle_save_path = Path("saved_models") / model_name / "history.pickle"

        with open(pickle_save_path, "wb") as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return model

    def load_or_train_model(
        model_name: str,
        self,
        X_train: List[float],
        y_train: List[float],
        X_val: List[float],
        y_val: List[float],
        learning_rate: float,
        epochs: int,
        train_override: bool,
    ):
        return super().load_or_train_model(
            self, X_train, y_train, X_val, y_val, learning_rate, epochs, train_override
        )
