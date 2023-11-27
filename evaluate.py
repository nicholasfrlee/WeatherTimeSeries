from typing import Tuple
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential


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
