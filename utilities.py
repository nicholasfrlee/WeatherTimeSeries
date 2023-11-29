from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


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


def plot_variable_comparison(df: pd.DataFrame, y_variable: str, epochs: int, num_models: int):
    plt.plot(np.arange(epochs), df[y_variable][0])
    for i in range(num_models):
        plt.plot(np.arange(epochs), df[y_variable][i], label=df["model_name"][i])
    plt.xlabel("Epoch")
    plt.ylabel(y_variable)
    plt.legend()
    plt.grid()
    plt.title(f"Plot for {y_variable}")
    plt.show()
    return
