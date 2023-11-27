from typing import Tuple
import numpy as np
import pandas as pd


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
