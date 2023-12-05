import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utilities import df_to_X_y, Dataset, plot_variable_comparison

from models import (
    Model,
    TensorFlowModel,
)


def main():
    WINDOW_SIZE = 5
    EPOCHS = 6
    LEARNING_RATE = 0.0001

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print("Running on GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Running on CPU")

    df = pd.read_csv("jena_climate_2009_2016.csv")
    hourly_df = df[5::6]
    hourly_df.index = pd.to_datetime(hourly_df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    temp = hourly_df["T (degC)"]

    X1, y1 = df_to_X_y(temp, WINDOW_SIZE)

    sample_dataset = Dataset(
        X1[:60000], y1[:60000], X1[60000:65000], y1[60000:65000], X1[65000:], y1[65000:]
    )
    lstm_model_1 = TensorFlowModel("LSTM1")
    lstm_model_2 = TensorFlowModel("LSTM2")
    lstm_model_3 = TensorFlowModel("LSTM3")
    lstm_model_4 = TensorFlowModel("LSTM4")
    gru_model1 = TensorFlowModel("GRU1")

    models_to_train = {
        lstm_model_1: "LSTM1",
        lstm_model_2: "LSTM2",
        lstm_model_3: "LSTM3",
        lstm_model_4: "LSTM4",
        gru_model1: "GRU1",
    }

    model_results_df = Model.train_evaluate_models(
        models_to_train, sample_dataset, LEARNING_RATE, EPOCHS, train_override=False
    )

    plot_variable_comparison(
        model_results_df, ["losses", "val_losses"], EPOCHS, len(models_to_train)
    )


if __name__ == "__main__":
    main()
