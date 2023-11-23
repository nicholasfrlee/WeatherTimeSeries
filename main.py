import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    df_to_X_y,
    build_model1,
    build_model2,
    compile_and_train_model,
    get_prediction_results,
    evaluate_model,
)

from tensorflow.keras.layers import *
from tensorflow.keras.models import load_model


def main():
    WINDOW_SIZE = 5
    LEARNING_RATE = 0.0001

    df = pd.read_csv("jena_climate_2009_2016.csv")
    hourly_df = df[5::6]
    hourly_df.index = pd.to_datetime(hourly_df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    temp = hourly_df["T (degC)"]

    X1, y1 = df_to_X_y(temp, WINDOW_SIZE)

    X_train1, y_train1 = X1[:60000], y1[:60000]
    X_val1, y_val1 = X1[60000:65000], y1[60000:65000]
    X_test1, y_test1 = X1[65000:], y1[65000:]

    try:
        trained_model1 = load_model("model1/")
        print("Pre-trained model loaded successfully.")
    except OSError:
        print("Pre-existing model not found. Training model.")
        model1 = build_model1()
        trained_model1 = compile_and_train_model(
            model1, X_train1, y_train1, X_val1, y_val1, LEARNING_RATE, epochs=10
        )
        trained_model1.save("model1/")
        print("Model trained and saved.")

    try:
        trained_model2 = load_model("model2/")
        print("Pre-trained model loaded successfully.")
    except OSError:
        print("Pre-existing model not found. Training model.")
        model2 = build_model2()
        trained_model2 = compile_and_train_model(
            model2, X_train1, y_train1, X_val1, y_val1, LEARNING_RATE, epochs=10
        )
        trained_model2.save("model2/")
        print("Model trained and saved.")

    _, model1_train_results = get_prediction_results(trained_model1, X_train1, y_train1)

    _, model1_val_results = get_prediction_results(trained_model1, X_val1, y_val1)

    model1_test_results, model1_rmse = evaluate_model(trained_model1, X_test1, y_test1)

    _, model2_train_results = get_prediction_results(trained_model2, X_train1, y_train1)

    _, model2_val_results = get_prediction_results(trained_model2, X_val1, y_val1)

    model2_test_results, model2_rmse = evaluate_model(trained_model2, X_test1, y_test1)

    print(f"Model 1 RMSE: {model1_rmse}")
    print(f"Model 2 RMSE: {model2_rmse}")


if __name__ == "__main__":
    main()
