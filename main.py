import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utilities import (
    df_to_X_y,
)

from models import (
    build_model1,
    build_model2,
    build_model3,
    load_or_train_model,
)

from evaluate import (
    get_prediction_results,
    evaluate_model,
)


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

    trained_model1 = load_or_train_model(
        "model1/",
        build_model1,
        X_train1,
        y_train1,
        X_val1,
        y_val1,
        LEARNING_RATE,
        epochs=10,
    )
    _, model1_train_results = get_prediction_results(trained_model1, X_train1, y_train1)
    _, model1_val_results = get_prediction_results(trained_model1, X_val1, y_val1)
    model1_test_results, model1_rmse = evaluate_model(trained_model1, X_test1, y_test1)

    trained_model2 = load_or_train_model(
        "model2/",
        build_model2,
        X_train1,
        y_train1,
        X_val1,
        y_val1,
        LEARNING_RATE,
        epochs=10,
    )
    _, model2_train_results = get_prediction_results(trained_model2, X_train1, y_train1)
    _, model2_val_results = get_prediction_results(trained_model2, X_val1, y_val1)
    model2_test_results, model2_rmse = evaluate_model(trained_model2, X_test1, y_test1)

    trained_model3 = load_or_train_model(
        "model3/",
        build_model3,
        X_train1,
        y_train1,
        X_val1,
        y_val1,
        LEARNING_RATE,
        epochs=10,
    )
    _, model3_train_results = get_prediction_results(trained_model3, X_train1, y_train1)
    _, model3_val_results = get_prediction_results(trained_model3, X_val1, y_val1)
    model3_test_results, model3_rmse = evaluate_model(trained_model3, X_test1, y_test1)

    print(f"Model 1 RMSE: {model1_rmse}")
    print(f"Model 2 RMSE: {model2_rmse}")
    print(f"Model 3 RMSE: {model3_rmse}")


if __name__ == "__main__":
    main()
