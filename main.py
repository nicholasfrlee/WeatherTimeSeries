import pandas as pd

from utilities import df_to_X_y, Dataset

from models import (
    build_lstm_model_1,
    build_lstm_model_2,
    build_lstm_model_3,
    build_gru_model1,
    train_evaluate_models,
)


def main():
    WINDOW_SIZE = 5
    LEARNING_RATE = 0.0001

    df = pd.read_csv("jena_climate_2009_2016.csv")
    hourly_df = df[5::6]
    hourly_df.index = pd.to_datetime(hourly_df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    temp = hourly_df["T (degC)"]

    X1, y1 = df_to_X_y(temp, WINDOW_SIZE)

    sample_dataset = Dataset(
        X1[:60000], y1[:60000], X1[60000:65000], y1[60000:65000], X1[65000:], y1[65000:]
    )

    models_to_train = {
        build_lstm_model_1: "lstm_1",
        build_lstm_model_2: "lstm_2",
        build_lstm_model_3: "lstm_3",
        build_gru_model1: "gru_1",
    }

    model_results = train_evaluate_models(
        models_to_train, sample_dataset, LEARNING_RATE
    )
    print(model_results)


if __name__ == "__main__":
    main()
