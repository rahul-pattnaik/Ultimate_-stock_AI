import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(input_shape):

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))

    model.add(LSTM(64))

    model.add(Dense(1))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model