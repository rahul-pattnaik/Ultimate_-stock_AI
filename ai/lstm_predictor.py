# ai/lstm_predictor.py
# ─────────────────────────────────────────────────────────────────────────────
# LSTM Price Predictor
# Features  : Close · Volume · RSI · MACD · BB position (multi-variate)
# Output    : next N-day price prediction + directional signal
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Feature Builder ───────────────────────────────────────────────────────────

def _build_lstm_features(df: pd.DataFrame) -> np.ndarray:
    """
    Multi-variate features for LSTM (normalised per-column).
    Returns array of shape (n_rows, n_features).
    """
    d = df.copy()
    close = d["Close"]

    # RSI-14
    delta = close.diff()
    g = delta.clip(lower=0).rolling(14).mean()
    l = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + g / l.replace(0, np.nan))

    # MACD histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    d["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # Bollinger Band position (0=lower, 1=upper)
    bb_mean = close.rolling(20).mean()
    bb_std  = close.rolling(20).std()
    d["bb_pos"] = (close - (bb_mean - 2*bb_std)) / (4 * bb_std + 1e-9)

    # Log volume ratio
    d["vol_ratio"] = np.log1p(d["Volume"] / d["Volume"].rolling(20).mean())

    # Daily return
    d["ret"] = close.pct_change()

    cols = ["Close", "rsi", "macd_hist", "bb_pos", "vol_ratio", "ret"]
    return d[cols].dropna().values


def _make_sequences(data: np.ndarray, seq_len: int):
    """Create sliding window (X, y) sequences for LSTM training."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, 0])   # predict Close (column 0)
    return np.array(X), np.array(y)


def _quiet_tensorflow_logs() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"Do not pass an `input_shape`/`input_dim` argument to a layer.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*triggered tf.function retracing.*",
        category=UserWarning,
    )
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
    except Exception:
        pass

    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except Exception:
        pass


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_lstm(input_shape: tuple):
    """
    Stacked LSTM with Dropout + BatchNorm for regularisation.

    Architecture:
        LSTM(128, return_sequences) → Dropout(0.2)
        LSTM(64,  return_sequences) → Dropout(0.2)
        LSTM(32)                    → Dropout(0.1)
        Dense(16, relu)
        Dense(1)
    """
    try:
        _quiet_tensorflow_logs()
        from tensorflow.keras import Input
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                             BatchNormalization)
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            BatchNormalization(),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="huber",              # robust to outliers vs MSE
            metrics=["mae"],
        )
        return model

    except ImportError:
        raise ImportError(
            "TensorFlow not installed. Run: pip install tensorflow"
        )


# ── Train & Predict ───────────────────────────────────────────────────────────

def train_lstm(df: pd.DataFrame,
               seq_len: int = 60,
               epochs: int = 50,
               batch_size: int = 32) -> dict:
    """
    Full training pipeline: feature engineering → scale → sequence →
    train with early stopping → return model + scaler + metrics.

    Args:
        df         : yfinance OHLCV DataFrame (≥120 rows recommended)
        seq_len    : look-back window in trading days (default 60)
        epochs     : max training epochs (early stopping kicks in)
        batch_size : mini-batch size

    Returns dict with model, scaler, history, and evaluation metrics.
    """
    try:
        _quiet_tensorflow_logs()
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    if df is None or len(df) < seq_len + 20:
        return {"error": f"Need at least {seq_len + 20} rows of data"}

    raw = _build_lstm_features(df)

    # Scale each feature column independently to [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw)

    # 80/20 temporal split
    split = int(len(scaled) * 0.80)
    train_data = scaled[:split]
    test_data  = scaled[split - seq_len:]  # include look-back for test

    X_train, y_train = _make_sequences(train_data, seq_len)
    X_test,  y_test  = _make_sequences(test_data,  seq_len)

    if len(X_train) == 0 or len(X_test) == 0:
        return {"error": "Not enough data after sequencing"}

    model = build_lstm(input_shape=(seq_len, raw.shape[1]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,       # CRITICAL: never shuffle time-series
        verbose=0,
    )

    # Evaluate
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse-transform close column only (column 0)
    def inv_close(arr):
        dummy = np.zeros((len(arr), raw.shape[1]))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    y_pred_actual = inv_close(y_pred_scaled)
    y_true_actual = inv_close(y_test)

    mae  = float(np.mean(np.abs(y_pred_actual - y_true_actual)))
    rmse = float(np.sqrt(np.mean((y_pred_actual - y_true_actual) ** 2)))
    mape = float(np.mean(np.abs((y_true_actual - y_pred_actual) /
                                (y_true_actual + 1e-9))) * 100)

    return {
        "model":   model,
        "scaler":  scaler,
        "seq_len": seq_len,
        "n_features": raw.shape[1],
        "metrics": {
            "MAE":  round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": f"{round(mape, 2)}%",
            "epochs_trained": len(history.history["loss"]),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 6),
        }
    }


def lstm_predict(df: pd.DataFrame,
                 model=None, scaler=None,
                 seq_len: int = 60,
                 days_ahead: int = 5) -> dict:
    """
    Predict next `days_ahead` closing prices using trained LSTM.
    If model/scaler are None, trains a fresh model first.

    Returns:
        current_price, predicted_prices (list), direction, change_pct, metrics
    """
    result = {}

    if model is None or scaler is None:
        train_result = train_lstm(df, seq_len=seq_len)
        if "error" in train_result:
            return train_result
        model   = train_result["model"]
        scaler  = train_result["scaler"]
        result["training_metrics"] = train_result["metrics"]

    raw    = _build_lstm_features(df)
    scaled = scaler.transform(raw)

    # Rolling multi-step forecast
    window     = list(scaled[-seq_len:])
    predictions = []

    for _ in range(days_ahead):
        X_input = np.array(window[-seq_len:]).reshape(1, seq_len, raw.shape[1])
        pred_scaled = float(model.predict(X_input, verbose=0)[0][0])

        # Inverse-transform close
        dummy       = np.zeros((1, raw.shape[1]))
        dummy[0, 0] = pred_scaled
        pred_price  = float(scaler.inverse_transform(dummy)[0, 0])
        predictions.append(round(pred_price, 2))

        # Append prediction as next step (only close changes, others held constant)
        next_row    = window[-1].copy()
        next_row[0] = pred_scaled
        window.append(next_row)

    current_price = float(df["Close"].dropna().iloc[-1])
    final_price   = predictions[-1]
    change_pct    = (final_price - current_price) / current_price * 100

    result.update({
        "current_price":    round(current_price, 2),
        "predicted_prices": predictions,
        "change_percent":   round(change_pct, 2),
        "direction":        "UP 📈" if final_price > current_price else "DOWN 📉",
    })
    return result
