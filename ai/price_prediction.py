# ai/price_prediction.py
# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Price Predictor
# Methods: Linear Regression · Polynomial Regression · Weighted Moving Average
# Output : predicted price, direction, % change, daily targets, confidence
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def _linear_forecast(close: np.ndarray, days: int) -> np.ndarray:
    X = np.arange(len(close)).reshape(-1, 1)
    model = LinearRegression().fit(X, close)
    future = np.arange(len(close), len(close) + days).reshape(-1, 1)
    return model.predict(future)


def _poly_forecast(close: np.ndarray, days: int, degree: int = 3) -> np.ndarray:
    """Polynomial regression — captures curvature in recent trend."""
    # Use only last 60 days to avoid overfitting long history
    window = close[-60:] if len(close) >= 60 else close
    X = np.arange(len(window)).reshape(-1, 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, window)
    future = np.arange(len(window), len(window) + days).reshape(-1, 1)
    preds  = model.predict(future)
    # Clip extreme polynomial extrapolation (±20% from current)
    cap = close[-1] * 0.20
    return np.clip(preds, close[-1] - cap, close[-1] + cap)


def _wma_forecast(close: np.ndarray, days: int) -> np.ndarray:
    """Exponentially weighted moving average extrapolation."""
    window  = min(30, len(close))
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    base     = np.dot(weights, close[-window:])
    # Momentum from last 5 days
    momentum = (close[-1] - close[-6]) / close[-6] if len(close) > 6 else 0.0
    decay    = 0.85  # momentum decays each day
    return np.array([base * (1 + momentum * (decay ** i))
                     for i in range(1, days + 1)])


def _confidence(close: np.ndarray,
                lr_pred: np.ndarray,
                poly_pred: np.ndarray,
                wma_pred: np.ndarray) -> float:
    """
    Confidence = agreement between 3 models + inverse of recent volatility.
    Higher agreement and lower volatility → higher confidence.
    """
    spread = np.std([lr_pred[-1], poly_pred[-1], wma_pred[-1]])
    spread_pct = spread / close[-1] * 100

    # Match numerator/denominator lengths for recent return volatility.
    vol = np.std(np.diff(close[-20:]) / close[-20:-1]) * 100 if len(close) > 21 else 3.0

    confidence = 90 - spread_pct * 5 - vol * 3
    return round(float(np.clip(confidence, 10, 95)), 1)


def predict_price(df: pd.DataFrame, days_ahead: int = 5) -> dict:
    """
    Ensemble price predictor using 3 methods blended by weight.

    Args:
        df         : yfinance DataFrame with 'Close' column
        days_ahead : how many trading days to forecast (default 5)

    Returns dict:
        current_price, predicted_price, change_percent,
        direction, confidence, daily_targets
    """
    if df is None or len(df) < 30:
        return {"error": "Need at least 30 days of data"}

    close = df["Close"].dropna().values.flatten().astype(float)

    lr_pred   = _linear_forecast(close, days_ahead)
    poly_pred = _poly_forecast(close, days_ahead)
    wma_pred  = _wma_forecast(close, days_ahead)

    # Weighted ensemble: poly captures curve better near-term
    ensemble = (0.35 * lr_pred + 0.40 * poly_pred + 0.25 * wma_pred)

    current   = float(close[-1])
    predicted = float(ensemble[-1])
    change    = (predicted - current) / current * 100

    confidence = _confidence(close, lr_pred, poly_pred, wma_pred)
    direction  = "UP 📈" if predicted > current else "DOWN 📉"

    return {
        "current_price":      round(current, 2),
        "predicted_price_5d": round(predicted, 2),
        "change_percent":     round(change, 2),
        "direction":          direction,
        "confidence":         f"{confidence}%",
        "daily_targets":      [round(float(p), 2) for p in ensemble],
        "model_forecasts": {
            "linear":     round(float(lr_pred[-1]), 2),
            "polynomial": round(float(poly_pred[-1]), 2),
            "wma":        round(float(wma_pred[-1]), 2),
        }
    }
