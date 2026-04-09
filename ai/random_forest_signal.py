from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def train_rf(df):

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    features = df[[
        "rsi",
        "macd",
        "ma50",
        "ma200"
    ]].dropna()

    target = df["target"].loc[features.index]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=200
    )

    model.fit(X_train, y_train)

    return model