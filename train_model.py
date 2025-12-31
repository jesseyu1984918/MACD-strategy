# train_models.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib
from sklearn.metrics import brier_score_loss

DATA_PATH = "data/weekly_ml_dataset.parquet"
os.makedirs("models", exist_ok=True)

FEATURES = [
    "Dist_to_MA20_%",
    "MA20_slope_5d",
    "UDR10",
    "TPS10",
    "VolRatio_5_20",
    "Context",
]

def train_one(X, y):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    # time-aware CV (calibration needs CV; this is a simple starting point)
    tscv = TimeSeriesSplit(n_splits=5)
    model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    model.fit(X, y)
    return model

def main():
    df = pd.read_parquet(DATA_PATH).sort_values("Date")
    df = pd.read_parquet(DATA_PATH).sort_values("Date")

    print("Rows:", len(df))
    print("Tickers:", df["Ticker"].nunique())
    print("Date range:", df["Date"].min(), "→", df["Date"].max())

    # Class balance (how often ±3% happens)
    up_rate = df["y_up"].mean()
    down_rate = df["y_down"].mean()
    print(f"y_up rate (R5 >= +3%): {up_rate:.3%}")
    print(f"y_down rate (R5 <= -3%): {down_rate:.3%}")

    # Missing values in features
    FEATURES = [
        "Dist_to_MA20_%", "MA20_slope_5d", "UDR10", "TPS10", "VolRatio_5_20", "Context"
    ]
    na_pct = df[FEATURES].isna().mean().sort_values(ascending=False) * 100
    print("\nMissing % by feature:")
    print(na_pct.round(2))

    # Quick peek
    print("\nHead:")
    print(df[["Date", "Ticker"] + FEATURES + ["y_up", "y_down"]].head(3))


    df = df.dropna(subset=["Date"])

    # Basic cleanup
    X = df[FEATURES]
    y_up = df["y_up"].astype(int)
    y_down = df["y_down"].astype(int)

    model_up = train_one(X, y_up)
    model_down = train_one(X, y_down)

    p_up = model_up.predict_proba(X)[:, 1]
    p_down = model_down.predict_proba(X)[:, 1]

    print("Brier score (up):", brier_score_loss(y_up, p_up))
    print("Brier score (down):", brier_score_loss(y_down, p_down))

    print("Mean predicted P_up:", p_up.mean())
    print("Mean predicted P_down:", p_down.mean())

    joblib.dump(model_up, "models/model_up.pkl")
    joblib.dump(model_down, "models/model_down.pkl")
    print("Saved models/models/model_up.pkl and models/models/model_down.pkl")

if __name__ == "__main__":
    main()
