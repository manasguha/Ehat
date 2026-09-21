"""
Feature Engineering & Preprocessing — Energy Consumption Forecast

Three things matter here, and all three are examinable:

1. Cyclic time encoding. A clock is a circle: feeding the raw hour makes the
   model believe 23:00 and 00:00 are 23 units apart. Following the assignment,
   "Time" is encoded as sin/cos pairs for hour-of-day *and* day-of-week.

2. The model inputs are exactly the assignment's three inputs — Time, Speed and
   Load. It is deliberately *not* autoregressive: the deployed app must predict
   from the planned duty cycle, never from the value it is trying to predict.
   The sequential (lag) features are implemented and measured separately so the
   report can quantify what they would buy, and why they are kept out of the
   live model.

3. The train/test split is *chronological*, not random: a random split would
   train on hours that come after the hours being tested, which is exactly the
   mistake that makes a forecasting project look better than it is.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Model inputs: the assignment's three inputs (time, speed, load).
BASE_FEATURE_COLS = [
    "hour_sin", "hour_cos",        # Time: hour of day (cyclic)
    "dow_sin", "dow_cos",          # Time: day of week (cyclic)
    "speed_kmh",                   # Speed
    "payload_kg",                  # Load
]

# Sequential (autoregressive) features — measured in the pipeline experiment.
LAG_FEATURE_COLS = ["energy_lag_1", "energy_lag_24", "energy_roll_24"]

FEATURE_COLS = BASE_FEATURE_COLS
TARGET_COL = "energy_kwh"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the cyclic Time features plus the derived driving descriptors.

    Added columns:
        hour_sin, hour_cos   cyclic time of day
        dow_sin, dow_cos     cyclic day of week
        is_weekend           binary flag
        is_rush              binary flag (07:00-09:59 and 17:00-19:59)
        period               morning / afternoon / evening / night
        distance_km          speed x 1 h
    """
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    rush_hours = set(range(7, 10)) | set(range(17, 20))
    df["is_rush"] = df["hour"].isin(rush_hours).astype(int)

    def _period(hour):
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"

    df["period"] = df["hour"].apply(_period)
    df["distance_km"] = (df["speed_kmh"] * 1.0).round(2)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sequential (lag) features from the target only, using shift() so that
    every feature is strictly in the past.
    """
    df = df.copy()
    df["energy_lag_1"] = df[TARGET_COL].shift(1)
    df["energy_lag_24"] = df[TARGET_COL].shift(24)
    df["energy_roll_24"] = df[TARGET_COL].shift(1).rolling(24).mean()
    return df


def prepare_features(df: pd.DataFrame, include_lags: bool = False) -> dict:
    """
    Build the supervised learning table.

    include_lags=False (default) uses only Time, Speed and Load, which is what the
    deployed dashboard predicts from. include_lags=True adds the sequential
    features, used by the pipeline's sensitivity experiment.

    Returns dict with X (features), y (target), feature_names, the engineered
    frame (used by the dashboard and the figures) and the column list used.
    """
    engineered = add_time_features(df)
    if include_lags:
        engineered = add_lag_features(engineered)
        cols = BASE_FEATURE_COLS + LAG_FEATURE_COLS
    else:
        cols = list(BASE_FEATURE_COLS)

    engineered = engineered.dropna(subset=cols + [TARGET_COL]).reset_index(drop=True)

    return {
        "X": engineered[cols].copy(),
        "y": engineered[TARGET_COL].copy(),
        "feature_names": list(cols),
        "frame": engineered,
        "columns": cols,
    }


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
    """
    Chronological split (the last 20% of hours are the test set) and
    standardisation fitted on the training portion only.

    Returns X_train, X_test, y_train, y_test, scaler and the split index.
    """
    n = len(X)
    n_train = int(round(n * (1 - test_size)))

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_test.columns, index=X_test.index)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "n_train": n_train,
        "split_index": n_train,
    }


def scale_features(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply an already-fitted scaler to a new feature frame (inference path)."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_dataset

    df = generate_dataset(num_days=30, seed=42)
    feat = prepare_features(df)

    print(f"Raw rows:    {len(df)}")
    print(f"Model rows:  {feat['X'].shape[0]}")
    print(f"Features:    {feat['feature_names']}")

    splits = split_and_scale(feat["X"], feat["y"])
    print(f"\nChronological split -- train {splits['n_train']} | "
          f"test {len(splits['X_test'])}")
    print(f"Train target mean: {splits['y_train'].mean():.2f} kWh")
    print(f"Test  target mean: {splits['y_test'].mean():.2f} kWh")
