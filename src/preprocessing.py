"""
Feature Engineering & Preprocessing

Transforms raw charging station data into ML-ready features.
Key technique: cyclic time encoding so that 23:00 and 00:00 are close.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based features from the 'hour' column.

    Features added:
        - hour_sin, hour_cos  (cyclic encoding)
        - is_peak             (binary: peak hour indicator)
        - is_offpeak          (binary: off-peak hour indicator)
        - period              (categorical: morning/afternoon/evening/night)
    """
    df = df.copy()

    # Cyclic encoding: preserves temporal proximity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Peak/off-peak flags
    peak_hours = set(range(7, 11)) | set(range(17, 21))
    offpeak_hours = set(range(22, 24)) | set(range(0, 7))

    df["is_peak"] = df["hour"].isin(peak_hours).astype(int)
    df["is_offpeak"] = df["hour"].isin(offpeak_hours).astype(int)

    # Period of day
    def _period(h):
        if 5 <= h < 12:
            return "morning"
        elif 12 <= h < 17:
            return "afternoon"
        elif 17 <= h < 21:
            return "evening"
        else:
            return "night"

    df["period"] = df["hour"].apply(_period)

    return df


def prepare_features(
    df: pd.DataFrame,
    include_poly: bool = False,
    poly_degree: int = 2,
) -> dict:
    """
    Prepare feature matrix and target vector.

    Returns dict with:
        X          : feature DataFrame
        y          : target Series (load_kw)
        feature_names: list of column names
        scaler     : fitted StandardScaler (for inference)
    """
    df = add_time_features(df)

    # Core features: cyclic time + vehicles
    feature_cols = ["hour_sin", "hour_cos", "vehicles"]

    # Optional: add polynomial interaction features
    if include_poly:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False, interaction_only=False)
        poly_cols = ["hour_sin", "hour_cos", "vehicles"]
        X_poly = poly.fit_transform(df[poly_cols])
        poly_names = poly.get_feature_names_out(poly_cols).tolist()
        X = pd.DataFrame(X_poly, columns=poly_names, index=df.index)
    else:
        X = df[feature_cols].copy()
        poly = None

    y = df["load_kw"].copy()

    return {
        "X": X,
        "y": y,
        "feature_names": X.columns.tolist(),
        "scaler": None,
        "poly": poly,
    }


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Split data into train/test sets and scale features.

    Returns dict with:
        X_train, X_test, y_train, y_test
        scaler : fitted StandardScaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_generator import generate_dataset

    df = generate_dataset(num_days=30, seed=42)
    result = prepare_features(df)

    print(f"Features shape: {result['X'].shape}")
    print(f"Feature names: {result['feature_names']}")
    print(f"\nSample features:")
    print(result["X"].head())

    splits = split_and_scale(result["X"], result["y"])
    print(f"\nTrain: {splits['X_train'].shape[0]} samples")
    print(f"Test:  {splits['X_test'].shape[0]} samples")
