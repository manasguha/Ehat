"""
Regression Models for Charging Station Load Prediction

Three models:
    1. Linear Regression (baseline)
    2. Polynomial Regression (nonlinear)
    3. Random Forest Regressor (captures interactions)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import pickle
from pathlib import Path


def train_linear_regression(X_train, y_train):
    """Train a simple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_polynomial_regression(X_train, y_train, degree=2):
    """Train polynomial regression (features transformed internally)."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    return model, poly


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """Train a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model", poly=None):
    """
    Evaluate a trained model on test data.

    Returns dict with predictions and metrics.
    """
    if poly is not None:
        X_eval = poly.transform(X_test)
    else:
        X_eval = X_test

    y_pred = model.predict(X_eval)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "model_name": model_name,
        "y_pred": y_pred,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def cross_validate_model(model, X, y, cv=5, scoring="r2"):
    """Perform k-fold cross-validation."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "mean_r2": round(scores.mean(), 4),
        "std_r2": round(scores.std(), 4),
        "scores": scores.tolist(),
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of evaluate_model() results.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "MAE (kW)": r["mae"],
            "RMSE (kW)": r["rmse"],
            "R²": r["r2"],
        })
    return pd.DataFrame(rows)


def save_model(model, path: str | Path) -> Path:
    """Save a trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str | Path):
    """Load a trained model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_load(model, hour_sin, hour_cos, vehicles, poly=None, scaler=None):
    """
    Predict load for a single observation.

    Parameters
    ----------
    hour_sin : float
    hour_cos : float
    vehicles : int or float
    poly     : PolynomialFeatures (only for polynomial model)
    scaler   : StandardScaler (must be provided -- models require scaled features)

    Returns
    -------
    float : predicted load in kW
    """
    X = np.array([[hour_sin, hour_cos, vehicles]])
    if scaler is not None:
        X = scaler.transform(X)
    if poly is not None:
        X = poly.transform(X)
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# CLI: train all three models and print comparison
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from data_generator import generate_dataset
    from preprocessing import prepare_features, split_and_scale

    # Generate data
    df = generate_dataset(num_days=90, seed=42)
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    # 1. Linear Regression
    lr = train_linear_regression(X_train, y_train)
    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")

    # 2. Polynomial Regression
    pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)

    # 3. Random Forest
    rf = train_random_forest(X_train, y_train, n_estimators=100)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")

    # Comparison table
    comparison = compare_models([lr_result, pr_result, rf_result])
    print("\n" + "=" * 55)
    print("  MODEL COMPARISON")
    print("=" * 55)
    print(comparison.to_string(index=False))
    print("=" * 55)

    # Save best model (Random Forest typically wins)
    best = max([lr_result, pr_result, rf_result], key=lambda x: x["r2"])
    print(f"\nBest model: {best['model_name']} (R² = {best['r2']})")
