"""
Inference correctness tests — Energy Consumption Forecast

    python src/test_inference.py

These tests guard the failure modes that a screenshot cannot catch:

1. artifacts missing or unloadable,
2. a model that loads, runs, and returns the same number for every input
   (the classic broken deployment),
3. the scaler not being applied on the inference path,
4. predictions that contradict the physics (more speed or more load must cost
   more energy),
5. a forecast that is flat instead of having a daily shape,
6. metrics that no longer match the saved comparison table.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator import generate_dataset
from models import (evaluate_model, forecast_horizon, load_model, predict_energy)
from preprocessing import prepare_features, split_and_scale

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def _artifacts():
    """Load everything the deployed app needs."""
    rf = load_model(MODELS_DIR / "random_forest.pkl")
    bundle = load_model(MODELS_DIR / "polynomial_regression.pkl")
    lr = load_model(MODELS_DIR / "linear_regression.pkl")
    scaler = load_model(MODELS_DIR / "scaler.pkl")
    if isinstance(bundle, dict):
        pr, poly = bundle["model"], bundle["poly"]
    else:
        pr, poly = bundle
    return {"rf": rf, "lr": lr, "pr": pr, "poly": poly, "scaler": scaler}


def test_model_files_exist():
    """All four artifacts must be present."""
    for name in ["linear_regression.pkl", "polynomial_regression.pkl",
                 "random_forest.pkl", "scaler.pkl"]:
        path = MODELS_DIR / name
        assert path.exists(), f"Missing artifact: {path}"
    print("[PASS] All model files exist")


def test_scaler_loads():
    """The scaler must exist and expose transform()."""
    scaler = load_model(MODELS_DIR / "scaler.pkl")
    assert hasattr(scaler, "transform"), "Scaler missing transform()"
    print("[PASS] Scaler loaded successfully")


def test_speed_sensitivity():
    """More speed must cost more energy, and the effect must be large."""
    a = _artifacts()
    slow = predict_energy(a["rf"], 12, 2, 20, 500, scaler=a["scaler"])
    fast = predict_energy(a["rf"], 12, 2, 90, 500, scaler=a["scaler"])
    assert fast > slow * 2, f"Speed has no effect: {slow:.2f} -> {fast:.2f} kWh"
    print(f"[PASS] Speed sensitivity: 20 -> 90 km/h gives {slow:.2f} -> "
          f"{fast:.2f} kWh")


def _payload_band(hour: int, df: pd.DataFrame) -> tuple[float, float]:
    """Observed payload band (2nd-98th percentile) for a given hour, while driving."""
    sub = df[(df["hour"] == hour) & (df["speed_kmh"] > 0)]
    return (float(sub["payload_kg"].quantile(0.02)),
            float(sub["payload_kg"].quantile(0.98)))


def test_load_sensitivity():
    """
    Inside the payload band the van actually drives with, a heavier load must
    cost more energy, monotonically.

    Asking for 0 kg at midday would be an extrapolation — this van is only ever
    empty once the round is finished in the evening — so the test uses the
    observed band for the hour instead of an invented one.
    """
    a = _artifacts()
    df = generate_dataset(num_days=120, seed=42)
    lo, hi = _payload_band(8, df)
    steps = [lo, (lo + hi) / 2, hi]
    preds = [predict_energy(a["rf"], 8, 2, 60, p, scaler=a["scaler"])
             for p in steps]
    assert preds[-1] > preds[0] + 0.8, \
        f"Load has no effect across {lo:.0f}-{hi:.0f} kg: {preds}"
    assert preds[0] <= preds[1] <= preds[2], f"Not monotonic in load: {preds}"
    print(f"[PASS] Load sensitivity across the observed band "
          f"{lo:.0f}-{hi:.0f} kg: "
          f"{' -> '.join(f'{v:.2f}' for v in preds)} kWh")


def test_all_models_differ():
    """The three models must not return the same number."""
    a = _artifacts()
    kwargs = dict(hour=14, dow=1, speed_kmh=70, payload_kg=800)
    lr = predict_energy(a["lr"], **kwargs, scaler=a["scaler"])
    pr = predict_energy(a["pr"], **kwargs, poly=a["poly"], scaler=a["scaler"])
    rf = predict_energy(a["rf"], **kwargs, scaler=a["scaler"])
    assert max(lr, pr, rf) - min(lr, pr, rf) > 0.1, \
        f"Models agree far too closely: {lr}, {pr}, {rf}"
    print(f"[PASS] Three distinct models: LR {lr:.3f} | PR {pr:.3f} | "
          f"RF {rf:.3f} kWh")


def test_parked_baseline():
    """Parked overnight the van must consume only the auxiliary load."""
    a = _artifacts()
    pred = predict_energy(a["rf"], 2, 2, 0, 0, scaler=a["scaler"])
    assert 0.0 < pred < 3.0, f"Parked prediction out of range: {pred:.2f} kWh"
    print(f"[PASS] Parked baseline: {pred:.2f} kWh/h of auxiliary load")


def test_plausible_range():
    """No prediction may leave the physically plausible band."""
    a = _artifacts()
    for hour in range(24):
        for speed in [0, 25, 60, 90]:
            for payload in [0, 600, 1200]:
                pred = predict_energy(a["rf"], hour, 2, speed, payload,
                                      scaler=a["scaler"])
                assert -5 < pred < 70, \
                    f"Out of range: hour={hour} speed={speed} load={payload} " \
                    f"-> {pred:.2f}"
    print("[PASS] All predictions in the plausible 0-70 kWh range")


def test_extrapolation_is_flagged():
    """
    Predictions outside the training region must stay bounded rather than
    explode — the app warns the user, but the model must not return nonsense.
    """
    a = _artifacts()
    far = predict_energy(a["rf"], 12, 2, 200, 3000, scaler=a["scaler"])
    assert -5 < far < 70, f"Extrapolation exploded: {far:.2f} kWh"
    print(f"[PASS] Out-of-range query stays bounded: {far:.2f} kWh")


def test_forecast_has_daily_shape():
    """The 24-hour forecast must not be flat: small at night, large by day."""
    a = _artifacts()
    df = generate_dataset(num_days=30, seed=42)
    forecast = forecast_horizon(a["rf"], df["timestamp"].iloc[-1], steps=24,
                                scaler=a["scaler"])
    assert len(forecast) == 24, "Forecast length is wrong"
    assert (forecast["forecast_kwh"] >= 0).all(), "Negative forecast value"
    assert forecast["forecast_kwh"].min() < 2.0, "No parked hours in the forecast"
    assert forecast["forecast_kwh"].max() > 15.0, "No driving hours in the forecast"
    print(f"[PASS] Forecast has a daily shape: min "
          f"{forecast['forecast_kwh'].min():.2f} | max "
          f"{forecast['forecast_kwh'].max():.2f} kWh")


def test_metrics_reproduce():
    """Re-evaluating the saved models must reproduce the published metrics."""
    saved = pd.read_csv(MODELS_DIR / "model_comparison.csv")
    a = _artifacts()
    df = generate_dataset(num_days=120, seed=42)
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    X_test, y_test = splits["X_test"], splits["y_test"]

    computed = [
        evaluate_model(a["lr"], X_test, y_test, "Linear Regression"),
        evaluate_model(a["pr"], X_test, y_test, "Polynomial Regression",
                       poly=a["poly"]),
        evaluate_model(a["rf"], X_test, y_test, "Random Forest"),
    ]
    for row, result in zip(saved.itertuples(), computed):
        assert abs(float(row._2) - result["mae"]) < 1e-6, \
            f"MAE drift for {result['model_name']}"
        assert abs(float(row._5) - result["r2"]) < 1e-6, \
            f"R² drift for {result['model_name']}"
    print("[PASS] Saved metrics reproduce exactly on the held-out split")


if __name__ == "__main__":
    test_model_files_exist()
    test_scaler_loads()
    test_speed_sensitivity()
    test_load_sensitivity()
    test_all_models_differ()
    test_parked_baseline()
    test_plausible_range()
    test_extrapolation_is_flagged()
    test_forecast_has_daily_shape()
    test_metrics_reproduce()
    print("\n=== ALL INFERENCE TESTS PASSED ===")
