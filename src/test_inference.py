"""
Inference correctness tests.

Verifies that:
1. All models load and predict with the scaler
2. Changing inputs produces changing outputs (not flat predictions)
3. Predictions are within plausible ranges
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import load_model, predict_load
from preprocessing import prepare_features, split_and_scale
from data_generator import generate_dataset


def test_predict_load_changes_with_inputs():
    """Changing vehicle count must change predicted load."""
    df = generate_dataset(num_days=90, seed=42)
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    scaler = splits["scaler"]

    rf = load_model(Path(__file__).resolve().parent.parent / "models" / "random_forest.pkl")

    hour_sin = np.sin(2 * np.pi * 18 / 24)
    hour_cos = np.cos(2 * np.pi * 18 / 24)

    loads = []
    for v in [5, 20, 40, 60]:
        pred = predict_load(rf, hour_sin, hour_cos, v, scaler=scaler)
        loads.append(pred)

    # Predictions must differ -- no flat curve
    assert loads[0] != loads[-1], f"Flat prediction: {loads}"
    # Monotonic increase expected (more vehicles = more load)
    assert loads[1] > loads[0], f"Expected increase: {loads}"
    print(f"[PASS] Vehicle sensitivity: {loads}")


def test_predict_load_changes_with_time():
    """Changing hour must change predicted load."""
    df = generate_dataset(num_days=90, seed=42)
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    scaler = splits["scaler"]

    rf = load_model(Path(__file__).resolve().parent.parent / "models" / "random_forest.pkl")

    loads = []
    for h in [3, 8, 14, 18, 22]:
        hs = np.sin(2 * np.pi * h / 24)
        hc = np.cos(2 * np.pi * h / 24)
        pred = predict_load(rf, hs, hc, 30, scaler=scaler)
        loads.append(pred)

    # Predictions must differ
    assert loads[0] != loads[-1], f"Flat prediction: {loads}"
    print(f"[PASS] Time sensitivity: {loads}")


def test_prediction_ranges():
    """All predictions should be within plausible load range."""
    df = generate_dataset(num_days=90, seed=42)
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    scaler = splits["scaler"]

    rf = load_model(Path(__file__).resolve().parent.parent / "models" / "random_forest.pkl")

    for hour in range(24):
        for veh in [5, 30, 60]:
            hs = np.sin(2 * np.pi * hour / 24)
            hc = np.cos(2 * np.pi * hour / 24)
            pred = predict_load(rf, hs, hc, veh, scaler=scaler)
            assert -50 < pred < 400, f"Out of range: hour={hour}, veh={veh}, pred={pred}"
    print("[PASS] All predictions in plausible range")


def test_scaler_saved():
    """scaler.pkl must exist and be loadable."""
    scaler_path = Path(__file__).resolve().parent.parent / "models" / "scaler.pkl"
    assert scaler_path.exists(), f"Missing: {scaler_path}"
    scaler = load_model(scaler_path)
    assert hasattr(scaler, "transform"), "Scaler missing transform method"
    print("[PASS] Scaler loaded successfully")


def test_models_saved():
    """All model files must exist."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    for name in ["linear_regression.pkl", "polynomial_regression.pkl", "random_forest.pkl", "scaler.pkl"]:
        p = models_dir / name
        assert p.exists(), f"Missing: {p}"
    print("[PASS] All model files exist")


if __name__ == "__main__":
    test_scaler_saved()
    test_models_saved()
    test_predict_load_changes_with_inputs()
    test_predict_load_changes_with_time()
    test_prediction_ranges()
    print("\n=== ALL INFERENCE TESTS PASSED ===")
