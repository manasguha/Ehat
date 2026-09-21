"""
Main Pipeline — Charging Station Load Prediction

Runs the complete ML pipeline:
    1. Generate simulated data
    2. Engineer features
    3. Train 3 regression models
    4. Evaluate and compare
    5. Generate all visualizations
    6. Save models and results
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_generator import (
    generate_dataset,
    generate_peak_offpeak_scenarios,
    save_dataset,
    STATION_CAPACITY_KW,
)
from preprocessing import prepare_features, split_and_scale, add_time_features
from models import (
    train_linear_regression,
    train_polynomial_regression,
    train_random_forest,
    evaluate_model,
    compare_models,
    save_model,
    predict_load,
)
from visualization import (
    plot_load_curve,
    plot_peak_offpeak,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_3d_load_surface,
    plot_sensitivity,
    plot_feature_importance,
    plot_prediction_distribution,
    plot_data_exploration,
    plot_whatif_table,
)

import pandas as pd
import numpy as np


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"


def main():
    print("=" * 60)
    print("  CHARGING STATION LOAD PREDICTION — ML PIPELINE")
    print("=" * 60)

    # ── Step 1: Generate Data ──────────────────────────────────────────────
    print("\n[1/7] Generating simulated charging station data...")
    df = generate_dataset(num_days=90, seed=42)
    csv_path = save_dataset(df, DATA_DIR / "charging_station_data.csv")
    print(f"      Saved {len(df)} rows ({df['day'].nunique()} days) to {csv_path}")
    print(f"      Load range: {df['load_kw'].min():.1f} – {df['load_kw'].max():.1f} kW")
    print(f"      Vehicle range: {df['vehicles'].min()} – {df['vehicles'].max()}")

    # ── Step 2: Feature Engineering ────────────────────────────────────────
    print("\n[2/7] Engineering features...")
    feat = prepare_features(df)
    print(f"      Features: {feat['feature_names']}")
    splits = split_and_scale(feat["X"], feat["y"])
    print(f"      Train: {splits['X_train'].shape[0]} | Test: {splits['X_test'].shape[0]}")

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # ── Step 3: Train Models ──────────────────────────────────────────────
    print("\n[3/7] Training regression models...")

    # Linear Regression
    lr = train_linear_regression(X_train, y_train)
    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")
    print(f"      Linear Regression      — MAE: {lr_result['mae']:.2f} | R²: {lr_result['r2']:.4f}")

    # Polynomial Regression
    pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)
    print(f"      Polynomial Regression  — MAE: {pr_result['mae']:.2f} | R²: {pr_result['r2']:.4f}")

    # Random Forest
    rf = train_random_forest(X_train, y_train, n_estimators=100)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")
    print(f"      Random Forest          — MAE: {rf_result['mae']:.2f} | R²: {rf_result['r2']:.4f}")

    # ── Step 4: Compare Models ────────────────────────────────────────────
    print("\n[4/7] Model comparison:")
    comparison = compare_models([lr_result, pr_result, rf_result])
    print(comparison.to_string(index=False))

    # Identify best model
    best_name = comparison.loc[comparison["R²"].idxmax(), "Model"]
    print(f"\n      Best model: {best_name}")

    # ── Step 5: Save Models + Scaler ─────────────────────────────────────
    print("\n[5/7] Saving models and scaler...")
    save_model(lr, MODEL_DIR / "linear_regression.pkl")
    save_model({"model": pr, "poly": poly}, MODEL_DIR / "polynomial_regression.pkl")
    save_model(rf, MODEL_DIR / "random_forest.pkl")
    save_model(splits["scaler"], MODEL_DIR / "scaler.pkl")
    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    print(f"      Saved to {MODEL_DIR}")

    # ── Step 6: Generate Visualizations ───────────────────────────────────
    print("\n[6/7] Generating visualizations...")

    # Data exploration
    df_with_features = add_time_features(df)
    p = plot_data_exploration(df_with_features, FIG_DIR)
    print(f"      [OK] Data exploration -> {p.name}")

    # Load curve
    p = plot_load_curve(df, FIG_DIR)
    print(f"      [OK] Load curve -> {p.name}")

    # Peak vs off-peak
    df_scenarios = generate_peak_offpeak_scenarios()
    p = plot_peak_offpeak(df_scenarios, FIG_DIR)
    print(f"      [OK] Peak vs off-peak -> {p.name}")

    # Actual vs predicted (all 3 models)
    for name, result, model_obj, p_obj in [
        ("Linear", lr_result, lr, None),
        ("Polynomial", pr_result, pr, poly),
        ("Random Forest", rf_result, rf, None),
    ]:
        p = plot_actual_vs_predicted(y_test.values, result["y_pred"], name, FIG_DIR)
        print(f"      [OK] Actual vs predicted ({name}) -> {p.name}")

    # Model comparison chart
    p = plot_model_comparison(comparison, FIG_DIR)
    print(f"      [OK] Model comparison -> {p.name}")

    # 3D load surface (using Random Forest -- typically best)
    best_model = rf
    p = plot_3d_load_surface(best_model, out_dir=FIG_DIR, scaler=splits["scaler"])
    print(f"      [OK] 3D load surface -> {p.name}")

    # Sensitivity analysis
    p = plot_sensitivity(best_model, FIG_DIR, scaler=splits["scaler"])
    print(f"      [OK] Sensitivity analysis -> {p.name}")

    # Feature importance
    p = plot_feature_importance(rf, feat["feature_names"], FIG_DIR)
    print(f"      [OK] Feature importance -> {p.name}")

    # Prediction distribution
    p = plot_prediction_distribution(y_test.values, rf_result["y_pred"], "Random Forest", FIG_DIR)
    print(f"      [OK] Prediction distribution -> {p.name}")

    # What-if table
    p = plot_whatif_table(best_model, FIG_DIR, scaler=splits["scaler"])
    print(f"      [OK] What-if table -> {p.name}")

    # ── Step 7: Summary ───────────────────────────────────────────────────
    print("\n[7/7] Pipeline complete!")
    print(f"\n  Project structure:")
    print(f"    {DATA_DIR}")
    print(f"    {MODEL_DIR}")
    print(f"    {FIG_DIR}")
    print(f"\n  Key outputs:")
    print(f"    Dataset:      {csv_path}")
    print(f"    Best model:   {best_name}")
    print(f"    Figures:      {len(list(FIG_DIR.glob('*.png')))} PNG files")
    print(f"\n  Station capacity: {STATION_CAPACITY_KW} kW")
    print("=" * 60)


if __name__ == "__main__":
    main()
