"""
Main Pipeline — Energy Consumption Forecast (Assignment 12)

    python src/run_pipeline.py

Steps
    1. Generate the sequential 120-day hourly dataset (Time, Speed, Load -> kWh)
    2. Engineer cyclic time features + lag features, split chronologically
    3. Train three regression models (linear, polynomial, random forest)
    4. Evaluate on the held-out hours and compare
    5. Save models, scaler, metrics and the 24-hour forecast
    6. Generate every figure used in the report and the presentation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from data_generator import (
    generate_dataset,
    generate_day_scenarios,
    save_dataset,
    KERB_MASS_KG,
    PAYLOAD_MAX_KG,
)
from preprocessing import prepare_features, split_and_scale
from models import (
    train_linear_regression,
    train_polynomial_regression,
    train_random_forest,
    evaluate_model,
    compare_models,
    save_model,
    forecast_horizon,
)
from visualization import (
    plot_energy_trend,
    plot_daily_profile,
    plot_weekday_vs_weekend,
    plot_actual_vs_predicted,
    plot_model_comparison,
    plot_3d_energy_surface,
    plot_sensitivity,
    plot_feature_importance,
    plot_error_distribution,
    plot_data_exploration,
    plot_forecast_vs_actual,
    plot_whatif_table,
    plot_residual_autocorrelation,
    build_whatif_rows,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "outputs" / "figures"

HORIZON_HOURS = 24


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("  ENERGY CONSUMPTION FORECAST — ML PIPELINE")
    print("=" * 68)

    # ── 1. Sequential data ────────────────────────────────────────────────
    print("\n[1/7] Generating sequential hourly telemetry (120 days)...")
    df = generate_dataset(num_days=120, seed=42)
    csv_path = save_dataset(df, DATA_DIR / "van_energy_data.csv")
    print(f"      {len(df)} hourly rows over {df['day'].nunique()} days -> {csv_path.name}")
    print(f"      Energy range: {df['energy_kwh'].min():.2f} – "
          f"{df['energy_kwh'].max():.2f} kWh/h")
    print(f"      Speed range:  {df['speed_kmh'].min():.0f} – "
          f"{df['speed_kmh'].max():.0f} km/h   "
          f"Load range: {df['payload_kg'].min():.0f} – {df['payload_kg'].max():.0f} kg")
    print(f"      Parked hours (speed = 0): {(df['speed_kmh'] == 0).sum()}")

    # ── 2. Features ───────────────────────────────────────────────────────
    print("\n[2/7] Engineering features (cyclic Time + Speed + Load)...")
    feat = prepare_features(df)
    frame = feat["frame"]
    splits = split_and_scale(feat["X"], feat["y"])
    print(f"      Model rows: {len(frame)} (all hours usable — no lag warm-up)")
    print(f"      Features:   {feat['feature_names']}")
    print(f"      Split: chronological — train {splits['n_train']} | "
          f"test {len(splits['X_test'])} hours")

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # ── 3. Models ─────────────────────────────────────────────────────────
    print("\n[3/7] Training regression models...")
    lr = train_linear_regression(X_train, y_train)
    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")
    print(f"      Linear Regression      — MAE {lr_result['mae']:.3f} kWh | "
          f"R² {lr_result['r2']:.4f}")

    pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)
    print(f"      Polynomial Regression  — MAE {pr_result['mae']:.3f} kWh | "
          f"R² {pr_result['r2']:.4f}")

    rf = train_random_forest(X_train, y_train, n_estimators=150)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")
    print(f"      Random Forest          — MAE {rf_result['mae']:.3f} kWh | "
          f"R² {rf_result['r2']:.4f}")

    # ── 4. Comparison ─────────────────────────────────────────────────────
    print("\n[4/7] Model comparison on the held-out hours:")
    comparison = compare_models([lr_result, pr_result, rf_result])
    print(comparison.to_string(index=False))
    # Selection rule: the deployed model is the one with the lowest MAE, because
    # MAE is in kWh - the same unit as the answer, and the number an operator can
    # act on. R² and RMSE are reported next to it rather than used as a tiebreak,
    # and the slide deck states plainly where the polynomial is ahead (R²/RMSE)
    # and where it loses badly (MAPE on the parked hours).
    best_name = comparison.loc[comparison["MAE (kWh)"].idxmin(), "Model"]
    print(f"\n      Deployed model (lowest MAE): {best_name}")

    # Degree sweep: shows that the physics is cubic in speed and that a higher
    # degree is not automatically better.
    print("\n      Polynomial degree sweep (same split):")
    sweep = []
    for degree in (1, 2, 3, 4):
        if degree == 1:
            result = lr_result
        else:
            model_d, poly_d = train_polynomial_regression(X_train, y_train,
                                                          degree=degree)
            result = evaluate_model(model_d, X_test, y_test,
                                    f"Polynomial degree {degree}", poly=poly_d)
        sweep.append({"Degree": degree, "MAE (kWh)": result["mae"],
                      "RMSE (kWh)": result["rmse"], "MAPE (%)": result["mape"],
                      "R2": result["r2"]})
    sweep.append({"Degree": "Random Forest", "MAE (kWh)": rf_result["mae"],
                  "RMSE (kWh)": rf_result["rmse"], "MAPE (%)": rf_result["mape"],
                  "R2": rf_result["r2"]})
    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(MODEL_DIR / "degree_sweep.csv", index=False)
    print(sweep_df.to_string(index=False))

    # ── 5. Save artifacts + the 24-hour forecast ──────────────────────────
    print("\n[5/7] Saving models, scaler and the forecast...")
    save_model(lr, MODEL_DIR / "linear_regression.pkl")
    save_model({"model": pr, "poly": poly}, MODEL_DIR / "polynomial_regression.pkl")
    save_model(rf, MODEL_DIR / "random_forest.pkl")
    save_model(splits["scaler"], MODEL_DIR / "scaler.pkl")
    comparison.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    # Honest backtest: forecast the last 24 hours of the test period from the
    # planned duty cycle alone (no peeking at the actual consumption), then
    # compare the forecast with what really happened.
    cut = len(frame) - HORIZON_HOURS
    truth = frame.iloc[cut:]
    start_ts = frame["timestamp"].iloc[cut]

    forecast = forecast_horizon(rf, start_ts, steps=HORIZON_HOURS,
                                scaler=splits["scaler"])
    forecast["actual_kwh"] = truth["energy_kwh"].values
    forecast["error_kwh"] = (forecast["forecast_kwh"] - forecast["actual_kwh"]).round(3)
    forecast.to_csv(MODEL_DIR / "forecast_24h.csv", index=False)

    fc_mae = float(np.mean(np.abs(forecast["error_kwh"])))
    fc_rmse = float(np.sqrt(np.mean(forecast["error_kwh"] ** 2)))
    fc_mape = float(np.mean(np.abs(forecast["error_kwh"] /
                                     forecast["actual_kwh"].replace(0, np.nan))) * 100)
    print(f"      {HORIZON_HOURS}-hour forecast backtest (duty cycle only): "
          f"MAE {fc_mae:.3f} kWh | RMSE {fc_rmse:.3f} | MAPE {fc_mape:.1f}%")
    print(f"      Forecast written to models/forecast_24h.csv")

    # ── 6. Figures ────────────────────────────────────────────────────────
    print("\n[6/7] Generating figures...")
    made = []

    made.append(plot_data_exploration(frame, FIG_DIR))
    made.append(plot_energy_trend(df, FIG_DIR))
    made.append(plot_daily_profile(df, FIG_DIR))
    made.append(plot_weekday_vs_weekend(generate_day_scenarios(), FIG_DIR))

    for name, result in [("Linear", lr_result), ("Polynomial", pr_result),
                         ("Random Forest", rf_result)]:
        made.append(plot_actual_vs_predicted(y_test.values, result["y_pred"],
                                             name, FIG_DIR))

    made.append(plot_model_comparison(comparison, FIG_DIR))
    made.append(plot_3d_energy_surface(rf, FIG_DIR, scaler=splits["scaler"]))
    # Sensitivity is measured inside the payload band the van actually drives
    # with at midday, so the figure never shows extrapolated nonsense.
    midday = frame[(frame["hour"] == 12) & (frame["speed_kmh"] > 0)]
    payload_band = (float(midday["payload_kg"].quantile(0.02)),
                    float(midday["payload_kg"].quantile(0.98)))
    made.append(plot_sensitivity(rf, FIG_DIR, scaler=splits["scaler"],
                                 payload_band=payload_band))
    made.append(plot_feature_importance(rf, feat["feature_names"], FIG_DIR))
    made.append(plot_error_distribution(y_test.values, rf_result["y_pred"],
                                       "Random Forest", FIG_DIR))
    made.append(plot_forecast_vs_actual(
        frame.iloc[:cut][["timestamp", "energy_kwh"]], forecast, FIG_DIR))
    whatif_rows = build_whatif_rows(rf, frame, scaler=splits["scaler"])
    made.append(plot_whatif_table(whatif_rows, FIG_DIR))
    made.append(plot_residual_autocorrelation(y_test.values, rf_result["y_pred"], FIG_DIR))

    for path in made:
        print(f"      [OK] {path.name}")
    print(f"      {len(made)} figure files in outputs/figures")

    # Sequential-feature experiment: what would autoregressive inputs buy, and
    # why are they not used in the deployed model?
    print("\n      Sequential-feature experiment (random forest):")
    feat_seq = prepare_features(df, include_lags=True)
    splits_seq = split_and_scale(feat_seq["X"], feat_seq["y"])
    rf_seq = train_random_forest(splits_seq["X_train"], splits_seq["y_train"])
    seq_result = evaluate_model(rf_seq, splits_seq["X_test"], splits_seq["y_test"],
                                "Random Forest + lags")
    seq_table = pd.DataFrame([
        {"Feature set": "Time + Speed + Load", "MAE (kWh)": rf_result["mae"],
         "R2": rf_result["r2"],
         "Used in the deployed app": "yes - the inputs are known in advance"},
        {"Feature set": "Time + Speed + Load + sequential lags",
         "MAE (kWh)": seq_result["mae"], "R2": seq_result["r2"],
         "Used in the deployed app": "no - the previous hours' consumption is "
                                     "unknown at forecast time"},
    ])
    seq_table.to_csv(MODEL_DIR / "sequential_feature_experiment.csv", index=False)
    print(seq_table.to_string(index=False))
    print(f"      Lag features change MAE by "
          f"{seq_result['mae'] - rf_result['mae']:+.3f} kWh (not deployable, "
          "recorded for the report)")

    # ── 7. Scenario table for the report ──────────────────────────────────
    print("\n[7/7] Engineering read-out (typical values, random forest):")
    scenario_df = pd.DataFrame(
        whatif_rows,
        columns=["Scenario", "Time", "Description", "Speed (km/h)", "Load (kg)",
                 "Energy (kWh)", "kWh/100 km", "Status"],
    )
    scenario_df.to_csv(MODEL_DIR / "whatif_scenarios.csv", index=False)
    print(scenario_df.to_string(index=False))

    deployed = {"Linear Regression": lr_result,
                "Polynomial Regression": pr_result,
                "Random Forest": rf_result}[best_name]
    print("\n" + "=" * 68)
    print(f"  Vehicle: {KERB_MASS_KG} kg kerb + up to {PAYLOAD_MAX_KG} kg payload")
    print(f"  Deployed model: {best_name}  (MAE {deployed['mae']:.3f} kWh | "
          f"R² {deployed['r2']:.4f})")
    print(f"  24-hour forecast backtest MAE: {fc_mae:.3f} kWh")
    print(f"  Figures: {len(made)}  |  Artifacts: {MODEL_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
