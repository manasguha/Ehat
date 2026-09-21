"""
Visualization Module

Generates all publication-quality figures for the charging station project:
    - Load curves (actual vs predicted)
    - Peak vs off-peak comparison
    - 3D load surface
    - Sensitivity analysis
    - Model comparison bar charts
    - Feature importance
    - Actual vs predicted scatter
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap


# Consistent style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

COLORS = {
    "actual": "#2c3e50",
    "predicted": "#e74c3c",
    "peak": "#e74c3c",
    "offpeak": "#3498db",
    "linear": "#95a5a6",
    "poly": "#f39c12",
    "rf": "#27ae60",
    "accent": "#8e44ad",
}


def _save(fig, name: str, out_dir: Path):
    """Save figure to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── 1. Daily Load Curve ────────────────────────────────────────────────────
def plot_load_curve(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot average hourly load curve across all simulated days.
    Shows the characteristic double-peak pattern.
    """
    hourly = df.groupby("hour").agg(
        mean_load=("load_kw", "mean"),
        std_load=("load_kw", "std"),
        mean_vehicles=("vehicles", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Load curve with shaded uncertainty
    ax1.fill_between(
        hourly["hour"],
        hourly["mean_load"] - hourly["std_load"],
        hourly["mean_load"] + hourly["std_load"],
        alpha=0.2, color=COLORS["actual"], label="±1 Std Dev",
    )
    ax1.plot(hourly["hour"], hourly["mean_load"], color=COLORS["actual"],
             linewidth=2.5, marker="o", markersize=5, label="Mean Load")

    # Peak zone shading
    ax1.axvspan(7, 11, alpha=0.08, color=COLORS["peak"], label="Peak Hours")
    ax1.axvspan(17, 21, alpha=0.08, color=COLORS["peak"])
    ax1.axvspan(22, 24, alpha=0.05, color=COLORS["offpeak"])
    ax1.axvspan(0, 7, alpha=0.05, color=COLORS["offpeak"])

    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Load (kW)", color=COLORS["actual"])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax1.set_title("Charging Station Daily Load Curve", fontweight="bold", pad=15)

    # Secondary axis: vehicles
    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["mean_vehicles"], color=COLORS["accent"],
             linewidth=1.5, linestyle="--", alpha=0.7, label="Avg Vehicles")
    ax2.set_ylabel("Number of Vehicles", color=COLORS["accent"])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "01_load_curve", out_dir)


# ── 2. Peak vs Off-Peak Comparison ─────────────────────────────────────────
def plot_peak_offpeak(df_scenarios: pd.DataFrame, out_dir: Path) -> Path:
    """Plot peak day vs off-peak day load curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for scenario, color, marker in [
        ("Peak Day", COLORS["peak"], "o"),
        ("Off-Peak Day", COLORS["offpeak"], "s"),
    ]:
        subset = df_scenarios[df_scenarios["scenario"] == scenario]
        ax.plot(subset["hour"], subset["load_kw"], color=color,
                linewidth=2.5, marker=marker, markersize=5, label=scenario)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Load (kW)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax.set_title("Peak vs Off-Peak Load Comparison", fontweight="bold", pad=15)
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "02_peak_vs_offpeak", out_dir)


# ── 3. Actual vs Predicted ─────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name: str, out_dir: Path) -> Path:
    """Scatter plot of actual vs predicted load."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color=COLORS["predicted"], edgecolors="none")

    # Perfect prediction line
    lims = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "--", color=COLORS["actual"], linewidth=1.5, label="Perfect Prediction")

    ax.set_xlabel("Actual Load (kW)")
    ax.set_ylabel("Predicted Load (kW)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontweight="bold", pad=15)
    ax.set_aspect("equal")
    ax.legend(framealpha=0.9)

    r2 = 1 - np.sum((np.array(y_test) - np.array(y_pred)) ** 2) / np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
    ax.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return _save(fig, f"03_actual_vs_predicted_{model_name.replace(' ', '_').lower()}", out_dir)


# ── 4. Model Comparison ────────────────────────────────────────────────────
def plot_model_comparison(comparison_df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart comparing MAE, RMSE, and R² across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [("MAE (kW)", "MAE", COLORS["linear"]),
               ("RMSE (kW)", "RMSE", COLORS["poly"]),
               ("R²", "R²", COLORS["rf"])]
    colors = [COLORS["linear"], COLORS["poly"], COLORS["rf"]]

    for ax, (col, title, color) in zip(axes, metrics):
        bars = ax.bar(comparison_df["Model"], comparison_df[col], color=colors, edgecolor="white", width=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(col)

        # Add value labels on bars
        for bar, val in zip(bars, comparison_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * bar.get_height(),
                    f"{val:.4f}" if isinstance(val, float) else str(val),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Regression Model Comparison", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "04_model_comparison", out_dir)


# ── 5. 3D Load Surface ─────────────────────────────────────────────────────
def plot_3d_load_surface(model, poly=None, out_dir: Path = None, scaler=None) -> Path:
    """
    3D surface: Time of Day × Vehicles → Load.
    Uses the best model to generate the surface.
    """
    from mpl_toolkits.mplot3d import Axes3D

    hours = np.linspace(0, 23, 100)
    vehicles = np.linspace(0, 60, 100)
    H, V = np.meshgrid(hours, vehicles)

    # Convert to model features
    hour_sin = np.sin(2 * np.pi * H / 24)
    hour_cos = np.cos(2 * np.pi * H / 24)

    X_grid = np.column_stack([hour_sin.ravel(), hour_cos.ravel(), V.ravel()])
    if scaler is not None:
        X_grid = scaler.transform(X_grid)
    if poly is not None:
        X_grid = poly.transform(X_grid)

    Z = model.predict(X_grid).reshape(H.shape)
    Z = np.clip(Z, 0, 300)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(H, V, Z, cmap="RdYlGn_r", alpha=0.85,
                           edgecolor="none", antialiased=True)

    ax.set_xlabel("Hour of Day", labelpad=10)
    ax.set_ylabel("Number of Vehicles", labelpad=10)
    ax.set_zlabel("Load (kW)", labelpad=10)
    ax.set_title("3D Load Surface: Time × Vehicles → Load", fontweight="bold", pad=20)
    ax.view_init(elev=25, azim=225)

    fig.colorbar(surf, shrink=0.5, aspect=15, label="Load (kW)")

    fig.tight_layout()
    return _save(fig, "05_3d_load_surface", out_dir)


# ── 6. Sensitivity Analysis ────────────────────────────────────────────────
def plot_sensitivity(model, out_dir: Path, poly=None, scaler=None) -> Path:
    """
    Two-panel sensitivity plot:
        Left:  Fix time=18:00, vary vehicles
        Right: Fix vehicles=40, vary time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Vehicles sensitivity at 18:00
    veh_range = np.arange(0, 61, 5)
    loads_v = []
    for v in veh_range:
        hour_sin = np.sin(2 * np.pi * 18 / 24)
        hour_cos = np.cos(2 * np.pi * 18 / 24)
        X = np.array([[hour_sin, hour_cos, v]])
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)
        loads_v.append(model.predict(X)[0])

    ax1.plot(veh_range, loads_v, color=COLORS["peak"], linewidth=2.5, marker="o", markersize=5)
    ax1.fill_between(veh_range, loads_v, alpha=0.15, color=COLORS["peak"])
    ax1.set_xlabel("Number of Vehicles")
    ax1.set_ylabel("Predicted Load (kW)")
    ax1.set_title("Sensitivity: Vehicles (at 18:00)", fontweight="bold")
    ax1.axhline(y=300, color="red", linestyle="--", alpha=0.5, label="Station Capacity (300 kW)")
    ax1.legend()

    # Panel 2: Time sensitivity at 40 vehicles
    hour_range = np.linspace(0, 23, 96)
    loads_h = []
    for h in hour_range:
        hour_sin = np.sin(2 * np.pi * h / 24)
        hour_cos = np.cos(2 * np.pi * h / 24)
        X = np.array([[hour_sin, hour_cos, 40]])
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)
        loads_h.append(model.predict(X)[0])

    ax2.plot(hour_range, loads_h, color=COLORS["offpeak"], linewidth=2.5)
    ax2.fill_between(hour_range, loads_h, alpha=0.15, color=COLORS["offpeak"])
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Predicted Load (kW)")
    ax2.set_title("Sensitivity: Time of Day (at 40 vehicles)", fontweight="bold")
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)])
    ax2.axhline(y=300, color="red", linestyle="--", alpha=0.5, label="Station Capacity (300 kW)")
    ax2.legend()

    fig.suptitle("Sensitivity Analysis", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "06_sensitivity_analysis", out_dir)


# ── 7. Feature Importance (Random Forest) ──────────────────────────────────
def plot_feature_importance(model, feature_names: list, out_dir: Path) -> Path:
    """Horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(
        [feature_names[i] for i in indices],
        importances[indices],
        color=COLORS["rf"],
        edgecolor="white",
        height=0.5,
    )

    for bar, val in zip(bars, importances[indices]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance — Random Forest", fontweight="bold", pad=15)

    fig.tight_layout()
    return _save(fig, "07_feature_importance", out_dir)


# ── 8. Prediction Distribution ─────────────────────────────────────────────
def plot_prediction_distribution(y_test, y_pred, model_name: str, out_dir: Path) -> Path:
    """Histogram of prediction errors."""
    errors = np.array(y_pred) - np.array(y_test)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error distribution
    ax1.hist(errors, bins=40, color=COLORS["predicted"], alpha=0.7, edgecolor="white")
    ax1.axvline(x=0, color=COLORS["actual"], linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Prediction Error (kW)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Error Distribution — {model_name}", fontweight="bold")

    # Cumulative error
    sorted_errors = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative, color=COLORS["accent"], linewidth=2)
    ax2.set_xlabel("Absolute Error (kW)")
    ax2.set_ylabel("Cumulative Fraction")
    ax2.set_title(f"Cumulative Error — {model_name}", fontweight="bold")
    ax2.axhline(y=0.9, color="gray", linestyle=":", alpha=0.5)
    ax2.text(sorted_errors[int(0.9 * len(sorted_errors))], 0.92,
             "90th percentile", fontsize=9, color="gray")

    fig.tight_layout()
    return _save(fig, f"08_prediction_distribution_{model_name.replace(' ', '_').lower()}", out_dir)


# ── 9. Data Exploration ────────────────────────────────────────────────────
def plot_data_exploration(df: pd.DataFrame, out_dir: Path) -> Path:
    """Multi-panel EDA figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Load distribution
    axes[0, 0].hist(df["load_kw"], bins=50, color=COLORS["actual"], alpha=0.7, edgecolor="white")
    axes[0, 0].set_xlabel("Load (kW)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Load Distribution", fontweight="bold")

    # 2. Vehicles vs Load
    axes[0, 1].scatter(df["vehicles"], df["load_kw"], alpha=0.15, s=8, color=COLORS["rf"])
    axes[0, 1].set_xlabel("Number of Vehicles")
    axes[0, 1].set_ylabel("Load (kW)")
    axes[0, 1].set_title("Vehicles vs Load", fontweight="bold")

    # 3. Load by day type
    for dt, color in [("weekday", COLORS["peak"]), ("weekend", COLORS["offpeak"]), ("holiday", COLORS["accent"])]:
        subset = df[df["day_type"] == dt]
        hourly = subset.groupby("hour")["load_kw"].mean()
        axes[1, 0].plot(hourly.index, hourly.values, color=color, linewidth=2, label=dt)
    axes[1, 0].set_xlabel("Hour of Day")
    axes[1, 0].set_ylabel("Mean Load (kW)")
    axes[1, 0].set_title("Load by Day Type", fontweight="bold")
    axes[1, 0].legend()

    # 4. Box plot by period
    periods = ["morning", "afternoon", "evening", "night"]
    period_data = [df[df["period"] == p]["load_kw"].values for p in periods]
    bp = axes[1, 1].boxplot(period_data, tick_labels=periods, patch_artist=True)
    colors_bp = [COLORS["offpeak"], COLORS["poly"], COLORS["peak"], COLORS["actual"]]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1, 1].set_ylabel("Load (kW)")
    axes[1, 1].set_title("Load by Time Period", fontweight="bold")

    fig.suptitle("Data Exploration", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, "09_data_exploration", out_dir)


# ── 10. What-If Scenario Table ─────────────────────────────────────────────
def plot_whatif_table(model, out_dir: Path, poly=None, scaler=None) -> Path:
    """Visual table of what-if scenarios."""
    scenarios = [
        ("A", "08:00", 20),
        ("B", "18:00", 20),
        ("C", "18:00", 40),
        ("D", "20:00", 60),
        ("E", "03:00", 5),
        ("F", "12:00", 35),
    ]

    rows = []
    for label, time_str, veh in scenarios:
        h = int(time_str.split(":")[0])
        hour_sin = np.sin(2 * np.pi * h / 24)
        hour_cos = np.cos(2 * np.pi * h / 24)
        X = np.array([[hour_sin, hour_cos, veh]])
        if scaler is not None:
            X = scaler.transform(X)
        if poly is not None:
            X = poly.transform(X)
        pred = model.predict(X)[0]
        status = "PEAK WARNING" if pred > 270 else "NORMAL"
        rows.append((label, time_str, veh, round(pred, 1), status))

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    col_labels = ["Scenario", "Time", "Vehicles", "Predicted Load (kW)", "Status"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style status cells
    for i, row in enumerate(rows, start=1):
        if row[4] == "PEAK WARNING":
            table[i, 4].set_facecolor("#fdecea")
            table[i, 4].set_text_props(color="#c0392b", fontweight="bold")
        else:
            table[i, 4].set_facecolor("#eafaf1")
            table[i, 4].set_text_props(color="#27ae60")

    ax.set_title("What-If Scenario Analysis", fontweight="bold", pad=20, fontsize=13)
    fig.tight_layout()
    return _save(fig, "10_whatif_table", out_dir)
