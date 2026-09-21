"""
Visualization Module — Energy Consumption Forecast

Generates every figure used in the report, the presentation and the dashboard:

    01_energy_trend                     the sequential series (assignment: "trend graph")
    02_daily_profile                    mean hour-of-day profile with speed and load
    03_weekday_vs_weekend                working day vs weekend duty cycle
    04_actual_vs_predicted_*            per-model scatter on the held-out hours
    05_model_comparison                 MAE / RMSE / MAPE / R² bars
    06_3d_energy_surface                speed x load -> kWh
    07_sensitivity_analysis             one input swept, the other fixed
    08_feature_importance               random forest ranking
    09_error_distribution_*             residual histogram + cumulative error
    10_data_exploration                 EDA panel
    11_forecast_vs_actual               24-hour recursive forecast vs truth
    12_whatif_table                     scenario table with efficiency status
    13_residual_autocorrelation         did the model remove the sequential structure?
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data_generator import PAYLOAD_MAX_KG
from preprocessing import FEATURE_COLS, scale_features

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
    "forecast": "#e74c3c",
    "linear": "#95a5a6",
    "poly": "#f39c12",
    "rf": "#27ae60",
    "speed": "#2980b9",
    "load": "#8e44ad",
    "accent": "#8e44ad",
}

# Efficiency thresholds (kWh per 100 km), taken from the simulated fleet:
# the 75th and 90th percentiles of the moving hours, not invented round numbers.
EFFICIENCY_EFFICIENT_MAX = 21.5     # p75 of moving-hour consumption
EFFICIENCY_HIGH_MAX = 24.5          # p90
PARKED_ENERGY_MAX = 1.5             # kWh/h: auxiliary load only


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _predict_frame(model, frame: pd.DataFrame, poly=None, scaler=None):
    """Apply the exact inference chain: scale -> polynomial -> model."""
    X = frame[FEATURE_COLS].copy()
    if scaler is not None:
        X = scale_features(X, scaler)
    X_eval = poly.transform(X) if poly is not None else X
    return np.asarray(model.predict(X_eval), dtype=float)


def _feature_row(hour, dow, speed, payload):
    """One feature row: the cyclic Time inputs plus Speed and Load."""
    return {
        "hour_sin": np.sin(2 * np.pi * (hour % 24) / 24),
        "hour_cos": np.cos(2 * np.pi * (hour % 24) / 24),
        "dow_sin": np.sin(2 * np.pi * (dow % 7) / 7),
        "dow_cos": np.cos(2 * np.pi * (dow % 7) / 7),
        "speed_kmh": float(speed),
        "payload_kg": float(payload),
    }


def status_for(efficiency_kwh_100km: float, energy_kwh: float) -> str:
    """Human-readable verdict for a predicted hour (also used by the dashboard)."""
    if not np.isfinite(efficiency_kwh_100km) or energy_kwh <= PARKED_ENERGY_MAX:
        return "PARKED"
    if efficiency_kwh_100km < EFFICIENCY_EFFICIENT_MAX:
        return "EFFICIENT"
    if efficiency_kwh_100km < EFFICIENCY_HIGH_MAX:
        return "HIGH DRAW"
    return "CRITICAL DRAW"


# ── 1. The sequential series ────────────────────────────────────────────────
def plot_energy_trend(df: pd.DataFrame, out_dir: Path, days: int = 14) -> Path:
    """Hourly energy consumption over a two-week window, with a 24-hour mean."""
    window = df.head(days * 24).copy()
    rolling = df["energy_kwh"].rolling(24).mean().head(days * 24)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(window["timestamp"], window["energy_kwh"], color=COLORS["actual"],
            linewidth=1.4, alpha=0.85, label="Hourly consumption")
    ax.plot(window["timestamp"], rolling, color=COLORS["forecast"],
            linewidth=2.6, label="24-hour rolling mean")

    ax.set_xlabel("Date")
    ax.set_ylabel("Energy consumption (kWh per hour)")
    ax.set_title(f"Sequential Energy Consumption — first {days} days",
                 fontweight="bold", pad=15)
    ax.legend(framealpha=0.9, loc="upper right")
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    return _save(fig, "01_energy_trend", out_dir)


# ── 2. Daily profile ───────────────────────────────────────────────────────
def plot_daily_profile(df: pd.DataFrame, out_dir: Path) -> Path:
    """Mean hour-of-day energy with speed and load on secondary axes."""
    hourly = df.groupby("hour").agg(
        mean_energy=("energy_kwh", "mean"),
        std_energy=("energy_kwh", "std"),
        mean_speed=("speed_kmh", "mean"),
        mean_payload=("payload_kg", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.fill_between(hourly["hour"],
                     hourly["mean_energy"] - hourly["std_energy"],
                     hourly["mean_energy"] + hourly["std_energy"],
                     alpha=0.18, color=COLORS["actual"], label="±1 Std Dev")
    ax1.plot(hourly["hour"], hourly["mean_energy"], color=COLORS["actual"],
             linewidth=2.6, marker="o", markersize=4, label="Mean energy")

    ax1.axvspan(7, 10, alpha=0.07, color=COLORS["forecast"])
    ax1.axvspan(17, 20, alpha=0.07, color=COLORS["forecast"])
    ax1.text(8.5, ax1.get_ylim()[1] * 0.94, "morning\nrush", ha="center",
             fontsize=8.5, color=COLORS["forecast"])
    ax1.text(18.5, ax1.get_ylim()[1] * 0.94, "evening\nrush", ha="center",
             fontsize=8.5, color=COLORS["forecast"])

    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Energy consumption (kWh)", color=COLORS["actual"])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax1.set_title("Daily Energy Consumption Profile with Its Inputs",
                  fontweight="bold", pad=15)

    ax2 = ax1.twinx()
    ax2.plot(hourly["hour"], hourly["mean_speed"], color=COLORS["speed"],
             linestyle="--", linewidth=1.8, label="Mean speed (km/h)")
    ax2.plot(hourly["hour"], hourly["mean_payload"], color=COLORS["load"],
             linestyle=":", linewidth=2.0, label="Mean load (kg)")
    ax2.set_ylabel("Speed (km/h)  /  Load (kg)")

    lines = (ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0])
    labels = (ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1])
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "02_daily_profile", out_dir)


# ── 3. Working day vs weekend ──────────────────────────────────────────────
def plot_weekday_vs_weekend(df_scenarios: pd.DataFrame, out_dir: Path) -> Path:
    """Two duty cycles compared: energy on top, speed underneath."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    styles = [("Working Day", COLORS["actual"], "o"),
              ("Weekend", COLORS["speed"], "s")]
    for scenario, color, marker in styles:
        subset = df_scenarios[df_scenarios["scenario"] == scenario]
        ax1.plot(subset["hour"], subset["energy_kwh"], color=color, linewidth=2.4,
                 marker=marker, markersize=4, label=scenario)
        ax2.plot(subset["hour"], subset["speed_kmh"], color=color, linewidth=2.0,
                 marker=marker, markersize=3.5, label=f"{scenario} speed")

    ax1.set_ylabel("Energy (kWh)")
    ax1.set_title("Working Day vs Weekend Duty Cycle", fontweight="bold", pad=15)
    ax1.legend(framealpha=0.9)

    ax2.set_xlabel("Hour of day")
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    ax2.legend(framealpha=0.9, ncol=2)

    fig.tight_layout()
    return _save(fig, "03_weekday_vs_weekend", out_dir)


# ── 4. Actual vs predicted ─────────────────────────────────────────────────
def plot_actual_vs_predicted(y_test, y_pred, model_name: str, out_dir: Path) -> Path:
    """Scatter of predicted against observed energy on the held-out hours."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, s=18, color=COLORS["forecast"],
               edgecolors="none")

    lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
    ax.plot(lims, lims, "--", color=COLORS["actual"], linewidth=1.5,
            label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual energy (kWh)")
    ax.set_ylabel("Predicted energy (kWh)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontweight="bold", pad=15)
    ax.set_aspect("equal")

    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    ax.text(0.05, 0.92, f"R² = {r2:.4f}\nMAE = {mae:.2f} kWh",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))
    ax.legend(framealpha=0.9, loc="lower right")

    fig.tight_layout()
    return _save(fig, f"04_actual_vs_predicted_{model_name.replace(' ', '_').lower()}",
                 out_dir)


# ── 5. Model comparison ────────────────────────────────────────────────────
def plot_model_comparison(comparison: pd.DataFrame, out_dir: Path) -> Path:
    """Four metrics side by side, each with the best model marked."""
    metrics = [("MAE (kWh)", "MAE — lower is better"),
               ("RMSE (kWh)", "RMSE — lower is better"),
               ("MAPE (%)", "MAPE — lower is better"),
               ("R²", "R² — higher is better")]
    colors = [COLORS["linear"], COLORS["poly"], COLORS["rf"]]
    higher_better = {"R²"}

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    for ax, (col, title) in zip(axes, metrics):
        values = comparison[col].values
        bars = ax.bar(comparison["Model"], values, color=colors,
                      edgecolor="white", width=0.62)
        best = int(np.argmax(values) if col in higher_better else np.argmin(values))
        bars[best].set_edgecolor("#1e8449")
        bars[best].set_linewidth(2.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.22)
        ax.tick_params(axis="x", rotation=18)

    fig.suptitle("Energy Consumption Model Comparison (held-out hours)",
                 fontweight="bold", fontsize=14, y=1.03)
    fig.tight_layout()
    return _save(fig, "05_model_comparison", out_dir)


# ── 6. 3D surface ──────────────────────────────────────────────────────────
def plot_3d_energy_surface(model, out_dir: Path, poly=None, scaler=None,
                           hour: float = 12.0, dow: int = 2) -> Path:
    """
    Speed x Load -> energy consumption during a free-flowing midday hour.

    The recent-consumption lags are held at their typical value, so the surface
    shows the pure effect of the two physical inputs.
    """
    speeds = np.linspace(0, 100, 90)
    payloads = np.linspace(0, PAYLOAD_MAX_KG, 90)
    S, P = np.meshgrid(speeds, payloads)

    frame = pd.DataFrame([_feature_row(hour, dow, s, p)
                          for s, p in zip(S.ravel(), P.ravel())])
    Z = _predict_frame(model, frame, poly, scaler).reshape(S.shape)

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(S, P, Z, cmap="viridis", alpha=0.9, edgecolor="none",
                           antialiased=True)

    ax.set_xlabel("Speed (km/h)", labelpad=10)
    ax.set_ylabel("Load / payload (kg)", labelpad=10)
    ax.set_zlabel("Energy (kWh)", labelpad=10)
    ax.set_title("3D Energy Surface: Speed x Load -> Consumption\n"
                 f"(midday duty cycle, hour {int(hour):02d}:00)",
                 fontweight="bold", pad=18)
    ax.view_init(elev=24, azim=232)
    fig.colorbar(surf, shrink=0.55, aspect=14, label="Energy (kWh)")

    fig.tight_layout()
    return _save(fig, "06_3d_energy_surface", out_dir)


# ── 7. Sensitivity ─────────────────────────────────────────────────────────
def plot_sensitivity(model, out_dir: Path, poly=None, scaler=None,
                     payload_band=(200.0, 1200.0)) -> Path:
    """
    Left:  speed swept at two payloads -> shows the non-linear drag effect and
           the interaction between speed and load.
    Right: payload swept at two speeds -> rolling resistance effect, and the
           rush-hour penalty at 25 km/h.

    The payload sweep stays inside the band the van actually drives with at that
    hour (passed in by the pipeline). Sweeping to an invented 0 kg at midday
    would be extrapolation, and the model is not allowed to pretend otherwise.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    speeds = np.arange(0, 101, 2.5)
    for payload, color, label in [(0, COLORS["speed"], "empty (0 kg)"),
                                  (PAYLOAD_MAX_KG, COLORS["forecast"],
                                   f"full load ({PAYLOAD_MAX_KG} kg)")]:
        vals = [_predict_frame(model, pd.DataFrame([_feature_row(12, 2, s, payload)]),
                               poly, scaler)[0] for s in speeds]
        ax1.plot(speeds, vals, color=color, linewidth=2.5, marker="", label=label)
    ax1.axvspan(0, 30, alpha=0.08, color=COLORS["forecast"])
    ax1.text(15, ax1.get_ylim()[1] * 0.05, "urban\npenalty zone", fontsize=8.5,
             ha="center", color=COLORS["forecast"])
    ax1.set_xlabel("Speed (km/h)")
    ax1.set_ylabel("Predicted energy (kWh)")
    ax1.set_title("Sensitivity: Speed (at midday)", fontweight="bold")
    ax1.legend(framealpha=0.9)

    payloads = np.arange(payload_band[0], payload_band[1] + 1, 20)
    for speed, color, label in [(25, COLORS["forecast"], "rush hour (25 km/h)"),
                                (80, COLORS["speed"], "free flow (80 km/h)")]:
        vals = [_predict_frame(model, pd.DataFrame([_feature_row(12, 2, speed, p)]),
                               poly, scaler)[0] for p in payloads]
        ax2.plot(payloads, vals, color=color, linewidth=2.5, label=label)
    ax2.set_xlabel(f"Payload (kg) — observed band at 12:00: "
                   f"{payload_band[0]:.0f}-{payload_band[1]:.0f} kg")
    ax2.set_ylabel("Predicted energy (kWh)")
    ax2.set_title("Sensitivity: Load (at midday)", fontweight="bold")
    ax2.legend(framealpha=0.9)

    fig.suptitle("Sensitivity Analysis", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "07_sensitivity_analysis", out_dir)


# ── 8. Feature importance ──────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list, out_dir: Path) -> Path:
    """Horizontal ranking of the random forest importances."""
    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh([feature_names[i] for i in order], importances[order],
                   color=COLORS["rf"], edgecolor="white", height=0.55)
    for bar, val in zip(bars, importances[order]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.set_xlim(0, importances.max() * 1.18)
    ax.set_title("Feature Importance — Random Forest", fontweight="bold", pad=15)

    fig.tight_layout()
    return _save(fig, "08_feature_importance", out_dir)


# ── 9. Error distribution ──────────────────────────────────────────────────
def plot_error_distribution(y_test, y_pred, model_name: str, out_dir: Path) -> Path:
    """Residual histogram plus cumulative absolute error."""
    y_test = np.asarray(y_test, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_test

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(errors, bins=45, color=COLORS["forecast"], alpha=0.75,
             edgecolor="white")
    ax1.axvline(0, color=COLORS["actual"], linestyle="--", linewidth=1.6)
    ax1.axvline(errors.mean(), color=COLORS["rf"], linestyle="-", linewidth=2,
                label=f"mean bias {errors.mean():+.2f} kWh")
    ax1.set_xlabel("Prediction error (kWh)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Error Distribution — {model_name}", fontweight="bold")
    ax1.legend(framealpha=0.9)

    absolute = np.sort(np.abs(errors))
    cumulative = np.arange(1, len(absolute) + 1) / len(absolute)
    ax2.plot(absolute, cumulative, color=COLORS["accent"], linewidth=2.4)
    ax2.axhline(0.9, color="gray", linestyle=":", alpha=0.7)
    p90 = absolute[min(int(0.9 * len(absolute)), len(absolute) - 1)]
    ax2.axvline(p90, color="gray", linestyle=":", alpha=0.7)
    ax2.annotate(f"90% of hours within {p90:.2f} kWh",
                 xy=(p90, 0.9), xytext=(p90 * 0.35, 0.62), fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color="gray"))
    ax2.set_xlabel("Absolute error (kWh)")
    ax2.set_ylabel("Cumulative fraction of hours")
    ax2.set_title(f"Cumulative Error — {model_name}", fontweight="bold")

    fig.tight_layout()
    return _save(fig, f"09_error_distribution_{model_name.replace(' ', '_').lower()}",
                 out_dir)


# ── 10. Data exploration ───────────────────────────────────────────────────
def plot_data_exploration(df: pd.DataFrame, out_dir: Path) -> Path:
    """EDA panel: target shape, the two inputs, and the traffic-penalty effect."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    axes[0, 0].hist(df["energy_kwh"], bins=55, color=COLORS["actual"],
                    alpha=0.8, edgecolor="white")
    axes[0, 0].axvline(df["energy_kwh"].mean(), color=COLORS["forecast"],
                       linestyle="--", linewidth=2,
                       label=f"mean {df['energy_kwh'].mean():.2f} kWh")
    axes[0, 0].set_xlabel("Energy consumption (kWh per hour)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Target distribution (zero-inflated by parking)",
                         fontweight="bold")
    axes[0, 0].legend(framealpha=0.9)

    axes[0, 1].scatter(df["speed_kmh"], df["energy_kwh"], alpha=0.18, s=8,
                       c=df["payload_kg"], cmap="viridis")
    axes[0, 1].set_xlabel("Speed (km/h)")
    axes[0, 1].set_ylabel("Energy (kWh)")
    axes[0, 1].set_title("Speed vs energy (colour = load)", fontweight="bold")

    axes[1, 0].scatter(df["payload_kg"], df["energy_kwh"], alpha=0.18, s=8,
                       color=COLORS["load"])
    axes[1, 0].set_xlabel("Payload (kg)")
    axes[1, 0].set_ylabel("Energy (kWh)")
    axes[1, 0].set_title("Load vs energy (moving hours only)", fontweight="bold")

    moving = df[df["speed_kmh"] > 0]
    regimes = ["urban", "suburban", "motorway"]
    data = [moving[moving["driving_regime"] == r]["efficiency_kwh_100km"].values
            for r in regimes]
    bp = axes[1, 1].boxplot(data, tick_labels=regimes, patch_artist=True)
    for patch, color in zip(bp["boxes"],
                            [COLORS["forecast"], COLORS["poly"], COLORS["speed"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    axes[1, 1].set_ylabel("kWh per 100 km")
    axes[1, 1].set_title("Efficiency by driving regime", fontweight="bold")

    fig.suptitle("Data Exploration", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    return _save(fig, "10_data_exploration", out_dir)


# ── 11. Forecast vs actual ─────────────────────────────────────────────────
def plot_forecast_vs_actual(history: pd.DataFrame, forecast: pd.DataFrame,
                            out_dir: Path, history_hours: int = 48) -> Path:
    """The trend graph deliverable: observed hours, then a 24-hour forecast."""
    hist = history.tail(history_hours)

    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    ax.plot(hist["timestamp"], hist["energy_kwh"], color=COLORS["actual"],
            linewidth=2.4, marker="o", markersize=4, label="Observed consumption")
    ax.plot(forecast["timestamp"], forecast["forecast_kwh"],
            color=COLORS["forecast"], linewidth=2.6, linestyle="--", marker="s",
            markersize=4, label="Forecast (recursive, 24 h ahead)")

    # Bridge the two lines so the eye reads one continuous trend
    bridge_x = [hist["timestamp"].iloc[-1], forecast["timestamp"].iloc[0]]
    bridge_y = [hist["energy_kwh"].iloc[-1], forecast["forecast_kwh"].iloc[0]]
    ax.plot(bridge_x, bridge_y, color=COLORS["forecast"], linewidth=2.6,
            linestyle="--")

    split_x = forecast["timestamp"].iloc[0]
    ax.axvline(split_x, color="gray", linestyle=":", alpha=0.8)
    ax.annotate("forecast starts", xy=(split_x, ax.get_ylim()[1] * 0.92),
                xytext=(-12, 0), textcoords="offset points", ha="right",
                fontsize=9.5, color="gray")

    ax.set_xlabel("Date and time")
    ax.set_ylabel("Energy consumption (kWh per hour)")
    ax.set_title("Trend Graph — Observed Hours and the Next 24 Hours",
                 fontweight="bold", pad=15)
    ax.legend(framealpha=0.9)
    fig.autofmt_xdate(rotation=30)

    fig.tight_layout()
    return _save(fig, "11_forecast_vs_actual", out_dir)


# ── 12. What-if table ──────────────────────────────────────────────────────
# One scenario list shared by the pipeline, the figure and the dashboard, so the
# numbers printed in the report are the numbers drawn on the slide.
WHATIF_SCENARIOS = [
    (2, "Parked overnight", 0, 0),
    (5, "Early motorway run, fully loaded", 80, 1200),
    (8, "Morning city rush, loaded", 25, 900),
    (12, "Midday free flow, half load", 60, 500),
    (14, "Afternoon motorway, fully loaded", 90, 1200),
    (18, "Evening city rush, empty", 22, 100),
    (19, "Return leg, light load", 75, 250),
]


def build_whatif_rows(model, frame: pd.DataFrame, poly=None, scaler=None) -> list:
    """
    Evaluate the shared scenario list and return the rows used by both the
    figure and the CSV, so the report and the slide can never disagree.
    """
    rows = []
    for i, (hour, desc, speed, payload) in enumerate(WHATIF_SCENARIOS):
        label = chr(ord("A") + i)
        pred = float(_predict_frame(
            model, pd.DataFrame([_feature_row(hour, 1, speed, payload)]),
            poly, scaler)[0])
        pred = max(pred, 0.0)
        distance = speed * 1.0
        efficiency = (pred / distance * 100) if distance > 0.5 else float("nan")
        eff_text = "—" if np.isnan(efficiency) else f"{efficiency:.1f}"
        rows.append((label, f"{hour:02d}:00", desc, speed, payload,
                     round(pred, 2), eff_text,
                     status_for(efficiency, pred)))
    return rows


def plot_whatif_table(rows: list, out_dir: Path) -> Path:
    """Render the scenario table produced by build_whatif_rows()."""
    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.axis("off")
    col_labels = ["", "Time", "Scenario", "Speed (km/h)", "Load (kg)",
                  "Energy (kWh)", "kWh/100 km", "Status"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.15, 1.8)

    status_colors = {"EFFICIENT": ("#eafaf1", "#1e8449"),
                     "HIGH DRAW": ("#fdebd0", "#b9770e"),
                     "CRITICAL DRAW": ("#fdecea", "#c0392b"),
                     "PARKED": ("#eef2f6", "#5b6b7d")}
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#0a1a2f")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(rows, start=1):
        fill, text = status_colors[row[-1]]
        table[i, len(col_labels) - 1].set_facecolor(fill)
        table[i, len(col_labels) - 1].set_text_props(color=text, fontweight="bold")

    ax.set_title("What-If Scenario Analysis", fontweight="bold", pad=22,
                 fontsize=13)
    fig.tight_layout()
    return _save(fig, "12_whatif_table", out_dir)


# ── 13. Residual autocorrelation ───────────────────────────────────────────
def plot_residual_autocorrelation(series, y_pred, out_dir: Path,
                                  max_lag: int = 48) -> Path:
    """
    A forecasting check, not a decoration. The raw series is heavily
    autocorrelated by construction; whatever is left in the residuals is the
    structure the inputs (Time, Speed, Load) could not explain. Reading that gap
    honestly is what justifies the future-work section.
    """
    series = np.asarray(series, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = series - y_pred

    def acf(values, lags):
        x = values - values.mean()
        denom = np.sum(x ** 2)
        return np.array([np.sum(x[:len(x) - k] * x[k:]) / denom
                         for k in range(1, lags + 1)])

    lags = np.arange(1, max_lag + 1)
    acf_resid = acf(residuals, max_lag)
    acf_series = acf(series, max_lag)
    band = 1.96 / np.sqrt(len(residuals))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(lags - 0.18, acf_series, width=0.36, color=COLORS["linear"],
           label="Raw series")
    ax.bar(lags + 0.18, acf_resid, width=0.36, color=COLORS["rf"],
           label="Model residuals")
    ax.axhline(band, color=COLORS["forecast"], linestyle="--", linewidth=1.2,
               label="95% confidence band")
    ax.axhline(-band, color=COLORS["forecast"], linestyle="--", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Residual Autocorrelation — has the model used the sequential structure?",
                 fontweight="bold", pad=15)
    ax.legend(framealpha=0.9, ncol=3)

    fig.tight_layout()
    return _save(fig, "13_residual_autocorrelation", out_dir)
