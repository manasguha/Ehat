"""
Interactive Dashboard — Energy Consumption Forecast

    streamlit run dashboard/app.py

Pages
    Overview              headline numbers and the trend of the sequential series
    Energy Predictor      Time + Speed + Load -> kWh, efficiency and status
    Forecast              multi-step forecast from the planned duty cycle
    Model Comparison      metrics, actual-vs-predicted, degree sweep, lag experiment
    Trend & Seasonality   daily profile, working day vs weekend, efficiency regimes
    Sensitivity            speed and load swept one at a time
    What-If Scenarios     the scenario table the report quotes

The models are loaded from the pickles that src/run_pipeline.py writes into
models/ (see load_models below), so the app does not retrain on startup. If an
artifact is missing or incompatible the app falls back to training in-process
instead of crashing in front of an audience.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_generator import (
    KERB_MASS_KG,
    PAYLOAD_MAX_KG,
    generate_dataset,
    generate_day_scenarios,
)
from preprocessing import prepare_features, split_and_scale
from models import (
    compare_models,
    evaluate_model,
    forecast_horizon,
    load_model,
    predict_energy,
    train_linear_regression,
    train_polynomial_regression,
    train_random_forest,
)
from visualization import (
    EFFICIENCY_EFFICIENT_MAX,
    EFFICIENCY_HIGH_MAX,
    PARKED_ENERGY_MAX,
    WHATIF_SCENARIOS,
    build_whatif_rows,
    status_for,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "van_energy_data.csv"

st.set_page_config(page_title="Energy Consumption Forecast",
                   layout="wide", initial_sidebar_state="expanded")


# ── Data and models ─────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the simulated dataset, regenerating only if the CSV is absent."""
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV, parse_dates=["timestamp"])
    return generate_dataset(num_days=120, seed=42)


def train_models(df):
    """Fallback: train all three models from scratch, in-process."""
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    lr = train_linear_regression(X_train, y_train)
    pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
    rf = train_random_forest(X_train, y_train, n_estimators=150)

    return {
        "lr": lr, "pr": pr, "poly": poly, "rf": rf,
        "lr_result": evaluate_model(lr, X_test, y_test, "Linear Regression"),
        "pr_result": evaluate_model(pr, X_test, y_test, "Polynomial Regression",
                                    poly=poly),
        "rf_result": evaluate_model(rf, X_test, y_test, "Random Forest"),
        "comparison": compare_models([
            evaluate_model(lr, X_test, y_test, "Linear Regression"),
            evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly),
            evaluate_model(rf, X_test, y_test, "Random Forest"),
        ]),
        "X_test": X_test, "y_test": y_test, "frame": feat["frame"],
        "scaler": splits["scaler"],
        "source": "trained in-app (artifacts missing)",
    }


@st.cache_resource
def load_models(df):
    """
    Load the artifacts written by src/run_pipeline.py and re-evaluate them on the
    same chronological test split, so the on-screen table is always computed from
    the loaded models rather than copied from a file.
    """
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    X_test, y_test = splits["X_test"], splits["y_test"]

    try:
        lr = load_model(MODELS_DIR / "linear_regression.pkl")
        bundle = load_model(MODELS_DIR / "polynomial_regression.pkl")
        pr, poly = (bundle["model"], bundle["poly"]) if isinstance(bundle, dict) \
            else bundle
        rf = load_model(MODELS_DIR / "random_forest.pkl")
        scaler = load_model(MODELS_DIR / "scaler.pkl")
    except Exception as exc:  # missing or incompatible artifacts
        st.warning(f"Could not load the saved models ({exc}). "
                   "Training from scratch instead — this takes a few seconds.")
        return train_models(df)

    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")

    return {
        "lr": lr, "pr": pr, "poly": poly, "rf": rf,
        "lr_result": lr_result, "pr_result": pr_result, "rf_result": rf_result,
        "comparison": compare_models([lr_result, pr_result, rf_result]),
        "X_test": X_test, "y_test": y_test, "frame": feat["frame"],
        "scaler": scaler,
        "source": "loaded from models/",
    }


df = load_data()
models = load_models(df)
frame = models["frame"]
scenarios_df = generate_day_scenarios()

# Observed auxiliary baseline, used when the user parks the van outside its
# usual overnight window (a region the model never saw during training).
parked_baseline = float(df.loc[df["speed_kmh"] <= 0.5, "energy_kwh"].median())

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("Energy Forecaster")
st.sidebar.caption("Assignment 12 — Energy Consumption Forecast")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Energy Predictor", "Forecast", "Model Comparison",
     "Trend & Seasonality", "Sensitivity Analysis", "What-If Scenarios"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Vehicle:** {KERB_MASS_KG} kg kerb + "
                    f"{PAYLOAD_MAX_KG} kg payload")
st.sidebar.markdown(f"**Models:** {models['source']}")
st.sidebar.markdown(f"**Dataset:** {len(df):,} hourly rows")
st.sidebar.markdown(f"**Days simulated:** {df['day'].nunique()}")
st.sidebar.markdown(f"**Best model MAE:** "
                    f"{models['comparison']['MAE (kWh)'].min():.2f} kWh")


# ── Page: Overview ──────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Energy Consumption Forecast")
    st.markdown("""
    **Predict how much energy a delivery van will consume, from the time of day,
    the speed it is driving at and the load it is carrying.**

    The dataset is a 120-day sequential simulation built from a road-load model,
    and three regression models are trained on Time, Speed and Load.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Hourly rows", f"{len(df):,}")
    col2.metric("Mean consumption", f"{df['energy_kwh'].mean():.2f} kWh/h")
    col3.metric("Peak hour", f"{df['energy_kwh'].max():.1f} kWh")
    col4.metric("Best model MAE", f"{models['comparison']['MAE (kWh)'].min():.2f} kWh")

    st.markdown("---")
    st.subheader("Sequential trend (first 14 days)")
    window = df.head(14 * 24)
    rolling = df["energy_kwh"].rolling(24).mean().head(14 * 24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["energy_kwh"],
                             name="Hourly consumption",
                             line=dict(color="#2c3e50", width=1.4)))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=rolling,
                             name="24-hour rolling mean",
                             line=dict(color="#e74c3c", width=3)))
    fig.update_layout(xaxis_title="Date", yaxis_title="Energy (kWh per hour)",
                      template="plotly_white", height=430)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Mean daily profile")
        hourly = df.groupby("hour")["energy_kwh"].mean().reset_index()
        fig = go.Figure(go.Scatter(x=hourly["hour"], y=hourly["energy_kwh"],
                                   mode="lines+markers",
                                   line=dict(color="#2c3e50", width=3),
                                   fill="tozeroy",
                                   fillcolor="rgba(44,62,80,0.12)"))
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Energy (kWh)",
                          template="plotly_white", height=360)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("Target distribution")
        fig = go.Figure(go.Histogram(x=df["energy_kwh"], nbinsx=55,
                                     marker_color="#27ae60"))
        fig.update_layout(xaxis_title="Energy (kWh per hour)",
                          yaxis_title="Hours", template="plotly_white", height=360)
        st.plotly_chart(fig, width="stretch")

    st.info("Hours at ~0.5 kWh are the overnight park: the van is stationary and "
            "only the auxiliary load (battery thermal management, telematics) runs.")


# ── Page: Energy Predictor ──────────────────────────────────────────────────
elif page == "Energy Predictor":
    st.title("Energy Predictor")
    st.markdown("Pick a time, a speed and a load — the model returns the expected "
                "energy consumption for that hour.")

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("Time of day", 0, 23, 12, format="%d:00")
        day_type = st.radio("Day type", ["Weekday", "Weekend"], horizontal=True)
        dow = 1 if day_type == "Weekday" else 5
    with col2:
        speed = st.slider("Speed (km/h)", 0, 100, 70, step=1)
        payload = st.slider("Load / payload (kg)", 0, PAYLOAD_MAX_KG, 800, step=25)

    # The van is loaded in the morning and emptied through the round, so at any
    # given hour the payload only spans a band. Outside it the model is
    # extrapolating, and the app says so instead of quietly guessing.
    band_rows = df[(df["hour"] == hour) & (df["speed_kmh"] > 0)]
    if len(band_rows):
        p_lo = float(band_rows["payload_kg"].quantile(0.02))
        p_hi = float(band_rows["payload_kg"].quantile(0.98))
        st.caption(f"Observed payload at {hour:02d}:00 is {p_lo:.0f}-{p_hi:.0f} kg. "
                   f"Outside that band the prediction is an extrapolation."
                   if speed > 0 else
                   f"The van is parked at {hour:02d}:00 in this schedule.")
        if speed > 0 and not (p_lo - 60 <= payload <= p_hi + 60):
            st.warning("This time/load combination never occurs in the simulated "
                       "duty cycle, so the model is extrapolating.")

    model_choice = st.radio("Model", ["Random Forest", "Polynomial Regression",
                                      "Linear Regression"], horizontal=True)
    if model_choice == "Random Forest":
        model, poly = models["rf"], None
    elif model_choice == "Polynomial Regression":
        model, poly = models["pr"], models["poly"]
    else:
        model, poly = models["lr"], None

    if speed == 0:
        st.info(f"At 0 km/h the van is parked. The model never saw a parked hour in "
                f"the middle of the day (it is always parked overnight), so the app "
                f"reports the observed auxiliary baseline instead: "
                f"{parked_baseline:.2f} kWh/h.")
        pred = parked_baseline
        distance = 0.0
        efficiency = float("nan")
    else:
        pred = max(predict_energy(model, hour, dow, speed, payload, poly=poly,
                                  scaler=models["scaler"]), 0.0)
        distance = float(speed)
        efficiency = pred / distance * 100 if distance > 0.5 else float("nan")

    status = status_for(efficiency, pred)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted energy", f"{pred:.2f} kWh")
    c2.metric("Distance this hour", f"{distance:.0f} km")
    c3.metric("Efficiency", "—" if np.isnan(efficiency) else f"{efficiency:.1f} kWh/100 km")
    c4.metric("Status", status)

    if status == "PARKED":
        st.success("Parked — auxiliary load only.")
    elif status == "EFFICIENT":
        st.success(f"Efficient: below the fleet's {EFFICIENCY_EFFICIENT_MAX} kWh/100 km "
                   f"75th percentile.")
    elif status == "HIGH DRAW":
        st.warning(f"High draw: between {EFFICIENCY_EFFICIENT_MAX} and "
                   f"{EFFICIENCY_HIGH_MAX} kWh/100 km.")
    else:
        st.error(f"Critical draw: above {EFFICIENCY_HIGH_MAX} kWh/100 km — the "
                 f"battery depletes fastest in this regime.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        title={"text": "Predicted energy (kWh)", "font": {"size": 20}},
        number={"suffix": " kWh"},
        gauge={
            "axis": {"range": [0, 45], "ticksuffix": " kWh"},
            "bar": {"color": "#2c3e50"},
            "steps": [
                {"range": [0, PARKED_ENERGY_MAX], "color": "#eef2f6"},
                {"range": [PARKED_ENERGY_MAX, 15], "color": "#d5f5e3"},
                {"range": [15, 30], "color": "#fdebd0"},
                {"range": [30, 45], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=330, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("How consumption changes with speed at this load")
    speeds = np.arange(0, 101, 5)
    curve = [parked_baseline if s == 0 else
             max(predict_energy(model, hour, dow, s, payload, poly=poly,
                                scaler=models["scaler"]), 0.0) for s in speeds]
    fig2 = go.Figure(go.Scatter(x=speeds, y=curve, mode="lines+markers",
                               line=dict(color="#e74c3c", width=3)))
    fig2.add_vline(x=speed, line_dash="dot", line_color="#2c3e50",
                   annotation_text=f"Current: {speed} km/h")
    fig2.update_layout(xaxis_title="Speed (km/h)", yaxis_title="Energy (kWh)",
                       template="plotly_white", height=360)
    st.plotly_chart(fig2, width="stretch")


# ── Page: Forecast ──────────────────────────────────────────────────────────
elif page == "Forecast":
    st.title("Forecast the Next Hours")
    st.markdown("""
    The forecast is **direct, not recursive**: every future hour is predicted from
    the planned duty cycle (the route schedule fixes when the van drives, how fast
    and how loaded). No actual consumption is used, so this is what a depot could
    really run before the shift starts.
    """)

    col1, col2 = st.columns(2)
    with col1:
        horizon = st.select_slider("Forecast horizon (hours)",
                                   options=[6, 12, 24, 48], value=24)
    with col2:
        model_choice = st.radio("Model", ["Random Forest", "Polynomial Regression",
                                          "Linear Regression"], horizontal=True)

    if model_choice == "Random Forest":
        model, poly = models["rf"], None
    elif model_choice == "Polynomial Regression":
        model, poly = models["pr"], models["poly"]
    else:
        model, poly = models["lr"], None

    start = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    forecast = forecast_horizon(model, start, steps=horizon, poly=poly,
                               scaler=models["scaler"])

    history = df.tail(48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["timestamp"], y=history["energy_kwh"],
                             name="Observed", line=dict(color="#2c3e50", width=2.4)))
    fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["forecast_kwh"],
                             name="Forecast", line=dict(color="#e74c3c", width=3,
                                                        dash="dash"),
                             mode="lines+markers"))
    fig.add_vline(x=start.timestamp() * 1000, line_dash="dot", line_color="gray",
                  annotation_text="forecast starts")
    fig.update_layout(xaxis_title="Date and time",
                      yaxis_title="Energy (kWh per hour)",
                      template="plotly_white", height=470)
    st.plotly_chart(fig, width="stretch")

    total = forecast["forecast_kwh"].sum()
    driving = forecast.loc[forecast["speed_kmh"] > 0, "forecast_kwh"].sum()
    distance = forecast["speed_kmh"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast total", f"{total:.1f} kWh")
    c2.metric("Driving share", f"{driving:.1f} kWh")
    c3.metric("Distance", f"{distance:.0f} km")
    c4.metric("Average intensity",
              "—" if distance == 0 else f"{total / distance * 100:.1f} kWh/100 km")

    st.markdown("---")
    st.subheader("Hour-by-hour forecast")
    st.dataframe(forecast[["timestamp", "hour", "day_type", "speed_kmh",
                           "payload_kg", "forecast_kwh"]],
                 width="stretch", hide_index=True)

    st.caption("Measured accuracy of this forecast method on the last 24 hours of "
               "the held-out test period is in models/forecast_24h.csv — the "
               "pipeline prints its MAE next to the model metrics.")


# ── Page: Model Comparison ──────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown("Metrics are computed on the chronologically held-out hours — the "
                "last 20% of the series, which the models never saw.")

    st.dataframe(models["comparison"], width="stretch", hide_index=True)

    comp = models["comparison"]
    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=("MAE (kWh)", "RMSE (kWh)", "MAPE (%)", "R²"))
    colors = ["#95a5a6", "#f39c12", "#27ae60"]
    for i, col in enumerate(["MAE (kWh)", "RMSE (kWh)", "MAPE (%)", "R²"]):
        fig.add_trace(go.Bar(x=comp["Model"], y=comp[col], marker_color=colors,
                             showlegend=False, text=comp[col].round(4),
                             textposition="outside"), row=1, col=i + 1)
    fig.update_layout(height=420, template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    st.info("The three models disagree by metric, and that is reported honestly: "
            "the degree-2 polynomial has the best R² and RMSE, while the random "
            "forest has the lowest MAE and by far the lowest MAPE (13% against "
            "23%). MAE is used to select the deployed model because it is in kWh, "
            "the same unit as the answer.")

    st.markdown("---")
    st.subheader("Actual vs Predicted — Random Forest")
    y_test = models["y_test"].values
    y_pred = models["rf_result"]["y_pred"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                              marker=dict(size=4, color="#e74c3c", opacity=0.45),
                              name="Predictions"))
    fig2.add_trace(go.Scatter(x=[0, 50], y=[0, 50], mode="lines",
                              line=dict(color="#2c3e50", dash="dash", width=2),
                              name="Perfect prediction"))
    fig2.update_layout(xaxis_title="Actual energy (kWh)", yaxis_title="Predicted (kWh)",
                       template="plotly_white", height=470)
    st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    st.subheader("Two experiments kept next to the model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Polynomial degree sweep** — why degree 2, and why not 4?")
        sweep_path = MODELS_DIR / "degree_sweep.csv"
        if sweep_path.exists():
            st.dataframe(pd.read_csv(sweep_path), width="stretch", hide_index=True)
        else:
            st.caption("Run src/run_pipeline.py to generate this table.")
    with col_b:
        st.markdown("**Sequential (lag) features** — measured, then deliberately "
                    "not deployed")
        seq_path = MODELS_DIR / "sequential_feature_experiment.csv"
        if seq_path.exists():
            st.dataframe(pd.read_csv(seq_path), width="stretch", hide_index=True)
        else:
            st.caption("Run src/run_pipeline.py to generate this table.")
        st.caption("Autoregressive lags improve accuracy, but at forecast time the "
                   "previous hours' consumption is unknown, so the deployed model "
                   "does not use them.")


# ── Page: Trend & Seasonality ───────────────────────────────────────────────
elif page == "Trend & Seasonality":
    st.title("Trend & Seasonality")

    tab1, tab2, tab3, tab4 = st.tabs(["Daily profile", "Working day vs weekend",
                                      "Efficiency by regime", "Long trend"])
    with tab1:
        hourly = df.groupby("hour").agg(mean_energy=("energy_kwh", "mean"),
                                        std_energy=("energy_kwh", "std"),
                                        mean_speed=("speed_kmh", "mean"),
                                        mean_payload=("payload_kg", "mean")
                                        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=hourly["hour"],
                                 y=hourly["mean_energy"] + hourly["std_energy"],
                                 line=dict(width=0), showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=hourly["hour"],
                                 y=hourly["mean_energy"] - hourly["std_energy"],
                                 fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(44,62,80,0.15)",
                                 name="±1 std dev"), secondary_y=False)
        fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["mean_energy"],
                                 line=dict(color="#2c3e50", width=3),
                                 name="Mean energy"), secondary_y=False)
        fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["mean_speed"],
                                 line=dict(color="#2980b9", width=2, dash="dash"),
                                 name="Mean speed (km/h)"), secondary_y=True)
        fig.add_trace(go.Scatter(x=hourly["hour"], y=hourly["mean_payload"],
                                 line=dict(color="#8e44ad", width=2, dash="dot"),
                                 name="Mean load (kg)"), secondary_y=True)
        fig.update_xaxes(title_text="Hour of day",
                         tickmode="array", tickvals=list(range(0, 24, 2)),
                         ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)])
        fig.update_yaxes(title_text="Energy (kWh)", secondary_y=False)
        fig.update_yaxes(title_text="Speed (km/h) / Load (kg)", secondary_y=True)
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        fig = go.Figure()
        for scenario, color in [("Working Day", "#2c3e50"), ("Weekend", "#2980b9")]:
            sub = scenarios_df[scenarios_df["scenario"] == scenario]
            fig.add_trace(go.Scatter(x=sub["hour"], y=sub["energy_kwh"],
                                     mode="lines+markers", name=scenario,
                                     line=dict(color=color, width=3)))
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Energy (kWh)",
                          template="plotly_white", height=480)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        moving = df[df["speed_kmh"] > 0.5]
        fig = go.Figure()
        for regime, color in [("urban", "#e74c3c"), ("suburban", "#f39c12"),
                              ("motorway", "#27ae60")]:
            sub = moving[moving["driving_regime"] == regime]
            fig.add_trace(go.Box(y=sub["efficiency_kwh_100km"], name=regime,
                                 marker_color=color, boxmean=True))
        fig.add_hline(y=EFFICIENCY_HIGH_MAX, line_dash="dash", line_color="red",
                      annotation_text="critical threshold")
        fig.update_layout(yaxis_title="kWh per 100 km", template="plotly_white",
                          height=480)
        st.plotly_chart(fig, width="stretch")

    with tab4:
        daily = df.groupby(df["timestamp"].dt.date)["energy_kwh"].sum().reset_index()
        daily.columns = ["date", "energy_kwh"]
        fig = go.Figure(go.Scatter(x=daily["date"], y=daily["energy_kwh"],
                                   mode="lines", line=dict(color="#2c3e50", width=2),
                                   fill="tozeroy",
                                   fillcolor="rgba(44,62,80,0.12)"))
        fig.update_layout(xaxis_title="Date", yaxis_title="Daily energy (kWh)",
                          template="plotly_white", height=480)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Mean daily energy: {daily['energy_kwh'].mean():.1f} kWh | "
                   f"weekday mean: "
                   f"{daily.loc[pd.to_datetime(daily['date']).dt.dayofweek < 5, 'energy_kwh'].mean():.1f} kWh")


# ── Page: Sensitivity Analysis ──────────────────────────────────────────────
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")
    st.markdown("One input is swept while the other is held fixed, so the effect of "
                "each input can be read on its own.")

    model = models["rf"]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Speed sensitivity")
        fixed_load = st.slider("Fixed load (kg)", 0, PAYLOAD_MAX_KG, 600, step=50)
        speeds = np.arange(0, 101, 2)
        fig = go.Figure()
        for day_label, dow in [("Midday (hour 12)", 2), ("Rush hour (hour 18)", 2)]:
            hour = 12 if "Midday" in day_label else 18
            curve = [parked_baseline if s == 0 else
                     max(predict_energy(model, hour, dow, s, fixed_load,
                                        scaler=models["scaler"]), 0.0)
                     for s in speeds]
            fig.add_trace(go.Scatter(x=speeds, y=curve, mode="lines", name=day_label))
        fig.update_layout(xaxis_title="Speed (km/h)", yaxis_title="Energy (kWh)",
                          template="plotly_white", height=430)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Load sensitivity")
        fixed_speed = st.slider("Fixed speed (km/h)", 0, 100, 80, step=5)
        loads = np.arange(0, PAYLOAD_MAX_KG + 1, 25)
        fig = go.Figure()
        for day_label, hour in [("Midday (hour 12)", 12), ("Rush hour (hour 18)", 18)]:
            curve = [max(predict_energy(model, hour, 2, fixed_speed, p,
                                        scaler=models["scaler"]), 0.0)
                     for p in loads]
            fig.add_trace(go.Scatter(x=loads, y=curve, mode="lines", name=day_label))
        fig.update_layout(xaxis_title="Load (kg)", yaxis_title="Energy (kWh)",
                          template="plotly_white", height=430)
        st.plotly_chart(fig, width="stretch")

    st.info("Below 30 km/h the drivetrain is penalised for stop-and-go losses and "
            "aerodynamic drag grows with the cube of speed above it, which is why "
            "the left-hand curve is not a straight line.")


# ── Page: What-If Scenarios ─────────────────────────────────────────────────
elif page == "What-If Scenarios":
    st.title("What-If Scenario Analysis")
    st.markdown("The same scenario set the report quotes, evaluated live from the "
                "saved random forest.")

    rows = build_whatif_rows(models["rf"], frame, scaler=models["scaler"])
    table = pd.DataFrame(rows, columns=["Scenario", "Time", "Description",
                                        "Speed (km/h)", "Load (kg)",
                                        "Energy (kWh)", "kWh/100 km", "Status"])
    st.dataframe(table, width="stretch", hide_index=True)

    fig = go.Figure(go.Bar(
        x=table["Scenario"], y=table["Energy (kWh)"],
        text=[f"{v} kWh" for v in table["Energy (kWh)"]],
        textposition="outside",
        marker_color=["#c0392b" if s == "CRITICAL DRAW" else
                      "#f39c12" if s == "HIGH DRAW" else
                      "#5b6b7d" if s == "PARKED" else "#27ae60"
                      for s in table["Status"]],
    ))
    fig.add_hline(y=15, line_dash="dash", line_color="#f39c12",
                  annotation_text="15 kWh/h — high draw")
    fig.add_hline(y=30, line_dash="dash", line_color="#c0392b",
                  annotation_text="30 kWh/h — critical draw")
    fig.update_layout(xaxis_title="Scenario", yaxis_title="Energy (kWh per hour)",
                      template="plotly_white", height=470)
    st.plotly_chart(fig, width="stretch")

    st.caption("Scenarios: " + " | ".join(
        f"{chr(ord('A') + i)} {desc}" for i, (_, desc, _, _) in
        enumerate(WHATIF_SCENARIOS)))

    st.markdown("---")
    st.subheader("What drives the answer")
    imp = models["rf"].feature_importances_
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "speed_kmh", "payload_kg"]
    order = np.argsort(imp)
    fig2 = go.Figure(go.Bar(x=[imp[i] for i in order],
                            y=[names[i] for i in order], orientation="h",
                            marker_color="#27ae60",
                            text=[f"{imp[i]:.3f}" for i in order],
                            textposition="outside"))
    fig2.update_layout(xaxis_title="Importance", template="plotly_white", height=400)
    st.plotly_chart(fig2, width="stretch")
    st.caption("Speed dominates because aerodynamic drag grows with the cube of "
               "speed; the clock adds little once speed and load are known, since "
               "the duty cycle is already encoded in them.")


st.sidebar.markdown("---")
st.sidebar.markdown("*Energy Consumption Forecast*")
st.sidebar.markdown("Time + Speed + Load → kWh")
