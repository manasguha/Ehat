"""
Interactive Dashboard — Charging Station Load Prediction

Streamlit app with:
    - Interactive load predictor (time + vehicles → load)
    - Peak/off-peak load curves
    - Model comparison
    - Sensitivity analysis
    - What-if scenarios
    - Peak-load warning

Models are loaded from the pickles that src/run_pipeline.py writes into
models/ (see load_models below), so the app does not retrain on startup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import (
    generate_dataset,
    generate_peak_offpeak_scenarios,
    STATION_CAPACITY_KW,
)
from preprocessing import prepare_features, split_and_scale, add_time_features
from models import (
    train_linear_regression,
    train_polynomial_regression,
    train_random_forest,
    evaluate_model,
    compare_models,
    predict_load,
    load_model,
)

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "charging_station_data.csv"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Charging Station Load Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cache Data & Models ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load the simulated dataset (regenerating it only if the CSV is absent)."""
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)
    return generate_dataset(num_days=90, seed=42)


def train_models(df):
    """Fallback: train all three models from scratch."""
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])

    X_train, X_test = splits["X_train"], splits["X_test"]
    y_train, y_test = splits["y_train"], splits["y_test"]

    # Linear
    lr = train_linear_regression(X_train, y_train)
    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")

    # Polynomial
    pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)

    # Random Forest
    rf = train_random_forest(X_train, y_train, n_estimators=100)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")

    comparison = compare_models([lr_result, pr_result, rf_result])

    return {
        "lr": lr, "pr": pr, "poly": poly, "rf": rf,
        "lr_result": lr_result, "pr_result": pr_result, "rf_result": rf_result,
        "comparison": comparison,
        "X_test": X_test, "y_test": y_test,
        "feat": feat,
        "scaler": splits["scaler"],
        "source": "trained in-app (artifacts missing)",
    }


@st.cache_resource
def load_models(df):
    """
    Load the models saved by src/run_pipeline.py and evaluate them on the
    held-out test split.

    The dataset and the train/test split are both seeded, so re-running the
    split reproduces the exact test set used at training time. That lets us
    rebuild the comparison table by predicting only -- no retraining needed
    when the app starts.
    """
    feat = prepare_features(df)
    splits = split_and_scale(feat["X"], feat["y"])
    X_test, y_test = splits["X_test"], splits["y_test"]

    try:
        lr = load_model(MODELS_DIR / "linear_regression.pkl")
        saved_poly = load_model(MODELS_DIR / "polynomial_regression.pkl")
        if isinstance(saved_poly, dict):
            pr, poly = saved_poly["model"], saved_poly["poly"]
        else:
            pr, poly = saved_poly
        rf = load_model(MODELS_DIR / "random_forest.pkl")
        scaler = load_model(MODELS_DIR / "scaler.pkl")
    except Exception as exc:  # missing/incompatible artifacts -> train instead
        st.warning(
            f"Could not load saved models ({exc}). Training from scratch -- "
            "this takes a few seconds."
        )
        return train_models(df)

    lr_result = evaluate_model(lr, X_test, y_test, "Linear Regression")
    pr_result = evaluate_model(pr, X_test, y_test, "Polynomial Regression", poly=poly)
    rf_result = evaluate_model(rf, X_test, y_test, "Random Forest")
    comparison = compare_models([lr_result, pr_result, rf_result])

    return {
        "lr": lr, "pr": pr, "poly": poly, "rf": rf,
        "lr_result": lr_result, "pr_result": pr_result, "rf_result": rf_result,
        "comparison": comparison,
        "X_test": X_test, "y_test": y_test,
        "feat": feat,
        "scaler": scaler,
        "source": "loaded from models/",
    }


# ── Load Everything ─────────────────────────────────────────────────────────
df = load_data()
models = load_models(df)
df_scenarios = generate_peak_offpeak_scenarios()
df_with_features = add_time_features(df)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("EV Load Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Interactive Predictor", "Model Comparison",
     "Load Curves", "Sensitivity Analysis", "What-If Scenarios"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Station Capacity:** {STATION_CAPACITY_KW} kW")
st.sidebar.markdown(f"**Models:** {models['source']}")
st.sidebar.markdown(f"**Dataset:** {len(df)} observations")
st.sidebar.markdown(f"**Days simulated:** {df['day'].nunique()}")


# ── Page: Overview ──────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Charging Station Load Prediction")
    st.markdown("""
    **Predict the electrical load of a charging station based on time of day and number of vehicles.**

    This project uses simulated peak/off-peak data to train regression models and generate load curves.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", f"{len(df):,}")
    col2.metric("Station Capacity", f"{STATION_CAPACITY_KW} kW")
    col3.metric("Best Model R²", f"{models['comparison']['R²'].max():.4f}")

    st.markdown("---")
    st.subheader("Daily Load Pattern")

    hourly = df.groupby("hour").agg(
        mean_load=("load_kw", "mean"),
        std_load=("load_kw", "std"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean_load"] + hourly["std_load"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean_load"] - hourly["std_load"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(44,62,80,0.15)", name="±1 Std Dev",
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["mean_load"],
        mode="lines+markers", line=dict(color="#2c3e50", width=3),
        marker=dict(size=6), name="Mean Load",
    ))

    # Peak zones
    fig.add_vrect(x0=7, x1=11, fillcolor="red", opacity=0.06, line_width=0, annotation_text="Morning Peak")
    fig.add_vrect(x0=17, x1=21, fillcolor="red", opacity=0.06, line_width=0, annotation_text="Evening Peak")

    fig.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Load (kW)",
        xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 2)),
                    ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig,     width="stretch")


# ── Page: Interactive Predictor ─────────────────────────────────────────────
elif page == "Interactive Predictor":
    st.title("Interactive Load Predictor")
    st.markdown("Enter time and vehicle count to predict station load.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time of Day")
        selected_hour = st.slider(
            "Hour", min_value=0, max_value=23, value=18, step=1,
            format="%d:00",
        )
        st.caption(f"Selected: **{selected_hour:02d}:00**")

        hour_sin = np.sin(2 * np.pi * selected_hour / 24)
        hour_cos = np.cos(2 * np.pi * selected_hour / 24)

        # Period label
        if 5 <= selected_hour < 12:
            period = "Morning"
        elif 12 <= selected_hour < 17:
            period = "Afternoon"
        elif 17 <= selected_hour < 21:
            period = "Evening"
        else:
            period = "Night"
        st.info(f"Period: {period}")

    with col2:
        st.subheader("Number of Vehicles")
        num_vehicles = st.number_input(
            "Vehicles at station", min_value=0, max_value=60, value=35, step=1,
        )

    st.markdown("---")

    # Model selection
    model_choice = st.radio(
        "Select Model",
        ["Random Forest", "Linear Regression", "Polynomial Regression"],
        horizontal=True,
    )

    if model_choice == "Random Forest":
        model = models["rf"]
        poly = None
    elif model_choice == "Linear Regression":
        model = models["lr"]
        poly = None
    else:
        model = models["pr"]
        poly = models["poly"]

    # Predict
    pred_load = predict_load(model, hour_sin, hour_cos, num_vehicles, poly=poly, scaler=models["scaler"])
    pred_load = max(0, min(pred_load, STATION_CAPACITY_KW))

    # Display result
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Predicted Load", f"{pred_load:.1f} kW")

    with col2:
        utilization = (pred_load / STATION_CAPACITY_KW) * 100
        st.metric("Station Utilization", f"{utilization:.1f}%")

    with col3:
        if pred_load > 270:
            st.error("PEAK WARNING")
        elif pred_load > 200:
            st.warning("High Load")
        else:
            st.success("Normal")

    # Load gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_load,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Predicted Load (kW)", "font": {"size": 20}},
        number={"suffix": " kW"},
        gauge={
            "axis": {"range": [0, STATION_CAPACITY_KW], "ticksuffix": " kW"},
            "bar": {"color": "#2c3e50"},
            "steps": [
                {"range": [0, 200], "color": "#d5f5e3"},
                {"range": [200, 270], "color": "#fdebd0"},
                {"range": [270, STATION_CAPACITY_KW], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 270,
            },
        },
    ))
    fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig,     width="stretch")

    # Sensitivity preview
    st.markdown("---")
    st.subheader("How Load Changes with Vehicles (at this hour)")

    veh_range = np.arange(0, 61, 5)
    loads = [predict_load(model, hour_sin, hour_cos, v, poly, scaler=models["scaler"]) for v in veh_range]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=veh_range, y=loads,
        mode="lines+markers",
        line=dict(color="#e74c3c", width=3),
        marker=dict(size=8),
        name="Predicted Load",
    ))
    fig2.add_hline(y=STATION_CAPACITY_KW, line_dash="dash", line_color="red",
                    annotation_text="Station Capacity")
    fig2.add_vline(x=num_vehicles, line_dash="dot", line_color="#2c3e50",
                    annotation_text=f"Current: {num_vehicles} vehicles")

    fig2.update_layout(
        xaxis_title="Number of Vehicles", yaxis_title="Load (kW)",
        template="plotly_white", height=350,
    )
    st.plotly_chart(fig2,     width="stretch")


# ── Page: Model Comparison ──────────────────────────────────────────────────
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.dataframe(models["comparison"],     width="stretch", hide_index=True)

    # Bar chart
    comp = models["comparison"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE (kW)", "RMSE (kW)", "R²"))

    colors = ["#95a5a6", "#f39c12", "#27ae60"]
    for i, col in enumerate(["MAE (kW)", "RMSE (kW)", "R²"]):
        fig.add_trace(go.Bar(
            x=comp["Model"], y=comp[col],
            marker_color=colors, name=col, showlegend=False,
            text=comp[col].round(4), textposition="outside",
        ), row=1, col=i + 1)

    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig,     width="stretch")

    # Actual vs Predicted for best model
    st.subheader("Actual vs Predicted — Random Forest")
    y_test = models["y_test"].values
    y_pred = models["rf_result"]["y_pred"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=y_test, y=y_pred, mode="markers",
        marker=dict(size=4, color="#e74c3c", opacity=0.5),
        name="Predictions",
    ))
    fig2.add_trace(go.Scatter(
        x=[0, 300], y=[0, 300], mode="lines",
        line=dict(color="#2c3e50", dash="dash", width=2),
        name="Perfect Prediction",
    ))
    fig2.update_layout(
        xaxis_title="Actual Load (kW)", yaxis_title="Predicted Load (kW)",
        template="plotly_white", height=500,
    )
    st.plotly_chart(fig2,     width="stretch")


# ── Page: Load Curves ──────────────────────────────────────────────────────
elif page == "Load Curves":
    st.title("Load Curves")

    tab1, tab2, tab3 = st.tabs(["Daily Pattern", "Peak vs Off-Peak", "By Day Type"])

    with tab1:
        hourly = df.groupby("hour").agg(
            mean_load=("load_kw", "mean"),
            std_load=("load_kw", "std"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["mean_load"] + hourly["std_load"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["mean_load"] - hourly["std_load"],
            fill="tonexty", fillcolor="rgba(44,62,80,0.15)", mode="lines",
            line=dict(width=0), name="±1 Std Dev",
        ))
        fig.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly["mean_load"],
            mode="lines+markers", line=dict(color="#2c3e50", width=3),
            name="Mean Load",
        ))
        fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Load (kW)",
            xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig,     width="stretch")

    with tab2:
        fig = go.Figure()
        for scenario, color in [("Peak Day", "#e74c3c"), ("Off-Peak Day", "#3498db")]:
            subset = df_scenarios[df_scenarios["scenario"] == scenario]
            fig.add_trace(go.Scatter(
                x=subset["hour"], y=subset["load_kw"],
                mode="lines+markers", line=dict(color=color, width=3),
                name=scenario,
            ))
        fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Load (kW)",
            xaxis=dict(tickmode="array", tickvals=list(range(0, 24, 2)),
                        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig,     width="stretch")

    with tab3:
        fig = go.Figure()
        for dt, color in [("weekday", "#e74c3c"), ("weekend", "#3498db"), ("holiday", "#8e44ad")]:
            subset = df_with_features[df_with_features["day_type"] == dt]
            hourly_dt = subset.groupby("hour")["load_kw"].mean().reset_index()
            fig.add_trace(go.Scatter(
                x=hourly_dt["hour"], y=hourly_dt["load_kw"],
                mode="lines", line=dict(color=color, width=2.5),
                name=dt.capitalize(),
            ))
        fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Mean Load (kW)",
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig,     width="stretch")


# ── Page: Sensitivity Analysis ─────────────────────────────────────────────
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")

    st.markdown("How each input independently affects predicted load.")

    model = models["rf"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vehicle Count Sensitivity")
        fixed_hour = st.slider("Fix time at hour:", 0, 23, 18, key="sens_hour")

        veh_range = np.arange(0, 61, 2)
        loads = []
        for v in veh_range:
            hs = np.sin(2 * np.pi * fixed_hour / 24)
            hc = np.cos(2 * np.pi * fixed_hour / 24)
            loads.append(predict_load(model, hs, hc, v, scaler=models["scaler"]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=veh_range, y=loads, mode="lines",
            line=dict(color="#e74c3c", width=3), fill="tozeroy",
            fillcolor="rgba(231,76,60,0.1)",
        ))
        fig.add_hline(y=STATION_CAPACITY_KW, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Vehicles", yaxis_title="Load (kW)",
            title=f"At {fixed_hour:02d}:00", template="plotly_white", height=400,
        )
        st.plotly_chart(fig,     width="stretch")

    with col2:
        st.subheader("Time-of-Day Sensitivity")
        fixed_vehicles = st.number_input("Fix vehicles at:", 0, 60, 40, key="sens_veh")

        hour_range = np.linspace(0, 23, 96)
        loads = []
        for h in hour_range:
            hs = np.sin(2 * np.pi * h / 24)
            hc = np.cos(2 * np.pi * h / 24)
            loads.append(predict_load(model, hs, hc, fixed_vehicles, scaler=models["scaler"]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hour_range, y=loads, mode="lines",
            line=dict(color="#3498db", width=3), fill="tozeroy",
            fillcolor="rgba(52,152,219,0.1)",
        ))
        fig.add_hline(y=STATION_CAPACITY_KW, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Hour of Day", yaxis_title="Load (kW)",
            title=f"With {fixed_vehicles} vehicles", template="plotly_white", height=400,
        )
        st.plotly_chart(fig,     width="stretch")


# ── Page: What-If Scenarios ────────────────────────────────────────────────
elif page == "What-If Scenarios":
    st.title("What-If Scenario Analysis")

    st.markdown("Compare predicted load across different scenarios.")

    scenarios = [
        ("A", "08:00", 20, "Morning commute"),
        ("B", "18:00", 20, "Evening, few vehicles"),
        ("C", "18:00", 40, "Evening, many vehicles"),
        ("D", "20:00", 60, "Peak demand"),
        ("E", "03:00", 5, "Late night"),
        ("F", "12:00", 35, "Midday"),
    ]

    model = models["rf"]
    rows = []
    for label, time_str, veh, desc in scenarios:
        h = int(time_str.split(":")[0])
        hs = np.sin(2 * np.pi * h / 24)
        hc = np.cos(2 * np.pi * h / 24)
        pred = predict_load(model, hs, hc, veh, scaler=models["scaler"])
        utilization = (pred / STATION_CAPACITY_KW) * 100
        status = "PEAK" if pred > 270 else ("HIGH" if pred > 200 else "NORMAL")
        rows.append({
            "Scenario": label,
            "Time": time_str,
            "Vehicles": veh,
            "Description": desc,
            "Predicted Load (kW)": round(pred, 1),
            "Utilization": f"{utilization:.1f}%",
            "Status": status,
        })

    st.dataframe(pd.DataFrame(rows),     width="stretch", hide_index=True)

    # Comparison bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r["Scenario"] for r in rows],
        y=[r["Predicted Load (kW)"] for r in rows],
        text=[f"{r['Predicted Load (kW)']} kW" for r in rows],
        textposition="outside",
        marker_color=["#e74c3c" if r["Predicted Load (kW)"] > 270 else
                       "#f39c12" if r["Predicted Load (kW)"] > 200 else "#27ae60"
                       for r in rows],
    ))
    fig.add_hline(y=STATION_CAPACITY_KW, line_dash="dash", line_color="red",
                   annotation_text=f"Capacity: {STATION_CAPACITY_KW} kW")
    fig.update_layout(
        xaxis_title="Scenario", yaxis_title="Predicted Load (kW)",
        template="plotly_white", height=450,
    )
    st.plotly_chart(fig,     width="stretch")


# ── Footer ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("*Charging Station Load Prediction*")
st.sidebar.markdown("ML-Based Approach: Time + Vehicles → Load")
