# Charging Station Load Prediction

**AI/ML-Based Charging Station Load Prediction Using Time of Day and Vehicle Count**

## Problem Statement

Predict the electrical load of an EV charging station based on:
- **Time of Day**
- **Number of Vehicles**

## Objective

Simulate realistic peak/off-peak charging data, train regression models, and generate load curves to predict station demand.

## Project Structure

```
charging-station-load-prediction/
│
├── data/
│   └── charging_station_data.csv      # Simulated dataset (2160 rows)
│
├── notebooks/
│   ├── 01_data_simulation_eda.ipynb   # Data generation & exploration
│   ├── 02_regression_models.ipynb     # Model training & comparison
│   └── 03_analysis.ipynb              # Sensitivity, 3D, what-if
│
├── src/
│   ├── data_generator.py              # Realistic data simulation
│   ├── preprocessing.py               # Feature engineering (cyclic time)
│   ├── models.py                      # Linear, Polynomial, Random Forest
│   ├── visualization.py               # All figures
│   ├── run_pipeline.py                # End-to-end pipeline
│   ├── test_inference.py              # Inference correctness tests
│   ├── narration_script.py            # Narration text (7 sections)
│   ├── generate_narration.py          # Kokoro TTS audio generator
│   ├── generate_subtitles.py          # SRT subtitle generator
│   └── generate_video_final.py        # Video builder (1920x1080)
│
├── dashboard/
│   └── app.py                         # Interactive Streamlit predictor
│
├── models/                            # Saved trained models + scaler
│
├── outputs/figures/                   # Generated visualizations
│
├── dist/
│   ├── charging_station_simulation_final.mp4   # Final video with narration
│   ├── narration.wav                            # Full narration audio
│   ├── narration_*.wav                          # Per-section audio
│   └── subtitles.srt                            # SRT subtitles
│
├── requirements.txt
├── requirements-video.txt              # figures / narration / video extras
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

That covers the dashboard and the ML pipeline. Only if you want the figures,
narration and video do you also need the heavier extras:

```bash
pip install -r requirements-video.txt
```

### 2. Run the full pipeline

```bash
python src/run_pipeline.py
```

This will:
- Generate 90 days of simulated charging data (2160 observations)
- Train 3 regression models
- Save the StandardScaler alongside models
- Evaluate and compare models
- Generate all visualizations (corrected for scaled inference)
- Save models, scaler, and results

### 3. Run inference tests

```bash
python src/test_inference.py
```

Verifies that:
- All models load and predict correctly with the scaler
- Changing inputs produces changing outputs (not flat predictions)
- Predictions are within plausible ranges

### 4. Generate narration and video

```bash
# Generate narration audio (requires kokoro package)
python src/generate_narration.py

# Generate subtitles
python src/generate_subtitles.py

# Build final video
python src/generate_video_final.py
```

### 5. Launch interactive dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard loads `models/*.pkl` straight from disk (it only trains from
scratch if those files are missing), so it starts without re-running training
and always reflects the artifacts you last saved.

## Methodology

### Data Simulation

- **90 days** × **24 hours** = **2160 observations**
- Realistic vehicle arrival patterns (morning commute, lunch, evening peak)
- Day types: weekday (55%), weekend (30%), holiday (15%)
- Controlled noise: ε ~ N(0, 0.12 × base_load)
- Station capacity: 300 kW

### Feature Engineering

- **Cyclic time encoding**: sin(2πt/24), cos(2πt/24)
  - Preserves temporal proximity (23:00 close to 00:00)
- **StandardScaler**: Zero mean, unit variance on all features
- Peak/off-peak binary flags
- Time period categories

### Regression Models

| Model | MAE (kW) | RMSE (kW) | R² |
|-------|----------|-----------|-----|
| Linear Regression | 8.98 | 13.95 | 0.9419 |
| Polynomial Regression | 8.72 | 12.78 | 0.9513 |
| **Random Forest** | **7.75** | **12.53** | **0.9532** |

### Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination — explained variance on the evaluated dataset)

## Key Visualizations

1. **Daily Load Curve** — Characteristic double-peak pattern
2. **Peak vs Off-Peak** — Two 24-hour profiles compared
3. **Actual vs Predicted** — Scatter plot for each model
4. **3D Load Surface** — Time × Vehicles → Load
5. **Sensitivity Analysis** — How each input affects load independently
6. **What-If Scenarios** — Scenario comparison table
7. **Feature Importance** — Random Forest feature contributions

## Demo Video

The `dist/charging_station_simulation_final.mp4` contains:
- 1920×1080 resolution, 24fps
- 7-section academic presentation with narration
- Dark navy + electric-blue design theme
- Kokoro-82M narration (Apache 2.0 license)
- SRT subtitles included

## Deploying the Live Dashboard

The dashboard is deployed to Streamlit Community Cloud so it can be opened from
a phone -- e.g. by scanning a QR code printed on the presentation slide.

### Repository layout the deployment expects

```
charging-station-load-prediction/
|
+-- dashboard/app.py            <- the entrypoint file
+-- requirements.txt            <- dashboard dependencies
+-- src/                        <- data generation, preprocessing, models
+-- data/
|   +-- charging_station_data.csv
+-- models/                     <- MUST be committed: app.py loads these
    +-- linear_regression.pkl
    +-- polynomial_regression.pkl
    +-- random_forest.pkl
    +-- scaler.pkl
```

### Steps

1. Push this folder to a GitHub repository (keep the `.pkl` files committed --
   the app loads them at startup instead of retraining).
2. Go to `share.streamlit.io`, sign in with GitHub, and create a new app.
3. Select the repository, branch `main`, and entrypoint file
   `dashboard/app.py`, then pick a memorable subdomain, e.g.
   `charging-station-load-prediction.streamlit.app`.
4. Open the URL on your phone **and on a different network** before the
   presentation -- the free tier sleeps after inactivity, so the very first
   load of the day can take ~30 s while the app wakes up.
5. Generate a QR code for the deployed URL and put both the QR and the plain
   URL on the final slide.

Because the app tracks the GitHub repo, pushing a change updates the deployed
dashboard without changing the QR code.

## Interactive Dashboard Features

- **Interactive Predictor** — Enter time + vehicles, get predicted load
- **Model Comparison** — Side-by-side metrics
- **Load Curves** — Daily, peak/off-peak, by day type
- **Sensitivity Analysis** — Fix one input, vary the other
- **What-If Scenarios** — Compare pre-defined scenarios
- **Peak Load Warning** — Station capacity monitoring

## Tech Stack

- **Python** — Core language
- **NumPy + Pandas** — Data generation and processing
- **Scikit-learn** — Regression models, StandardScaler
- **Matplotlib / Seaborn** — Static visualizations (figures only, see `requirements-video.txt`)
- **Plotly** — Interactive charts
- **Streamlit** — Dashboard + cloud deployment
- **Kokoro-82M** — Text-to-speech narration (Apache 2.0)
- **ffmpeg** — Video encoding

## Input → Output

```
Time of Day + Number of Vehicles → ML Regression → Load (kW)
```

## Narration TTS

This project uses [Kokoro-82M](https://github.com/hexgrad/kokoro) for narration:
- **License**: Apache 2.0
- **Voice**: af_heart (American English female)
- **Model**: 82M parameters, auto-downloaded from HuggingFace
- **Sample rate**: 24kHz
- No paid API keys required

## License

Academic project — for educational purposes.
