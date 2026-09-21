"""
Video Generation — Charging Station Load Simulation (v2)

Features:
    - AI voiceover narration (gTTS)
    - Subtitle overlays synced to narration
    - Better spacing (no overlapping text)
    - Full HD 1920x1080
    - ~58 seconds duration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import os

from data_generator import generate_dataset, generate_peak_offpeak_scenarios, STATION_CAPACITY_KW
from preprocessing import prepare_features, split_and_scale, add_time_features
from models import (
    train_linear_regression,
    train_polynomial_regression,
    train_random_forest,
    evaluate_model,
    compare_models,
    predict_load,
)

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#ffffff",
    "text.color": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
    "grid.color": "#0f3460",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
})

C = {
    "p": "#e94560", "s": "#0f3460", "a": "#533483",
    "ok": "#00b894", "w": "#fdcb6e", "bg": "#1a1a2e", "pn": "#16213e"
}

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
AUDIO_DIR = OUT / "audio"
TEMP_DIR = OUT / "temp_frames"
FINAL_VIDEO = OUT / "charging_station_simulation.mp4"

# ── Generate Data ───────────────────────────────────────────────────────────
print("[1/5] Preparing data...")
df = generate_dataset(num_days=90, seed=42)
feat = prepare_features(df)
splits = split_and_scale(feat["X"], feat["y"])
X_train, X_test, y_train, y_test = splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]

lr = train_linear_regression(X_train, y_train)
lr_r = evaluate_model(lr, X_test, y_test, "Linear")
pr, poly = train_polynomial_regression(X_train, y_train, degree=2)
pr_r = evaluate_model(pr, X_test, y_test, "Poly", poly=poly)
rf = train_random_forest(X_train, y_train, n_estimators=100)
rf_r = evaluate_model(rf, X_test, y_test, "RF")
comp = compare_models([lr_r, pr_r, rf_r])
df_sc = generate_peak_offpeak_scenarios()
df_f = add_time_features(df)

# ── Narration Script ────────────────────────────────────────────────────────
SECTIONS = [
    {
        "name": "title",
        "frames": 150,  # 6.25s at 24fps
        "narration": "Charging Station Load Prediction. An AI based approach to predict electrical demand using time of day and number of vehicles.",
        "subtitle": "CHARGING STATION LOAD PREDICTION\nAI/ML-Based Approach",
    },
    {
        "name": "data_sim",
        "frames": 240,  # 10s
        "narration": "We simulated 90 days of charging station data, generating 2 thousand 160 observations. The dataset captures realistic daily patterns including morning commute, lunch, and evening peak demand periods. Day types include weekdays, weekends, and low-demand holidays.",
        "subtitle": "90 Days x 24 Hours = 2,160 Observations\nRealistic peak/off-peak patterns with controlled randomness",
    },
    {
        "name": "load_curve",
        "frames": 240,  # 10s
        "narration": "The daily load curve reveals a characteristic double peak pattern. Morning demand rises around 7 AM, with the main evening peak occurring between 5 and 8 PM. The shaded region shows variability across all simulated days.",
        "subtitle": "Daily Load Curve\nDouble-peak pattern: Morning (7-11 AM) and Evening (5-9 PM)",
    },
    {
        "name": "peak_offpeak",
        "frames": 192,  # 8s
        "narration": "Two distinct demand patterns emerge. Peak days show significantly higher vehicle arrivals and load, while off-peak days like holidays demonstrate reduced demand. This contrast is central to the prediction task.",
        "subtitle": "Peak vs Off-Peak Day Comparison\nClear separation in demand patterns",
    },
    {
        "name": "models",
        "frames": 240,  # 10s
        "narration": "Three regression models were trained and compared. Linear regression serves as the baseline. Polynomial regression captures nonlinear relationships. Random Forest handles complex interactions between features.",
        "subtitle": "3 Regression Models: Linear, Polynomial, Random Forest\nTrain-Test Split: 80% Training, 20% Testing",
    },
    {
        "name": "results",
        "frames": 216,  # 9s
        "narration": "Random Forest achieved the best performance with an R squared of 0.9532 and mean absolute error of 7.75 kilowatts. The actual versus predicted scatter shows tight clustering along the ideal line.",
        "subtitle": "Best Model: Random Forest\nR² = 0.9532 | MAE = 7.75 kW | RMSE = 12.53 kW",
    },
    {
        "name": "summary",
        "frames": 180,  # 7.5s
        "narration": "In summary, time of day and vehicle count can effectively predict charging station load. The 3D surface visualizes this relationship. Future work should validate with real world charging station data.",
        "subtitle": "Conclusion\nTime + Vehicles --> Load | Future: Real-world validation",
    },
]

FPS = 24
TOTAL_FRAMES = sum(s["frames"] for s in SECTIONS)
DURATION = TOTAL_FRAMES / FPS

print(f"Total frames: {TOTAL_FRAMES} | Duration: {DURATION:.1f}s")

# ── Generate Audio ──────────────────────────────────────────────────────────
print("[2/5] Generating narration audio...")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

audio_files = []
for i, sec in enumerate(SECTIONS):
    audio_path = AUDIO_DIR / f"section_{i:02d}.mp3"
    if not audio_path.exists():
        tts = gTTS(text=sec["narration"], lang="en", slow=False)
        tts.save(str(audio_path))
    audio_files.append(audio_path)
    print(f"  [{i+1}/{len(SECTIONS)}] {sec['name']}: {sec['narration'][:50]}...")

# Combine audio with gaps
print("  Combining audio tracks...")
combined = AudioSegment.empty()
for af in audio_files:
    segment = AudioSegment.from_mp3(str(af))
    combined += segment
    combined += AudioSegment.silent(duration=300)  # 300ms gap between sections

# Pad to match video duration
audio_duration_ms = len(combined)
video_duration_ms = int(DURATION * 1000)
if audio_duration_ms < video_duration_ms:
    combined += AudioSegment.silent(duration=video_duration_ms - audio_duration_ms)
elif audio_duration_ms > video_duration_ms:
    combined = combined[:video_duration_ms]

final_audio = AUDIO_DIR / "narration.wav"
combined.export(str(final_audio), format="wav")
print(f"  Audio saved: {final_audio} ({len(combined)/1000:.1f}s)")

# ── Generate Video Frames ───────────────────────────────────────────────────
print("[3/5] Rendering frames...")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(19.2, 10.8), facecolor=C["bg"])  # 1920x1080

# Track subtitle text per frame
subtitle_map = {}

def add_subtitle_bar(ax, text, y_pos=0.02):
    """Add a semi-transparent subtitle bar at the bottom of an axes."""
    ax.text(0.5, y_pos, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=14, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
            multialignment="center")

def make_section_title(ax, title, subtitle=""):
    """Add a clean section title at the top."""
    ax.text(0.5, 1.08, title, transform=ax.transAxes, ha="center",
            fontsize=22, fontweight="bold", color=C["p"])
    if subtitle:
        ax.text(0.5, 1.03, subtitle, transform=ax.transAxes, ha="center",
                fontsize=13, color="#aaaaaa")

def style_ax(ax):
    ax.set_facecolor(C["pn"])
    for spine in ax.spines.values():
        spine.set_color(C["s"])
        spine.set_linewidth(0.5)

frame_idx = 0

def render_section(section_name, local_frame, total_frames):
    """Render a single frame for the given section."""

    if section_name == "title":
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["bg"]); ax.axis("off")

        a = min(local_frame / 40, 1.0)
        ax.text(0.5, 0.62, "CHARGING STATION", fontsize=56, fontweight="bold",
                color=C["p"], ha="center", alpha=a)
        ax.text(0.5, 0.48, "LOAD PREDICTION", fontsize=56, fontweight="bold",
                color=C["p"], ha="center", alpha=a)
        if local_frame > 40:
            a2 = min((local_frame - 40) / 40, 1.0)
            ax.text(0.5, 0.32, "AI/ML-Based Simulation & Prediction", fontsize=22,
                    color=C["a"], ha="center", alpha=a2)
            ax.text(0.5, 0.22, "Time of Day + Vehicle Count  -->  Load (kW)",
                    fontsize=18, color="#888888", ha="center", alpha=a2)

    elif section_name == "data_sim":
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        style_ax(ax1); style_ax(ax2)

        n = int(min(local_frame / total_frames * len(df), len(df)))
        sub = df.iloc[:n]

        make_section_title(ax1, "Simulating Charging Station Data")
        ax1.scatter(sub["hour"], sub["load_kw"], c=sub["vehicles"], cmap="plasma",
                   s=8, alpha=0.5, edgecolors="none")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Load (kW)")
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 4)])
        ax1.set_title(f"Observations: {n:,} / {len(df):,}", fontsize=14, color=C["w"])

        if n > 100:
            counts = sub["day_type"].value_counts()
            colors_pie = [C["p"], C["a"], C["ok"]]
            ax2.pie(counts.values, labels=counts.index, colors=colors_pie,
                   autopct="%1.0f%%", textprops={"color": "#fff", "fontsize": 13})
            ax2.set_title("Day Type Distribution", fontsize=16, fontweight="bold", color=C["p"])

        add_subtitle_bar(ax1, "90 Days x 24 Hours = 2,160 Observations")

    elif section_name == "load_curve":
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        style_ax(ax1); style_ax(ax2)

        hourly = df.groupby("hour").agg(m=("load_kw", "mean"), s=("load_kw", "std")).reset_index()
        n_h = int(min(local_frame / total_frames * 24, 24))
        h_sub = hourly.iloc[:n_h]

        make_section_title(ax1, "Daily Load Curve")
        if n_h > 0:
            ax1.fill_between(h_sub["hour"], h_sub["m"] - h_sub["s"], h_sub["m"] + h_sub["s"],
                            alpha=0.25, color=C["p"], label="+-1 Std Dev")
            ax1.plot(h_sub["hour"], h_sub["m"], color=C["p"], lw=3, marker="o", ms=7, label="Mean Load")
        ax1.axvspan(7, 11, alpha=0.08, color=C["w"], label="Morning Peak")
        ax1.axvspan(17, 21, alpha=0.08, color=C["w"], label="Evening Peak")
        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Load (kW)")
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
        ax1.legend(loc="upper left", fontsize=11, facecolor=C["pn"], edgecolor=C["s"])
        ax1.set_ylim(0, 300)

        # Vehicles overlay
        ax1b = ax1.twinx()
        ax1b.plot(hourly["hour"], hourly["m"] / 5, color=C["a"], ls="--", lw=1.5, alpha=0.6, label="Vehicles (scaled)")
        ax1b.set_ylabel("Vehicles (scaled)", color=C["a"])

        make_section_title(ax2, "Vehicles vs Load Relationship")
        ax2.scatter(df["vehicles"], df["load_kw"], s=6, alpha=0.2, color=C["p"])
        ax2.set_xlabel("Number of Vehicles")
        ax2.set_ylabel("Load (kW)")
        ax2.set_title("Clear positive correlation", fontsize=14, color=C["ok"])

        add_subtitle_bar(ax1, "Double-peak pattern: Morning (7-11 AM) and Evening (5-9 PM)")

    elif section_name == "peak_offpeak":
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        style_ax(ax1); style_ax(ax2)

        make_section_title(ax1, "Peak vs Off-Peak Comparison")
        progress = min(local_frame / total_frames, 1.0)
        n = int(progress * 24)

        for sc, co, ls in [("Peak Day", C["p"], "-"), ("Off-Peak Day", C["a"], "--")]:
            s = df_sc[df_sc["scenario"] == sc].iloc[:n]
            if len(s) > 0:
                ax1.plot(s["hour"], s["load_kw"], color=co, lw=3, ls=ls,
                        marker="o", ms=5, label=sc)

        ax1.set_xlabel("Hour of Day")
        ax1.set_ylabel("Load (kW)")
        ax1.set_xticks(range(0, 24, 4))
        ax1.legend(fontsize=13, facecolor=C["pn"], edgecolor=C["s"])
        ax1.set_ylim(0, 320)
        ax1.axhline(y=STATION_CAPACITY_KW, color=C["w"], ls=":", lw=2, label="Capacity")

        # By day type
        make_section_title(ax2, "Load by Day Type")
        for dt, co in [("weekday", C["p"]), ("weekend", C["a"]), ("holiday", C["ok"])]:
            s = df_f[df_f["day_type"] == dt].groupby("hour")["load_kw"].mean()
            ax2.plot(s.index, s.values, color=co, lw=2.5, label=dt)
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("Mean Load (kW)")
        ax2.legend(fontsize=13, facecolor=C["pn"], edgecolor=C["s"])

        add_subtitle_bar(ax1, "Two distinct demand patterns: Peak Day vs Off-Peak Day")

    elif section_name == "models":
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        for a in [ax1, ax2, ax3, ax4]: style_ax(a)

        make_section_title(ax1, "Model Comparison (R Score)", "Higher is better")
        names = ["Linear", "Poly", "RF"]
        r2s = [lr_r["r2"], pr_r["r2"], rf_r["r2"]]
        colors = [C["s"], C["a"], C["ok"]]
        x = np.arange(3)
        bars = ax1.bar(x, r2s, 0.6, color=colors, edgecolor="white", lw=0.5)
        ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=13)
        ax1.set_ylabel("R Score")
        ax1.set_ylim(0, 1.05)
        for i, v in enumerate(r2s):
            ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold")

        make_section_title(ax2, "Mean Absolute Error", "Lower is better")
        maes = [lr_r["mae"], pr_r["mae"], rf_r["mae"]]
        ax2.bar(x, maes, 0.6, color=colors, edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=13)
        ax2.set_ylabel("MAE (kW)")
        for i, v in enumerate(maes):
            ax2.text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")

        make_section_title(ax3, "Actual vs Predicted (Random Forest)")
        ax3.scatter(y_test, rf_r["y_pred"], s=12, alpha=0.4, color=C["p"], edgecolors="none")
        ax3.plot([0, 300], [0, 300], "--", color="#ffffff", lw=2)
        ax3.set_xlabel("Actual Load (kW)")
        ax3.set_ylabel("Predicted Load (kW)")
        ax3.set_xlim(0, 300); ax3.set_ylim(0, 300)
        ax3.set_aspect("equal")

        make_section_title(ax4, "Feature Importance (Random Forest)")
        imps = rf.feature_importances_
        ax4.barh(feat["feature_names"], imps, color=C["ok"], height=0.5)
        ax4.set_xlabel("Importance")
        for i, v in enumerate(imps):
            ax4.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=12, fontweight="bold")

        add_subtitle_bar(ax1, "3 Models Compared: Linear, Polynomial, Random Forest")

    elif section_name == "results":
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        style_ax(ax1); style_ax(ax2)

        make_section_title(ax1, "Best Model: Random Forest")

        # Results table
        ax1.axis("off")
        table_data = [
            ["Metric", "Value"],
            ["R Score", f"{rf_r['r2']:.4f}"],
            ["MAE", f"{rf_r['mae']:.2f} kW"],
            ["RMSE", f"{rf_r['rmse']:.2f} kW"],
            ["Train Samples", f"{X_train.shape[0]:,}"],
            ["Test Samples", f"{X_test.shape[0]:,}"],
        ]
        table = ax1.table(cellText=table_data[1:], colLabels=table_data[0],
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.3, 2.0)
        for j in range(2):
            table[0, j].set_facecolor(C["s"])
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(table_data)):
            table[i, 0].set_text_props(fontweight="bold")
        ax1.set_title("Performance Metrics", fontsize=18, fontweight="bold", color=C["p"], pad=20)

        make_section_title(ax2, "Actual vs Predicted Load")
        ax2.scatter(y_test, rf_r["y_pred"], s=15, alpha=0.5, color=C["p"], edgecolors="none")
        ax2.plot([0, 300], [0, 300], "--", color="#ffffff", lw=2, label="Perfect Prediction")
        ax2.set_xlabel("Actual Load (kW)", fontsize=14)
        ax2.set_ylabel("Predicted Load (kW)", fontsize=14)
        ax2.set_xlim(0, 300); ax2.set_ylim(0, 300)
        ax2.legend(fontsize=12, facecolor=C["pn"], edgecolor=C["s"])

        # R annotation
        ax2.text(0.05, 0.92, f"R = {rf_r['r2']:.4f}", transform=ax2.transAxes,
                fontsize=16, fontweight="bold", color=C["ok"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["pn"], edgecolor=C["ok"]))

        add_subtitle_bar(ax1, "Random Forest: R = 0.9532 | MAE = 7.75 kW")

    elif section_name == "summary":
        ax = fig.add_subplot(111)
        ax.set_facecolor(C["pn"]); ax.axis("off")

        make_section_title(ax, "Project Summary")

        findings = [
            ("PROBLEM", "Predict EV charging station load using time of day and vehicle count"),
            ("DATA", "90-day simulation with 2,160 observations (peak/off-peak patterns)"),
            ("APPROACH", "3 regression models: Linear, Polynomial, Random Forest"),
            ("BEST MODEL", f"Random Forest (R = {rf_r['r2']:.4f}, MAE = {rf_r['mae']:.2f} kW)"),
            ("KEY FINDING", "Load is nonlinear -- Random Forest captures interactions better"),
            ("FUTURE WORK", "Validate with real-world charging station data"),
        ]

        y = 0.85
        for i, (label, value) in enumerate(findings):
            a = min(max((local_frame - i * 12) / 20, 0), 1.0)
            if a > 0:
                ax.text(0.08, y, f"{label}:", fontsize=16, fontweight="bold",
                       color=C["p"], transform=ax.transAxes, alpha=a)
                ax.text(0.25, y, value, fontsize=15, color="#ffffff",
                       transform=ax.transAxes, alpha=a)
                y -= 0.13

        add_subtitle_bar(ax, "Time + Vehicles --> Load | Future: Real-world validation")

def animate(frame):
    fig.clear()

    # Find which section this frame belongs to
    cumsum = 0
    for sec in SECTIONS:
        if frame < cumsum + sec["frames"]:
            local_frame = frame - cumsum
            render_section(sec["name"], local_frame, sec["frames"])

            # Draw subtitle bar at the very bottom of the figure
            subtitle_text = sec["subtitle"].split("\n")[0]  # First line only
            fig.text(0.5, 0.01, subtitle_text, ha="center", va="bottom",
                    fontsize=15, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.75))
            break
        cumsum += sec["frames"]

    return []

# ── Render ──────────────────────────────────────────────────────────────────
print(f"[4/5] Rendering {TOTAL_FRAMES} frames...")
anim = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000 // FPS, blit=False)

# Save frames to temp directory first
print("  Saving frames as PNG sequence...")
anim.save(str(TEMP_DIR / "frame_%05d.png"), writer="pillow", dpi=100)
plt.close()

print("[5/5] Combining video with audio...")

# Use ffmpeg to combine image sequence + audio
cmd = [
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(TEMP_DIR / "frame_%05d.png"),
    "-i", str(final_audio),
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac",
    "-b:a", "192k",
    "-shortest",
    "-movflags", "+faststart",
    str(FINAL_VIDEO),
]

result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print(f"  ffmpeg error: {result.stderr[-500:]}")
else:
    size_mb = FINAL_VIDEO.stat().st_size / 1024 / 1024
    print(f"\n  Video saved: {FINAL_VIDEO}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Duration: {DURATION:.1f} seconds")

# Clean up temp frames
print("  Cleaning up temp frames...")
for f in TEMP_DIR.glob("*.png"):
    f.unlink()
TEMP_DIR.rmdir()

print("\nDone!")
