"""
Fast Video Generation with Audio — Charging Station Load Simulation (v3)

Uses matplotlib's FFMpegWriter for direct rendering (much faster).
Audio generated via gTTS and merged with ffmpeg.
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
from gtts import gTTS
from pydub import AudioSegment
import subprocess
import tempfile

from data_generator import generate_dataset, generate_peak_offpeak_scenarios, STATION_CAPACITY_KW
from preprocessing import prepare_features, split_and_scale, add_time_features
from models import (
    train_linear_regression, train_polynomial_regression, train_random_forest,
    evaluate_model, compare_models, predict_load,
)

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e", "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560", "axes.labelcolor": "#ffffff",
    "text.color": "#ffffff", "xtick.color": "#ffffff", "ytick.color": "#ffffff",
    "grid.color": "#0f3460", "grid.alpha": 0.3, "font.size": 13,
    "axes.titlesize": 18, "axes.labelsize": 15,
})

C = {"p":"#e94560","s":"#0f3460","a":"#533483","ok":"#00b894","w":"#fdcb6e","bg":"#1a1a2e","pn":"#16213e"}
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# ── Data ────────────────────────────────────────────────────────────────────
print("[1/4] Preparing data...")
df = generate_dataset(num_days=90, seed=42)
feat = prepare_features(df)
splits = split_and_scale(feat["X"], feat["y"])
X_tr, X_te, y_tr, y_te = splits["X_train"], splits["X_test"], splits["y_train"], splits["y_test"]

lr = train_linear_regression(X_tr, y_tr); lr_r = evaluate_model(lr, X_te, y_te, "LR")
pr, poly = train_polynomial_regression(X_tr, y_tr, 2); pr_r = evaluate_model(pr, X_te, y_te, "PR", poly=poly)
rf = train_random_forest(X_tr, y_tr, 100); rf_r = evaluate_model(rf, X_te, y_te, "RF")
df_sc = generate_peak_offpeak_scenarios()
df_f = add_time_features(df)

# ── Audio Generation ────────────────────────────────────────────────────────
print("[2/4] Generating narration...")
NARRATIONS = [
    "Title. Charging Station Load Prediction. An AI based approach to predict electrical demand using time of day and number of vehicles.",
    "Data Simulation. We simulated 90 days of charging station data, generating 2 thousand 160 observations. The dataset captures realistic daily patterns including morning commute, lunch, and evening peak demand.",
    "Load Curve. The daily load curve reveals a characteristic double peak pattern. Morning demand rises around 7 AM, with the main evening peak between 5 and 8 PM.",
    "Peak vs Off Peak. Two distinct demand patterns emerge. Peak days show significantly higher vehicle arrivals and load, while off peak days demonstrate reduced demand.",
    "Models. Three regression models were trained: Linear regression as baseline, Polynomial regression for nonlinear relationships, and Random Forest for complex feature interactions.",
    "Results. Random Forest achieved the best performance with R squared of 0.9532 and mean absolute error of 7.75 kilowatts.",
    "Summary. Time of day and vehicle count can effectively predict charging station load. Future work should validate with real world charging station data.",
]

SUBTITLES = [
    "CHARGING STATION LOAD PREDICTION | AI/ML-Based Approach",
    "90 Days x 24 Hours = 2,160 Observations | Peak/Off-Peak Patterns",
    "Daily Load Curve | Morning (7-11 AM) and Evening (5-9 PM) Peaks",
    "Peak Day vs Off-Peak Day | Clear Demand Separation",
    "3 Models: Linear, Polynomial, Random Forest | 80/20 Train-Test Split",
    "Best: Random Forest | R=0.9532 | MAE=7.75 kW | RMSE=12.53 kW",
    "Conclusion | Future: Real-World Validation",
]

DURATIONS = [5, 10, 10, 8, 10, 9, 7]  # seconds per section

audio_dir = OUT / "audio"
audio_dir.mkdir(parents=True, exist_ok=True)

audio_parts = []
for i, (text, dur) in enumerate(zip(NARRATIONS, DURATIONS)):
    mp3_path = audio_dir / f"narr_{i:02d}.mp3"
    if not mp3_path.exists():
        gTTS(text=text, lang="en", slow=False).save(str(mp3_path))
    part = AudioSegment.from_mp3(str(mp3_path))
    # Pad or trim to exact duration
    target_ms = dur * 1000
    if len(part) < target_ms:
        part += AudioSegment.silent(duration=target_ms - len(part))
    else:
        part = part[:target_ms]
    audio_parts.append(part)
    print(f"  [{i+1}/7] {dur}s - {text[:40]}...")

# Concatenate with 200ms gaps
combined = AudioSegment.empty()
for p in audio_parts:
    combined += p
    combined += AudioSegment.silent(duration=200)

final_audio_path = audio_dir / "narration.wav"
combined.export(str(final_audio_path), format="wav")
print(f"  Audio: {final_audio_path} ({len(combined)/1000:.1f}s)")

# ── Video Rendering ─────────────────────────────────────────────────────────
print("[3/4] Rendering video...")

FPS = 24
TOTAL_FRAMES = sum(d * FPS for d in DURATIONS)
SECTION_STARTS = []
cumul = 0
for d in DURATIONS:
    SECTION_STARTS.append(cumul)
    cumul += d * FPS

fig = plt.figure(figsize=(16, 9), facecolor=C["bg"])

def get_section(frame):
    for i in range(len(DURATIONS) - 1, -1, -1):
        if frame >= SECTION_STARTS[i]:
            return i, frame - SECTION_STARTS[i]
    return 0, frame

def style(ax):
    ax.set_facecolor(C["pn"])
    for s in ax.spines.values(): s.set_color(C["s"]); s.set_linewidth(0.5)

def title(ax, t, sub=""):
    ax.text(0.5, 1.10, t, transform=ax.transAxes, ha="center", fontsize=22,
            fontweight="bold", color=C["p"])
    if sub: ax.text(0.5, 1.04, sub, transform=ax.transAxes, ha="center",
                    fontsize=13, color="#aaa")

def sub_bar(fig, text):
    fig.text(0.5, 0.01, text, ha="center", va="bottom", fontsize=14, color="white",
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.8))

def animate(frame):
    fig.clear()
    sec, lf = get_section(frame)
    dur_frames = DURATIONS[sec] * FPS

    if sec == 0:  # TITLE
        ax = fig.add_subplot(111); ax.set_facecolor(C["bg"]); ax.axis("off")
        a = min(lf / 30, 1.0)
        ax.text(0.5, 0.62, "CHARGING STATION", fontsize=52, fontweight="bold",
                color=C["p"], ha="center", alpha=a)
        ax.text(0.5, 0.48, "LOAD PREDICTION", fontsize=52, fontweight="bold",
                color=C["p"], ha="center", alpha=a)
        if lf > 30:
            a2 = min((lf-30)/30, 1.0)
            ax.text(0.5, 0.32, "AI/ML-Based Simulation & Prediction", fontsize=20,
                    color=C["a"], ha="center", alpha=a2)
            ax.text(0.5, 0.22, "Time of Day + Vehicle Count  -->  Load (kW)",
                    fontsize=16, color="#888", ha="center", alpha=a2)

    elif sec == 1:  # DATA SIM
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        style(ax1); style(ax2)
        n = int(lf / dur_frames * len(df))
        sub = df.iloc[:n]
        title(ax1, "Simulating Charging Station Data")
        ax1.scatter(sub["hour"], sub["load_kw"], c=sub["vehicles"], cmap="plasma",
                   s=8, alpha=0.5, edgecolors="none")
        ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Load (kW)")
        ax1.set_xticks(range(0, 24, 4))
        ax1.set_title(f"Observations: {n:,} / {len(df):,}", fontsize=14, color=C["w"])
        if n > 100:
            ct = sub["day_type"].value_counts()
            ax2.pie(ct.values, labels=ct.index, colors=[C["p"],C["a"],C["ok"]],
                   autopct="%1.0f%%", textprops={"color":"#fff","fontsize":13})
            ax2.set_title("Day Type Distribution", fontsize=16, fontweight="bold", color=C["p"])

    elif sec == 2:  # LOAD CURVE
        ax1, ax2 = fig.add_subplot(211), fig.add_subplot(212)
        style(ax1); style(ax2)
        hourly = df.groupby("hour").agg(m=("load_kw","mean"),s=("load_kw","std")).reset_index()
        n_h = int(lf / dur_frames * 24)
        h_sub = hourly.iloc[:n_h]
        title(ax1, "Daily Load Curve")
        if n_h > 0:
            ax1.fill_between(h_sub["hour"], h_sub["m"]-h_sub["s"], h_sub["m"]+h_sub["s"],
                            alpha=0.25, color=C["p"])
            ax1.plot(h_sub["hour"], h_sub["m"], color=C["p"], lw=3, marker="o", ms=7)
        ax1.axvspan(7, 11, alpha=0.08, color=C["w"])
        ax1.axvspan(17, 21, alpha=0.08, color=C["w"])
        ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Load (kW)")
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
        ax1.set_ylim(0, 300)
        title(ax2, "Vehicles vs Load")
        ax2.scatter(df["vehicles"], df["load_kw"], s=6, alpha=0.2, color=C["p"])
        ax2.set_xlabel("Number of Vehicles"); ax2.set_ylabel("Load (kW)")

    elif sec == 3:  # PEAK/OFF-PEAK
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        style(ax1); style(ax2)
        title(ax1, "Peak vs Off-Peak Comparison")
        progress = min(lf / dur_frames, 1.0)
        n = int(progress * 24)
        for sc, co in [("Peak Day", C["p"]), ("Off-Peak Day", C["a"])]:
            s = df_sc[df_sc["scenario"]==sc].iloc[:n]
            if len(s)>0: ax1.plot(s["hour"], s["load_kw"], color=co, lw=3, marker="o", ms=5, label=sc)
        ax1.legend(fontsize=13, facecolor=C["pn"], edgecolor=C["s"])
        ax1.set_xlabel("Hour"); ax1.set_ylabel("Load (kW)"); ax1.set_ylim(0, 320)
        title(ax2, "Load by Day Type")
        for dt, co in [("weekday",C["p"]),("weekend",C["a"]),("holiday",C["ok"])]:
            s = df_f[df_f["day_type"]==dt].groupby("hour")["load_kw"].mean()
            ax2.plot(s.index, s.values, color=co, lw=2.5, label=dt)
        ax2.legend(fontsize=13, facecolor=C["pn"], edgecolor=C["s"])
        ax2.set_xlabel("Hour"); ax2.set_ylabel("Load (kW)")

    elif sec == 4:  # MODELS
        axes = [fig.add_subplot(221+i) for i in range(4)]
        for a in axes: style(a)
        names = ["Linear","Poly","RF"]
        r2s = [lr_r["r2"],pr_r["r2"],rf_r["r2"]]
        maes = [lr_r["mae"],pr_r["mae"],rf_r["mae"]]
        cols = [C["s"],C["a"],C["ok"]]
        x = np.arange(3)
        title(axes[0], "R Score (Higher = Better)")
        axes[0].bar(x, r2s, 0.6, color=cols, edgecolor="white")
        axes[0].set_xticks(x); axes[0].set_xticklabels(names)
        axes[0].set_ylim(0, 1.05)
        for i,v in enumerate(r2s): axes[0].text(i, v+0.02, f"{v:.4f}", ha="center", fontweight="bold")
        title(axes[1], "MAE in kW (Lower = Better)")
        axes[1].bar(x, maes, 0.6, color=cols, edgecolor="white")
        axes[1].set_xticks(x); axes[1].set_xticklabels(names)
        for i,v in enumerate(maes): axes[1].text(i, v+0.2, f"{v:.2f}", ha="center", fontweight="bold")
        title(axes[2], "Actual vs Predicted (RF)")
        axes[2].scatter(y_te, rf_r["y_pred"], s=12, alpha=0.4, color=C["p"], edgecolors="none")
        axes[2].plot([0,300],[0,300],"--",color="white",lw=2)
        axes[2].set_xlabel("Actual"); axes[2].set_ylabel("Predicted")
        axes[2].set_xlim(0,300); axes[2].set_ylim(0,300)
        title(axes[3], "Feature Importance")
        axes[3].barh(feat["feature_names"], rf.feature_importances_, color=C["ok"], height=0.5)
        axes[3].set_xlabel("Importance")

    elif sec == 5:  # RESULTS
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
        style(ax1); style(ax2)
        title(ax1, "Random Forest Performance")
        ax1.axis("off")
        rows = [["R Score",f"{rf_r['r2']:.4f}"],["MAE",f"{rf_r['mae']:.2f} kW"],
                ["RMSE",f"{rf_r['rmse']:.2f} kW"],["Train",f"{X_tr.shape[0]:,}"],["Test",f"{X_te.shape[0]:,}"]]
        tbl = ax1.table(cellText=rows, colLabels=["Metric","Value"], loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(14); tbl.scale(1.3, 2.0)
        for j in range(2):
            tbl[0,j].set_facecolor(C["s"]); tbl[0,j].set_text_props(color="white",fontweight="bold")
        title(ax2, "Actual vs Predicted")
        ax2.scatter(y_te, rf_r["y_pred"], s=15, alpha=0.5, color=C["p"], edgecolors="none")
        ax2.plot([0,300],[0,300],"--",color="white",lw=2)
        ax2.set_xlabel("Actual (kW)"); ax2.set_ylabel("Predicted (kW)")
        ax2.set_xlim(0,300); ax2.set_ylim(0,300)
        ax2.text(0.05, 0.92, f"R={rf_r['r2']:.4f}", transform=ax2.transAxes, fontsize=16,
                fontweight="bold", color=C["ok"], bbox=dict(boxstyle="round",fc=C["pn"],ec=C["ok"]))

    elif sec == 6:  # SUMMARY
        ax = fig.add_subplot(111); ax.set_facecolor(C["pn"]); ax.axis("off")
        title(ax, "Project Summary")
        items = [
            ("PROBLEM","Predict EV station load using time of day and vehicle count"),
            ("DATA","90-day simulation, 2,160 observations, peak/off-peak patterns"),
            ("APPROACH","3 regression models compared on train/test split"),
            ("BEST MODEL",f"Random Forest (R={rf_r['r2']:.4f}, MAE={rf_r['mae']:.2f} kW)"),
            ("FINDING","Nonlinear relationship captured better by Random Forest"),
            ("FUTURE","Validate with real-world charging station data"),
        ]
        y = 0.85
        for i,(lab,val) in enumerate(items):
            a = min(max((lf - i*10)/15, 0), 1.0)
            if a > 0:
                ax.text(0.08, y, f"{lab}:", fontsize=16, fontweight="bold", color=C["p"],
                       transform=ax.transAxes, alpha=a)
                ax.text(0.25, y, val, fontsize=15, color="white", transform=ax.transAxes, alpha=a)
                y -= 0.13

    # Subtitle at bottom
    sub_bar(fig, SUBTITLES[sec])
    return []

anim = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES, interval=1000//FPS, blit=False)

temp_video = OUT / "temp_video.mp4"
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000,
                                 extra_args=["-vcodec","libx264","-pix_fmt","yuv420p"])
anim.save(str(temp_video), writer=writer, dpi=100)
plt.close()
print(f"  Temp video: {temp_video}")

# ── Merge Audio + Video ─────────────────────────────────────────────────────
print("[4/4] Merging audio and video...")
final = OUT / "charging_station_simulation.mp4"

cmd = [
    "ffmpeg", "-y",
    "-i", str(temp_video),
    "-i", str(final_audio_path),
    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
    "-shortest", "-movflags", "+faststart",
    str(final),
]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(f"  Error: {r.stderr[-300:]}")
else:
    size = final.stat().st_size / 1024 / 1024
    print(f"\n  Final video: {final}")
    print(f"  Size: {size:.1f} MB")
    print(f"  Duration: {sum(DURATIONS)}s")

# Cleanup
temp_video.unlink(missing_ok=True)
print("\nDone!")
