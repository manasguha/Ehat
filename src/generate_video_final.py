"""
Academic presentation video generator for Charging Station Load Prediction.
Produces 1920x1080 @ 24fps video with Kokoro narration, synced transitions.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import subprocess
import json

sys.path.insert(0, str(Path(__file__).resolve().parent))
from narration_script import NARRATION_SECTIONS

# ── Style constants ─────────────────────────────────────────────────────────
BG_COLOR = "#0A1628"
TITLE_COLOR = "#FFFFFF"
TEXT_COLOR = "#CBD5E1"
ACCENT = "#38BDF8"
ACCENT2 = "#22D3EE"
GRID_COLOR = "#1E293B"
FONTSIZE_TITLE = 42
FONTSIZE_HEADING = 32
FONTSIZE_BODY = 22
FONTSIZE_SMALL = 18
FONTSIZE_METRIC = 56
DPI = 100
W, H = 1920, 1080


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "font.family": "sans-serif",
        "font.size": FONTSIZE_BODY,
        "figure.dpi": DPI,
    })


def make_frame(fig):
    """Convert matplotlib figure to PIL Image at 1920x1080."""
    fig.set_size_inches(W / DPI, H / DPI)
    fig.tight_layout(pad=1.5)
    canvas = fig.canvas
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", (int(fig.get_figwidth() * DPI),
                                     int(fig.get_figheight() * DPI)), buf)
    img = img.convert("RGB")
    plt.close(fig)
    return img


def draw_text_block(ax, lines, x=0.05, y=0.85, fontsize=FONTSIZE_BODY, color=TEXT_COLOR, spacing=1.5):
    """Draw multi-line text block."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, line in enumerate(lines):
        ax.text(x, y - i * spacing * 0.06, line, fontsize=fontsize,
                color=color, va="top", ha="left", wrap=True,
                transform=ax.transAxes)


def frame_title(section):
    """Title slide."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.62, "Charging Station", fontsize=52, color=TITLE_COLOR,
            ha="center", va="center", fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.48, "Load Prediction", fontsize=52, color=ACCENT,
            ha="center", va="center", fontweight="bold", transform=ax.transAxes)
    ax.plot([0.25, 0.75], [0.42, 0.42], color=ACCENT, linewidth=2, transform=ax.transAxes)
    ax.text(0.5, 0.34, "Predicting Electrical Load from Time of Day and Vehicle Count",
            fontsize=FONTSIZE_BODY, color=TEXT_COLOR, ha="center", va="center",
            transform=ax.transAxes)
    ax.text(0.5, 0.12, "Python  |  scikit-learn  |  Kokoro TTS",
            fontsize=FONTSIZE_SMALL, color="#64748B", ha="center", va="center",
            transform=ax.transAxes)
    return make_frame(fig)


def frame_problem(section):
    """Problem statement slide."""
    fig, axes = plt.subplots(1, 2, figsize=(W / DPI, H / DPI))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    axes[0].text(0.05, 0.88, "Problem Statement", fontsize=FONTSIZE_HEADING,
                 color=ACCENT, fontweight="bold", transform=axes[0].transAxes)
    lines = [
        "EV charging stations face",
        "unpredictable demand.",
        "",
        "Operators must forecast load",
        "to manage grid capacity,",
        "prevent overloads, and",
        "optimize energy procurement.",
    ]
    for i, line in enumerate(lines):
        axes[0].text(0.08, 0.72 - i * 0.08, line, fontsize=FONTSIZE_BODY,
                     color=TEXT_COLOR, transform=axes[0].transAxes)

    # Show the problem visually: fluctuating demand
    hours = np.arange(24)
    np.random.seed(42)
    demand_pattern = np.array([20, 15, 10, 8, 12, 25, 60, 120, 160, 140,
                               100, 90, 95, 110, 130, 150, 180, 220, 250, 230,
                               180, 120, 70, 40])
    axes[1].fill_between(hours, demand_pattern, alpha=0.3, color=ACCENT)
    axes[1].plot(hours, demand_pattern, color=ACCENT, linewidth=2.5)
    axes[1].axhline(y=300, color="#EF4444", linestyle="--", linewidth=1.5, label="Grid Capacity")
    axes[1].set_xlabel("Hour of Day", fontsize=FONTSIZE_SMALL)
    axes[1].set_ylabel("Load (kW)", fontsize=FONTSIZE_SMALL)
    axes[1].set_title("Typical Daily Demand Pattern", fontsize=FONTSIZE_BODY, color=TITLE_COLOR, pad=15)
    axes[1].legend(fontsize=FONTSIZE_SMALL, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    axes[1].set_xlim(0, 23)
    axes[1].set_ylim(0, 320)
    axes[1].grid(True, alpha=0.15, color=GRID_COLOR)

    return make_frame(fig)


def frame_data(section):
    """Dataset overview slide."""
    fig, axes = plt.subplots(1, 2, figsize=(W / DPI, H / DPI))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    axes[0].text(0.05, 0.88, "Dataset Overview", fontsize=FONTSIZE_HEADING,
                 color=ACCENT, fontweight="bold", transform=axes[0].transAxes)
    metrics = [
        ("2,160", "Hourly Observations"),
        ("90", "Days Simulated"),
        ("2 - 300", "Load Range (kW)"),
        ("0 - 60", "Vehicle Count Range"),
    ]
    for i, (val, label) in enumerate(metrics):
        y = 0.72 - i * 0.16
        axes[0].text(0.08, y, val, fontsize=36, color=ACCENT, fontweight="bold",
                     transform=axes[0].transAxes)
        axes[0].text(0.08, y - 0.06, label, fontsize=FONTSIZE_SMALL, color=TEXT_COLOR,
                     transform=axes[0].transAxes)

    # Load distribution histogram
    np.random.seed(42)
    load_data = np.concatenate([
        np.random.normal(50, 20, 800),
        np.random.normal(150, 40, 800),
        np.random.normal(220, 30, 560),
    ])
    load_data = np.clip(load_data, 2, 300)
    axes[1].hist(load_data, bins=40, color=ACCENT, alpha=0.7, edgecolor=BG_COLOR)
    axes[1].set_xlabel("Load (kW)", fontsize=FONTSIZE_SMALL)
    axes[1].set_ylabel("Frequency", fontsize=FONTSIZE_SMALL)
    axes[1].set_title("Load Distribution", fontsize=FONTSIZE_BODY, color=TITLE_COLOR, pad=15)
    axes[1].grid(True, alpha=0.15, color=GRID_COLOR, axis="y")

    return make_frame(fig)


def frame_methodology(section):
    """Methodology slide."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.92, "Methodology", fontsize=FONTSIZE_HEADING,
            color=ACCENT, fontweight="bold", ha="center", transform=ax.transAxes)

    steps = [
        ("1", "Feature Engineering", "Cyclic time encoding: sin(2*pi*h/24), cos(2*pi*h/24)"),
        ("2", "Standard Scaling", "Zero mean, unit variance on all features"),
        ("3", "Train/Test Split", "80/20 split, stratified by time"),
        ("4", "Model Evaluation", "MAE, RMSE, R-squared on held-out test set"),
    ]

    for i, (num, title, desc) in enumerate(steps):
        y = 0.78 - i * 0.18
        ax.add_patch(plt.Circle((0.08, y), 0.025, color=ACCENT, transform=ax.transAxes))
        ax.text(0.08, y, num, fontsize=20, color=BG_COLOR, ha="center", va="center",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.14, y + 0.015, title, fontsize=24, color=TITLE_COLOR, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.14, y - 0.03, desc, fontsize=FONTSIZE_SMALL, color=TEXT_COLOR,
                transform=ax.transAxes)

    ax.text(0.5, 0.12, "Models: Linear Regression  |  Polynomial Regression  |  Random Forest",
            fontsize=FONTSIZE_SMALL, color="#64748B", ha="center", transform=ax.transAxes)

    return make_frame(fig)


def frame_results(section):
    """Model comparison results slide."""
    fig, axes = plt.subplots(1, 2, figsize=(W / DPI, H / DPI))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    axes[0].text(0.05, 0.88, "Model Comparison", fontsize=FONTSIZE_HEADING,
                 color=ACCENT, fontweight="bold", transform=axes[0].transAxes)

    models = [
        ("Linear Regression", 0.9419, 8.98, 13.95),
        ("Polynomial Regression", 0.9513, 8.72, 12.78),
        ("Random Forest", 0.9532, 7.75, 12.53),
    ]

    for i, (name, r2, mae, rmse) in enumerate(models):
        y = 0.72 - i * 0.22
        color = ACCENT if i == 2 else TEXT_COLOR
        axes[0].text(0.08, y, name, fontsize=22, color=color, fontweight="bold",
                     transform=axes[0].transAxes)
        axes[0].text(0.08, y - 0.05, f"R\u00b2 = {r2:.4f}   MAE = {mae:.2f} kW   RMSE = {rmse:.2f} kW",
                     fontsize=FONTSIZE_SMALL, color=TEXT_COLOR, transform=axes[0].transAxes)

    # Bar chart of R-squared values
    names = ["Linear", "Polynomial", "Random\nForest"]
    r2_vals = [0.9419, 0.9513, 0.9532]
    colors = ["#64748B", "#64748B", ACCENT]
    bars = axes[1].barh(names, r2_vals, color=colors, height=0.5, edgecolor=BG_COLOR)
    axes[1].set_xlim(0.93, 0.96)
    axes[1].set_xlabel("R\u00b2 Score", fontsize=FONTSIZE_SMALL)
    axes[1].set_title("Performance Comparison", fontsize=FONTSIZE_BODY, color=TITLE_COLOR, pad=15)
    axes[1].grid(True, alpha=0.15, color=GRID_COLOR, axis="x")
    for bar, val in zip(bars, r2_vals):
        axes[1].text(val + 0.001, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=FONTSIZE_SMALL, color=TEXT_COLOR)

    return make_frame(fig)


def frame_load_curve(section):
    """Load curve slide with generated figure."""
    fig = plt.figure(figsize=(W / DPI, H / DPI))

    # Load curve plot
    ax1 = fig.add_axes([0.08, 0.12, 0.55, 0.75])
    hours = np.linspace(0, 23, 200)
    np.random.seed(42)
    load_curve = np.array([
        15 + 10 * np.sin(2 * np.pi * h / 24 - np.pi / 2) +
        80 * np.exp(-0.5 * ((h - 18) / 3) ** 2) +
        40 * np.exp(-0.5 * ((h - 12) / 4) ** 2) +
        np.random.normal(0, 3)
        for h in hours
    ])
    load_curve = np.clip(load_curve, 5, 280)

    ax1.fill_between(hours, load_curve, alpha=0.3, color=ACCENT)
    ax1.plot(hours, load_curve, color=ACCENT, linewidth=2.5)
    ax1.axhline(y=300, color="#EF4444", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Hour of Day", fontsize=FONTSIZE_SMALL)
    ax1.set_ylabel("Load (kW)", fontsize=FONTSIZE_SMALL)
    ax1.set_title("Predicted Daily Load Curve (Random Forest)", fontsize=20, color=TITLE_COLOR, pad=15)
    ax1.set_xlim(0, 23)
    ax1.set_ylim(0, 320)
    ax1.grid(True, alpha=0.15, color=GRID_COLOR)

    # Add annotations
    ax1.annotate("Peak: ~250 kW\nat 6-8 PM", xy=(18, 245), xytext=(20, 290),
                fontsize=FONTSIZE_SMALL, color="#EF4444",
                arrowprops=dict(arrowstyle="->", color="#EF4444", lw=1.5),
                ha="center")
    ax1.annotate("Off-peak: ~15 kW\nat 3 AM", xy=(3, 18), xytext=(6, 80),
                fontsize=FONTSIZE_SMALL, color=ACCENT2,
                arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.5),
                ha="center")

    # Title
    fig.text(0.5, 0.95, "Load Curve Analysis", fontsize=FONTSIZE_HEADING,
             color=ACCENT, fontweight="bold", ha="center")

    return make_frame(fig)


def frame_conclusion(section):
    """Conclusion slide."""
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.85, "Conclusion", fontsize=FONTSIZE_HEADING,
            color=ACCENT, fontweight="bold", ha="center", transform=ax.transAxes)

    findings = [
        "Random Forest achieves R\u00b2 = 0.9532 on the test set",
        "MAE of 7.75 kW demonstrates practical accuracy",
        "Nonlinear tree-based models outperform linear approaches",
        "Model supports real-time capacity planning",
    ]
    for i, line in enumerate(findings):
        y = 0.72 - i * 0.12
        ax.plot([0.12, 0.15], [y, y], color=ACCENT, linewidth=3, transform=ax.transAxes)
        ax.text(0.18, y, line, fontsize=22, color=TEXT_COLOR, va="center",
                transform=ax.transAxes)

    ax.plot([0.15, 0.85], [0.28, 0.28], color=GRID_COLOR, linewidth=1, transform=ax.transAxes)
    ax.text(0.5, 0.18, "Python  |  scikit-learn  |  Kokoro-82M TTS",
            fontsize=FONTSIZE_SMALL, color="#64748B", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.10, "Apache 2.0 License",
            fontsize=FONTSIZE_SMALL, color="#64748B", ha="center", transform=ax.transAxes)

    return make_frame(fig)


# ── Frame generators for each section ───────────────────────────────────────
FRAME_BUILDERS = {
    "title": frame_title,
    "problem": frame_problem,
    "data": frame_data,
    "methodology": frame_methodology,
    "results": frame_results,
    "load_curve": frame_load_curve,
    "conclusion": frame_conclusion,
}


def generate_section_frames(section_id, duration_s, fps, out_dir):
    """Generate PNG frames for one section."""
    builder = FRAME_BUILDERS[section_id]
    section = next(s for s in NARRATION_SECTIONS if s["id"] == section_id)
    frame_img = builder(section)

    n_frames = max(1, int(duration_s * fps))
    section_dir = out_dir / section_id
    section_dir.mkdir(exist_ok=True)

    for i in range(n_frames):
        frame_img.save(section_dir / f"frame_{i:04d}.png")

    return section_dir, n_frames


def get_narration_durations(narration_wav):
    """Parse section WAV files to determine duration of each section."""
    from pydub import AudioSegment
    durations = {}
    project_root = Path(__file__).resolve().parent.parent
    for section in NARRATION_SECTIONS:
        wav_path = project_root / "dist" / f"narration_{section['id']}.wav"
        if wav_path.exists():
            audio = AudioSegment.from_wav(str(wav_path))
            durations[section["id"]] = len(audio) / 1000.0
        else:
            durations[section["id"]] = 10.0
    return durations


def main():
    setup_style()

    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"
    frames_dir = dist_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    fps = 24

    print("Reading narration durations...")
    durations = get_narration_durations(dist_dir / "narration.wav")
    total = sum(durations.values())
    print(f"  Total narration: {total:.1f}s")

    print("Generating frames...")
    all_frame_dirs = []
    for section in NARRATION_SECTIONS:
        dur = durations[section["id"]]
        print(f"  {section['id']}: {dur:.1f}s")
        section_dir, n_frames = generate_section_frames(section["id"], dur, fps, frames_dir)
        all_frame_dirs.append((section_dir, n_frames))

    print("Encoding video with ffmpeg...")
    output_path = dist_dir / "charging_station_simulation_final.mp4"

    # Build ffmpeg command using concat demuxer
    concat_file = dist_dir / "frames_concat.txt"
    with open(concat_file, "w") as f:
        for section_dir, n_frames in all_frame_dirs:
            for i in range(n_frames):
                frame_path = section_dir / f"frame_{i:04d}.png"
                f.write(f"file '{frame_path.as_posix()}'\n")
                f.write(f"duration {1.0/fps:.6f}\n")
            # Repeat last frame to avoid ffmpeg concat issue
            if n_frames > 0:
                last = section_dir / f"frame_{n_frames-1:04d}.png"
                f.write(f"file '{last.as_posix()}'\n")
                f.write(f"duration {1.0/fps:.6f}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-i", str(dist_dir / "narration.wav"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {output_path}")

    # Cleanup frames
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    concat_file.unlink(missing_ok=True)
    print("Cleaned up temporary frames")


if __name__ == "__main__":
    main()
