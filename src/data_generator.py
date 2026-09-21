"""
Charging Station Load Data Generator

Generates realistic simulated data for EV charging station load prediction.
Simulates daily patterns with peak/off-peak variation, controlled randomness,
and realistic vehicle-load relationships.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STATION_CAPACITY_KW = 300  # Maximum station capacity
HOURS_IN_DAY = 24
NOISE_STD_FRACTION = 0.12  # 12% noise relative to base load

# Peak hours: 07:00-10:00, 17:00-21:00
PEAK_HOURS = set(range(7, 11)) | set(range(17, 21))

# Off-peak: 22:00-06:00
OFFPEAK_HOURS = set(range(22, 24)) | set(range(0, 7))


def _vehicle_arrival_profile(hour: float, day_type: str = "weekday") -> float:
    """
    Return expected number of vehicles at a given hour.

    Uses a mixture of Gaussian peaks to mimic real arrival patterns:
      - Morning commute peak ~08:00
      - Lunch peak ~13:00
      - Evening peak ~18:00 (largest)
      - Late-night trough ~03:00
    """
    profiles = {
        "weekday": [
            {"mean": 8.5, "std": 1.5, "weight": 0.35},   # morning commute
            {"mean": 13.0, "std": 1.8, "weight": 0.25},   # lunch
            {"mean": 18.0, "std": 2.0, "weight": 0.40},   # evening
        ],
        "weekend": [
            {"mean": 11.0, "std": 2.5, "weight": 0.45},   # late morning
            {"mean": 16.0, "std": 2.5, "weight": 0.55},   # afternoon
        ],
        "holiday": [
            {"mean": 12.0, "std": 3.0, "weight": 1.0},    # spread out
        ],
    }

    peaks = profiles.get(day_type, profiles["weekday"])
    total = 0.0
    for p in peaks:
        total += p["weight"] * np.exp(-0.5 * ((hour - p["mean"]) / p["std"]) ** 2)
    return max(total, 0.02)


def _base_load(hour: float, vehicles: float) -> float:
    """
    Compute expected load (kW) from time and vehicle count.

    The relationship is approximately:
        load ≈ vehicles * avg_power_per_vehicle

    But avg power varies by time (fast chargers used more during peak,
    slower overnight charging), creating a nonlinear relationship.
    """
    # Average power per vehicle varies by time of day
    if hour in PEAK_HOURS:
        avg_power = 5.0 + np.random.normal(0, 0.4)  # fast charging during peak
    elif hour in OFFPEAK_HOURS:
        avg_power = 3.5 + np.random.normal(0, 0.3)  # slower overnight
    else:
        avg_power = 4.2 + np.random.normal(0, 0.35)  # transitional

    base = vehicles * max(avg_power, 1.5)

    # Add small time-dependent base load (lighting, HVAC, standby)
    time_load = 5.0 + 3.0 * np.sin(2 * np.pi * hour / 24)

    return base + time_load


def generate_daily_profile(
    day: int,
    day_type: str = "weekday",
    noise_std_fraction: float = NOISE_STD_FRACTION,
) -> pd.DataFrame:
    """
    Generate 24 hourly observations for a single day.

    Returns DataFrame with columns:
        day, hour, day_type, vehicles, load_kw
    """
    rows = []
    for hour in range(HOURS_IN_DAY):
        # Expected vehicle count from arrival profile
        expected_vehicles = _vehicle_arrival_profile(hour, day_type)

        # Scale to realistic station size (0-60 vehicles)
        max_vehicles = 60
        vehicles = int(np.clip(
            expected_vehicles * max_vehicles + np.random.normal(0, 2),
            0,
            max_vehicles,
        ))

        # Base load from time + vehicles
        load = _base_load(hour, vehicles)

        # Add controlled noise: epsilon ~ N(0, sigma)
        noise = np.random.normal(0, load * noise_std_fraction)
        load = max(load + noise, 0.0)

        # Cap at station capacity
        load = min(load, STATION_CAPACITY_KW)

        rows.append({
            "day": day,
            "hour": hour,
            "day_type": day_type,
            "vehicles": vehicles,
            "load_kw": round(load, 2),
        })

    return pd.DataFrame(rows)


def generate_dataset(
    num_days: int = 90,
    seed: int = 42,
    noise_std_fraction: float = NOISE_STD_FRACTION,
) -> pd.DataFrame:
    """
    Generate a multi-day charging station dataset.

    Default: 90 days x 24 hours = 2160 observations.

    Day types are assigned to create realistic variation:
        - ~55% weekdays
        - ~30% weekends
        - ~15% holidays (low demand days)
    """
    np.random.seed(seed)

    all_days = []
    for day in range(num_days):
        # Assign day type with realistic distribution
        r = np.random.random()
        if r < 0.55:
            day_type = "weekday"
        elif r < 0.85:
            day_type = "weekend"
        else:
            day_type = "holiday"

        daily = generate_daily_profile(day, day_type, noise_std_fraction)
        all_days.append(daily)

    df = pd.concat(all_days, ignore_index=True)

    # Add a proper timestamp column
    start_date = pd.Timestamp("2025-01-01")
    df["timestamp"] = df.apply(
        lambda r: start_date + pd.Timedelta(days=r["day"], hours=r["hour"]),
        axis=1,
    )

    return df


def generate_peak_offpeak_scenarios() -> pd.DataFrame:
    """
    Generate two clean 24-hour profiles: peak day vs off-peak day.
    Used for load-curve comparison visualization.
    """
    np.random.seed(99)

    peak = generate_daily_profile(0, day_type="weekday", noise_std_fraction=0.06)
    peak["scenario"] = "Peak Day"

    offpeak = generate_daily_profile(1, day_type="holiday", noise_std_fraction=0.06)
    offpeak["scenario"] = "Off-Peak Day"

    return pd.concat([peak, offpeak], ignore_index=True)


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Save dataset to CSV. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating 90-day charging station dataset...")
    df = generate_dataset(num_days=90, seed=42)
    out = save_dataset(df, Path(__file__).resolve().parent.parent / "data" / "charging_station_data.csv")
    print(f"Saved {len(df)} rows to {out}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))
    print(f"\nStatistics:")
    print(df[["vehicles", "load_kw"]].describe().round(2))
