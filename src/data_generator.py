"""
Energy Consumption Data Generator (Assignment 12 — Energy Consumption Forecast)

Inputs : Time, Speed, Load
Output : Energy consumption (kWh for each hourly interval)

The generator produces a *sequential* (time-ordered) hourly telemetry series for
one electric delivery van. Energy is not invented randomly: it is computed from
a standard road-load equation

    P_wheel = Crr * m * g * v  +  0.5 * rho * Cd * A * v^3
    energy  = P_wheel / drivetrain_efficiency / cycle_penalty + auxiliary_load

so the target inherits real physics:

  * aerodynamic drag grows with v^3 -> the relationship between speed and energy
    consumption is strongly non-linear,
  * rolling resistance is proportional to (kerb mass + payload) * v -> the Load
    input only matters while the van is moving (a time x load interaction), and
    it is the reason a fully loaded van consumes ~20% more than an empty one,
  * stop-and-go traffic below 30 km/h costs efficiency -> the same speed costs
    more energy during rush hour,
  * payload is consumed through the delivery day (loaded in the morning, light in
    the evening) and the service window is 05:00-22:59 with only the auxiliary
    load overnight.

A slow-moving driving-conditions term (AR(1) process) is applied on top, which is
what makes the series genuinely sequential: today's consumption depends on the
recent past, not only on the current hour. That is why the models later use lag
features and why the dashboard can forecast a horizon forward.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration — vehicle, physics and simulation constants
# ---------------------------------------------------------------------------
KERB_MASS_KG = 2200          # unladen van mass
PAYLOAD_MAX_KG = 1200        # full delivery load (heavy LCV)
RHO = 1.225                  # air density (kg/m^3)
CD_A = 2.24                  # drag coefficient x frontal area (m^2)
CRR = 0.014                  # rolling resistance coefficient (loaded LCV tyre)
GRAVITY = 9.81               # m/s^2
DRIVETRAIN_EFFICIENCY = 0.88  # battery -> wheels
CITY_CYCLE_PENALTY = 0.80    # extra stop-and-go losses below 30 km/h
AUX_STANDBY_KWH = 0.55       # battery thermal management / telematics, always on
NOISE_STD_FRACTION = 0.08    # 8% measurement + behaviour noise
CONDITION_AR = 0.75          # AR(1) coefficient of driving conditions
SERVICE_HOURS = range(5, 23)  # van in service 05:00-22:59 (18 h/day)
START_DATE = "2026-01-05"    # a Monday, so weekday/weekend align with the index

# Speed thresholds used by the dashboard to describe the driving regime
URBAN_MAX_KMH = 30
MOTORWAY_MIN_KMH = 60

# Hourly energy per 100 km thresholds (EV efficiency status)
EFFICIENCY_NORMAL_MAX = 25.0   # kWh/100 km
EFFICIENCY_HIGH_MAX = 35.0     # kWh/100 km


# ---------------------------------------------------------------------------
# Explainable input profiles
# ---------------------------------------------------------------------------
def speed_profile(hour: float, day_type: str = "weekday") -> float:
    """
    Planned average speed (km/h) at a given hour — the duty cycle.

    Free-flow speed is reduced by Gaussian congestion peaks, so the two rush
    hours are slow and the middle of the day is fast. Weekends and holidays see
    much lighter congestion. Outside the service window the van is parked, so
    the planned speed is 0 and only the auxiliary load remains.
    """
    if hour not in SERVICE_HOURS:
        return 0.0

    free_flow = 88.0   # motorway leg of the delivery round
    congestion = {
        "weekday": [(8.5, 1.4, 56.0), (18.0, 1.8, 60.0)],
        "weekend": [(11.0, 2.6, 26.0), (16.5, 2.4, 34.0)],
        "holiday": [(12.0, 3.0, 16.0)],
    }.get(day_type, [(8.5, 1.4, 56.0), (18.0, 1.8, 60.0)])

    speed = free_flow
    for mean, std, drop in congestion:
        speed -= drop * np.exp(-0.5 * ((hour - mean) / std) ** 2)
    return max(speed, 8.0)


def payload_profile(hour: float, day_type: str = "weekday") -> float:
    """
    Expected payload on board (kg) at a given hour.

    The van leaves the depot loaded and the load is delivered through the day, so
    the mass on board decays. Weekends and holidays carry far less.
    """
    if hour not in SERVICE_HOURS:
        return 0.0
    day_factor = {"weekday": 1.0, "weekend": 0.55, "holiday": 0.35}.get(day_type, 1.0)
    progress = (hour - SERVICE_HOURS.start) / (len(SERVICE_HOURS) - 1)  # 0 -> 1
    remaining = max(1.0 - 0.85 * progress, 0.0)
    return PAYLOAD_MAX_KG * remaining * day_factor


def auxiliary_load(hour: float) -> float:
    """
    Auxiliary consumption (kWh per hour): cabin cooling in the afternoon, a
    smaller heating peak in the morning, plus a permanent standby draw.
    """
    cooling = 0.55 * np.exp(-0.5 * ((hour - 15.0) / 4.0) ** 2)
    heating = 0.25 * np.exp(-0.5 * ((hour - 7.0) / 2.5) ** 2)
    return AUX_STANDBY_KWH + cooling + heating


def road_load_power_kw(speed_kmh: float, payload_kg: float) -> float:
    """At-wheel power demand (kW) at a constant speed with a given payload."""
    v = speed_kmh / 3.6
    mass = KERB_MASS_KG + payload_kg
    p_rolling = CRR * mass * GRAVITY * v          # W
    p_aero = 0.5 * RHO * CD_A * v ** 3            # W
    return (p_rolling + p_aero) / 1000.0


def _base_energy(hour: float, speed_kmh: float, payload_kg: float,
                 condition: float = 1.0) -> float:
    """
    Expected energy consumption (kWh) for a one-hour interval.

    Driving power covers rolling + aerodynamic load, divided by drivetrain
    efficiency and (in dense traffic) by the stop-and-go penalty. Auxiliary load
    is added on top and applies even when the van is parked.
    """
    efficiency = DRIVETRAIN_EFFICIENCY
    if speed_kmh < URBAN_MAX_KMH:
        efficiency *= CITY_CYCLE_PENALTY

    driving_kwh = road_load_power_kw(speed_kmh, payload_kg) / efficiency
    return driving_kwh * condition + auxiliary_load(hour)


def _day_type(timestamp: pd.Timestamp, rng: np.random.Generator) -> str:
    """
    Day type follows the real calendar: Saturday and Sunday are weekends, a few
    weekdays are holidays, everything else is a working day.
    """
    if timestamp.dayofweek >= 5:
        return "weekend"
    if rng.random() < 0.06:
        return "holiday"
    return "weekday"


# ---------------------------------------------------------------------------
# Sequential generation
# ---------------------------------------------------------------------------
def _simulate_day(day: int, day_type: str, condition: float,
                  rng: np.random.Generator,
                  noise_std_fraction: float = NOISE_STD_FRACTION
                  ) -> tuple[pd.DataFrame, float]:
    """
    Simulate 24 hours and return the frame plus the carried-over state.

    Each day the depot loads a different amount (workload varies by day), which
    matters for the study: if the payload were a fixed function of the clock, the
    Load input would be perfectly collinear with Time and the regression could
    never attribute an effect to it.
    """
    day_load_factor = float(np.clip(rng.normal(1.0, 0.28), 0.45, 1.55))
    rows = []
    for hour in range(24):
        in_service = hour in SERVICE_HOURS

        # Driving conditions evolve as a slow, mean-reverting AR(1) process --
        # this is what makes consecutive hours similar to each other
        # (sequential dependence), and it is carried across day boundaries.
        condition = 1.0 + CONDITION_AR * (condition - 1.0) + rng.normal(0.0, 0.10)
        condition = float(np.clip(condition, 0.60, 1.40))

        # Speed: scheduled profile + measurement noise, zero when parked
        if in_service:
            speed = speed_profile(hour, day_type) + rng.normal(0.0, 1.5)
            speed = float(np.clip(speed, 0.0, 110.0))
        else:
            speed = 0.0

        # Load: the day's loading plan + small loading noise
        payload = payload_profile(hour, day_type) * day_load_factor
        if in_service:
            payload = float(np.clip(payload + rng.normal(0.0, 20.0), 0.0,
                                    PAYLOAD_MAX_KG))

        energy = _base_energy(hour, speed, payload, condition)
        energy += rng.normal(0.0, energy * noise_std_fraction)
        energy = max(energy, 0.0)

        rows.append({
            "day": day,
            "hour": hour,
            "day_type": day_type,
            "speed_kmh": round(speed, 2),
            "payload_kg": round(payload, 1),
            "energy_kwh": round(energy, 3),
        })
    return pd.DataFrame(rows), condition


def generate_daily_profile(day: int, day_type: str = "weekday",
                           condition: float = 1.0,
                           rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate the 24 hourly observations of one day (no state carried back)."""
    rng = rng or np.random.default_rng(0)
    return _simulate_day(day, day_type, condition, rng)[0]

def generate_dataset(num_days: int = 120, seed: int = 42,
                     noise_std_fraction: float = NOISE_STD_FRACTION) -> pd.DataFrame:
    """
    Generate the full sequential dataset.

    Default: 120 days x 24 hours = 2,880 hourly observations, in time order.
    """
    rng = np.random.default_rng(seed)
    frames = []
    condition = 1.0
    for day in range(num_days):
        timestamp = pd.Timestamp(START_DATE) + pd.Timedelta(days=day)
        day_type = _day_type(timestamp, rng)
        daily, condition = _simulate_day(day, day_type, condition, rng,
                                         noise_std_fraction)
        frames.append(daily)

    df = pd.concat(frames, ignore_index=True)

    start = pd.Timestamp(START_DATE)
    df["timestamp"] = start + pd.to_timedelta(df["day"], unit="D") \
        + pd.to_timedelta(df["hour"], unit="h")
    df["dow"] = df["timestamp"].dt.dayofweek          # 0 = Monday
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["distance_km"] = (df["speed_kmh"] * 1.0).round(2)   # one-hour intervals
    df["efficiency_kwh_100km"] = np.where(
        df["distance_km"] > 0.5,
        (df["energy_kwh"] / df["distance_km"] * 100).round(2),
        np.nan,
    )
    df["driving_regime"] = np.where(
        df["speed_kmh"] <= 0.0, "parked",
        np.where(df["speed_kmh"] < URBAN_MAX_KMH, "urban",
                 np.where(df["speed_kmh"] < MOTORWAY_MIN_KMH, "suburban", "motorway")),
    )
    return df


def generate_day_scenarios() -> pd.DataFrame:
    """
    Two clean 24-hour profiles (a working day and a quiet day) for the
    trend/seasonality figures: same shapes the dashboard lets the user explore.
    """
    rng = np.random.default_rng(99)
    busy = generate_daily_profile(0, "weekday", 1.0, rng)
    busy["scenario"] = "Working Day"
    quiet = generate_daily_profile(1, "weekend", 1.0, rng)
    quiet["scenario"] = "Weekend"
    return pd.concat([busy, quiet], ignore_index=True)


def save_dataset(df: pd.DataFrame, path: str | Path) -> Path:
    """Save the dataset to CSV and return the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating 120-day sequential energy consumption dataset...")
    df = generate_dataset(num_days=120, seed=42)
    out = save_dataset(
        df, Path(__file__).resolve().parent.parent / "data" / "van_energy_data.csv"
    )
    print(f"Saved {len(df)} rows to {out}")
    print(f"\nDataset shape: {df.shape}")
    print("\nFirst 8 hours (time order preserved):")
    print(df.head(8).to_string(index=False))
    print("\nEnergy statistics (kWh per hour):")
    print(df["energy_kwh"].describe().round(3))
    print("\nBy driving regime:")
    print(df.groupby("driving_regime")["energy_kwh"]
            .agg(["count", "mean"]).round(2))
