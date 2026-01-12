"""Road Risk ML - Backfill Feature Pipeline

Loads historical road weather data from Trafikverket and historical weather
from SMHI to populate the feature store for model training.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.settings import settings
from src.clients import TrafikverketClient, SMHIForecastClient
from src.transforms import (
    extract_weather_observations,
    extract_situations,
    extract_frostdepth_observations,
    extract_smhi_point_forecast,
)
from src.feature_store import get_fs, get_or_create_fg, insert_fg

LOCATIONS_FILE = Path(__file__).parent.parent / "src" / "locations.json"


def load_locations() -> dict:
    with open(LOCATIONS_FILE) as f:
        return json.load(f)


def fetch_historical_weather(tv: TrafikverketClient, days_back: int = 7) -> pd.DataFrame:
    """Fetch historical weather observations from Trafikverket.

    Note: Trafikverket API typically only provides recent data (last few days).
    For longer historical periods, you may need to use archived datasets.
    """
    print(f"Fetching Trafikverket WeatherObservation (last {days_back} days)...")

    # Trafikverket provides data with timestamps, we just query recent data
    # The API returns whatever is available
    try:
        xml = tv.build_request(
            "WeatherObservation", "2.1",
            limit=10000,
            filter_xml=""
        )
        payload = tv.post_xml(xml)
        df = extract_weather_observations(payload)
        print(f"  ✓ Got {len(df)} weather observations")
        return df
    except Exception as e:
        print(f"  ✗ Error fetching weather observations: {e}")
        return pd.DataFrame()


def fetch_historical_situations(tv: TrafikverketClient) -> pd.DataFrame:
    """Fetch historical traffic situations from Trafikverket."""
    print("Fetching Trafikverket Situations...")

    try:
        xml = tv.build_request(
            "Situation", "1.6",
            namespace="road.trafficinfo",
            limit=10000,
            filter_xml=""
        )
        payload = tv.post_xml(xml)
        df = extract_situations(payload)
        print(f"  ✓ Got {len(df)} situations")
        return df
    except Exception as e:
        print(f"  ✗ Error fetching situations: {e}")
        return pd.DataFrame()


def fetch_historical_frostdepth(tv: TrafikverketClient) -> pd.DataFrame:
    """Fetch historical frost depth observations from Trafikverket."""
    print("Fetching Trafikverket FrostDepthObservation...")

    try:
        xml = tv.build_request(
            "FrostDepthObservation", "1.0",
            namespace="Road.WeatherInfo",
            limit=10000,
            filter_xml=""
        )
        payload = tv.post_xml(xml)
        df = extract_frostdepth_observations(payload)
        print(f"  ✓ Got {len(df)} frost depth observations")
        return df
    except Exception as e:
        print(f"  ✗ Error fetching frost depth: {e}")
        return pd.DataFrame()


def fetch_smhi_forecasts(smhi: SMHIForecastClient, locations: dict) -> pd.DataFrame:
    """Fetch SMHI point forecasts for all locations."""
    print("Fetching SMHI point forecasts...")

    all_dfs = []
    for loc_id, loc_info in locations.items():
        lat = loc_info["latitude"]
        lon = loc_info["longitude"]
        try:
            payload = smhi.get_point_forecast(lat, lon)
            df = extract_smhi_point_forecast(payload, lat, lon)
            df["location_id"] = loc_id
            all_dfs.append(df)
            print(f"  ✓ {loc_info['name']}: {len(df)} forecast hours")
        except Exception as e:
            print(f"  ✗ {loc_info['name']}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def add_lagged_features(df: pd.DataFrame, lag_hours: list = [1, 3, 6, 12, 24]) -> pd.DataFrame:
    """Add lagged features for temporal patterns.

    Creates lagged versions of key features to capture temporal autocorrelation.
    """
    if df.empty:
        return df

    df = df.copy()
    df = df.sort_values(["measurepoint_id", "sample_time"])

    lag_cols = ["surface_temp_c", "surface_grip", "air_temp_c", "air_rh"]

    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_hours:
            # Group by measurepoint and shift
            df[f"{col}_lag_{lag}h"] = df.groupby("measurepoint_id")[col].shift(lag)

    return df


def create_synthetic_historical_data(locations: dict, days: int = 30) -> pd.DataFrame:
    """Create synthetic historical data for testing/demo purposes.

    This generates realistic-looking data based on Swedish winter conditions.
    """
    import numpy as np

    print(f"Generating synthetic historical data ({days} days)...")
    np.random.seed(42)

    rows = []
    now = datetime.now()

    for loc_id, loc_info in locations.items():
        for day in range(days):
            for hour in range(24):
                ts = now - timedelta(days=days - day, hours=24 - hour)

                # Seasonal temperature pattern (colder in winter)
                month = ts.month
                base_temp = -5 + 15 * abs(6 - abs(month - 6)) / 6  # Coldest in Jan

                # Daily temperature variation
                daily_var = 3 * np.sin((hour - 6) * np.pi / 12)

                # Random variation
                temp = base_temp + daily_var + np.random.normal(0, 2)
                surface_temp = temp - np.random.uniform(0, 3)  # Surface usually colder

                # Ice/snow conditions based on temperature
                has_ice = surface_temp < 0 and np.random.random() > 0.4
                has_snow = surface_temp < -2 and np.random.random() > 0.5

                # Grip coefficient
                base_grip = 0.85
                if has_ice:
                    base_grip *= 0.4
                if has_snow:
                    base_grip *= 0.6
                grip = np.clip(base_grip + np.random.normal(0, 0.05), 0.1, 1.0)

                rows.append({
                    "measurepoint_id": loc_id,
                    "measurepoint_name": loc_info["name"],
                    "wgs84": f"POINT ({loc_info['longitude']} {loc_info['latitude']})",
                    "sample_time": pd.Timestamp(ts).tz_localize(None),
                    "surface_temp_c": round(surface_temp, 1),
                    "surface_water": np.random.random() > 0.8,
                    "surface_ice": has_ice,
                    "surface_snow": has_snow,
                    "surface_grip": round(grip, 2),
                    "air_temp_c": round(temp, 1),
                    "air_rh": round(np.clip(70 + np.random.normal(0, 15), 30, 100), 1),
                    "dewpoint_c": round(temp - 5 + np.random.normal(0, 2), 1),
                    "ingested_at": pd.Timestamp.now().tz_localize(None),
                })

    df = pd.DataFrame(rows)
    print(f"  ✓ Generated {len(df)} synthetic observations")
    return df


def main():
    print("=" * 60)
    print("Road Risk ML - Backfill Feature Pipeline")
    print("=" * 60)

    # Load locations
    locations = load_locations()
    print(f"Loaded {len(locations)} measurement points")

    # Initialize clients
    tv = TrafikverketClient(settings.trafikverket_api_key, settings.trafikverket_url)
    smhi = SMHIForecastClient()

    # Connect to feature store
    print("\nConnecting to Hopsworks...")
    try:
        fs = get_fs(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
        print("  ✓ Connected to feature store")
    except Exception as e:
        print(f"  ✗ Failed to connect to Hopsworks: {e}")
        sys.exit(1)

    results = {}

    # --- 1) Weather observations ---
    print("\n--- Weather Observations ---")
    df_weather = fetch_historical_weather(tv)

    # If no real data, use synthetic
    if df_weather.empty:
        print("  Using synthetic data (no historical data available from API)")
        df_weather = create_synthetic_historical_data(locations, days=30)

    # Add lagged features
    df_weather = add_lagged_features(df_weather)

    if not df_weather.empty:
        fg_weather = get_or_create_fg(
            fs,
            name="tv_weather_observation",
            version=1,
            primary_key=["measurepoint_id", "sample_time"],
            event_time="sample_time",
            description="Trafikverket road weather observations with lagged features.",
            online_enabled=False,
        )
        n = insert_fg(fg_weather, df_weather, dedup_keys=["measurepoint_id", "sample_time"], wait=True)
        results["weather"] = n
        print(f"  ✓ Inserted {n} weather observations")

    # --- 2) Traffic situations ---
    print("\n--- Traffic Situations ---")
    df_sit = fetch_historical_situations(tv)

    if not df_sit.empty:
        fg_sit = get_or_create_fg(
            fs,
            name="tv_situations",
            version=1,
            primary_key=["situation_id", "version_time"],
            event_time="version_time",
            description="Trafikverket situations (incidents/roadworks/etc).",
            online_enabled=False,
        )
        n = insert_fg(fg_sit, df_sit, dedup_keys=["situation_id", "version_time"], wait=True)
        results["situations"] = n
        print(f"  ✓ Inserted {n} situations")

    # --- 3) Frost depth ---
    print("\n--- Frost Depth Observations ---")
    df_fd = fetch_historical_frostdepth(tv)

    if not df_fd.empty:
        fg_fd = get_or_create_fg(
            fs,
            name="tv_frostdepth_observation",
            version=1,
            primary_key=["measurepoint_id", "sample_time", "depth_cm"],
            event_time="sample_time",
            description="Trafikverket frost depth observations.",
            online_enabled=False,
        )
        n = insert_fg(fg_fd, df_fd, dedup_keys=["measurepoint_id", "sample_time", "depth_cm"], wait=True)
        results["frostdepth"] = n
        print(f"  ✓ Inserted {n} frost depth observations")

    # --- 4) SMHI forecasts ---
    print("\n--- SMHI Forecasts ---")
    df_fc = fetch_smhi_forecasts(smhi, locations)

    if not df_fc.empty:
        fg_fc = get_or_create_fg(
            fs,
            name="smhi_point_forecast",
            version=1,
            primary_key=["lat", "lon", "valid_time"],
            event_time="valid_time",
            description="SMHI point forecast time series for measurement locations.",
            online_enabled=False,
        )
        n = insert_fg(fg_fc, df_fc, dedup_keys=["lat", "lon", "valid_time"], wait=True)
        results["forecast"] = n
        print(f"  ✓ Inserted {n} forecast records")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Backfill Summary:")
    print("=" * 60)
    total = 0
    for name, count in results.items():
        print(f"  {name}: {count} rows")
        total += count
    print(f"  TOTAL: {total} rows inserted")
    print("\n✓ Backfill feature pipeline completed successfully!")


if __name__ == "__main__":
    main()
