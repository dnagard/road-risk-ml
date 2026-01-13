"""Road Risk ML - Daily Feature Pipeline

Fetches current road weather data from Trafikverket and SMHI forecasts,
then inserts into Hopsworks feature groups.
"""
# Test plan:
# 1) uv run python scripts/daily_feature_pipeline.py
# 2) Verify smhi_point_forecast v3 insert succeeds and logs version.
import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def fetch_trafikverket_weather(tv: TrafikverketClient) -> tuple:
    """Fetch weather observations from Trafikverket."""
    print("Fetching Trafikverket WeatherObservation...")
    try:
        xml = tv.build_request("WeatherObservation", "2.1", limit=1000, filter_xml="")
        payload = tv.post_xml(xml)
        df = extract_weather_observations(payload)
        print(f"  ✓ Got {len(df)} weather observations")
        return df, None
    except Exception as e:
        print(f"  ✗ Error fetching weather observations: {e}")
        return None, str(e)


def fetch_trafikverket_situations(tv: TrafikverketClient) -> tuple:
    """Fetch traffic situations (incidents/roadworks) from Trafikverket."""
    print("Fetching Trafikverket Situations...")
    try:
        xml = tv.build_request(
            "Situation", "1.6",
            namespace="road.trafficinfo",
            limit=1000,
            filter_xml=""
        )
        payload = tv.post_xml(xml)
        df = extract_situations(payload)
        print(f"  ✓ Got {len(df)} situations")
        return df, None
    except Exception as e:
        print(f"  ✗ Error fetching situations: {e}")
        return None, str(e)


def fetch_trafikverket_frostdepth(tv: TrafikverketClient) -> tuple:
    """Fetch frost depth observations from Trafikverket."""
    print("Fetching Trafikverket FrostDepthObservation...")
    try:
        xml = tv.build_request(
            "FrostDepthObservation", "1.0",
            namespace="Road.WeatherInfo",
            limit=5000,
            filter_xml=""
        )
        payload = tv.post_xml(xml)
        df = extract_frostdepth_observations(payload)
        print(f"  ✓ Got {len(df)} frost depth observations")
        return df, None
    except Exception as e:
        print(f"  ✗ Error fetching frost depth: {e}")
        return None, str(e)


def fetch_smhi_forecasts(smhi: SMHIForecastClient, locations: dict) -> tuple:
    """Fetch SMHI point forecasts for all locations."""
    print("Fetching SMHI point forecasts...")
    import pandas as pd

    all_dfs = []
    errors = []

    for loc_id, loc_info in locations.items():
        lat = loc_info["latitude"]
        lon = loc_info["longitude"]
        tv_id = loc_info.get("tv_measurepoint_id")
        if tv_id is None:
            msg = f"{loc_id}: missing tv_measurepoint_id"
            print(f"  ✗ {loc_info.get('name', loc_id)}: {msg}")
            errors.append(msg)
            continue
        try:
            payload = smhi.get_point_forecast(lat, lon)
            df = extract_smhi_point_forecast(payload, lat, lon, measurepoint_id=tv_id)
            all_dfs.append(df)
            print(f"  ✓ {loc_info['name']}: {len(df)} forecast hours")
        except Exception as e:
            print(f"  ✗ {loc_info['name']}: {e}")
            errors.append(f"{loc_id}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df, errors if errors else None
    return None, errors


def main():
    print("=" * 60)
    print("Road Risk ML - Daily Feature Pipeline")
    print("=" * 60)

    # Load locations
    locations = load_locations()
    print(f"Loaded {len(locations)} measurement points")

    if not settings.hopsworks_api_key or not settings.hopsworks_project:
        print("Missing HOPSWORKS_API_KEY/HOPSWORKS_PROJECT; exiting.")
        sys.exit(1)

    tv_enabled = bool(settings.trafikverket_api_key)
    if not tv_enabled:
        print("Missing TRAFIKVERKET_API_KEY; skipping Trafikverket fetches.")

    # Initialize clients
    tv = TrafikverketClient(settings.trafikverket_api_key, settings.trafikverket_url) if tv_enabled else None
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
    errors = []

    # --- 1) Trafikverket WeatherObservation ---
    if tv_enabled:
        df_weather, err = fetch_trafikverket_weather(tv)
        if df_weather is not None and not df_weather.empty:
            fg_weather = get_or_create_fg(
                fs,
                name="tv_weather_observation",
                version=1,
                primary_key=["measurepoint_id", "sample_time"],
                event_time="sample_time",
                description="Trafikverket road weather observations (rolling retention).",
                online_enabled=False,
            )
            print("  Writing FG tv_weather_observation v1")
            n = insert_fg(fg_weather, df_weather, dedup_keys=["measurepoint_id", "sample_time"], wait=True)
            results["weather"] = n
        else:
            if err:
                errors.append(f"weather: {err}")

    # --- 2) Trafikverket Situations ---
    if tv_enabled:
        df_sit, err = fetch_trafikverket_situations(tv)
        if df_sit is not None and not df_sit.empty:
            fg_sit = get_or_create_fg(
                fs,
                name="tv_situations",
                version=1,
                primary_key=["situation_id", "version_time"],
                event_time="version_time",
                description="Trafikverket situations (incidents/roadworks/etc).",
                online_enabled=False,
            )
            print("  Writing FG tv_situations v1")
            n = insert_fg(fg_sit, df_sit, dedup_keys=["situation_id", "version_time"], wait=True)
            results["situations"] = n
        else:
            if err:
                errors.append(f"situations: {err}")

    # --- 3) Trafikverket FrostDepthObservation ---
    if tv_enabled:
        df_fd, err = fetch_trafikverket_frostdepth(tv)
        if df_fd is not None and not df_fd.empty:
            fg_fd = get_or_create_fg(
                fs,
                name="tv_frostdepth_observation",
                version=1,
                primary_key=["measurepoint_id", "sample_time", "depth_cm"],
                event_time="sample_time",
                description="Trafikverket frost depth observations.",
                online_enabled=False,
            )
            print("  Writing FG tv_frostdepth_observation v1")
            n = insert_fg(fg_fd, df_fd, dedup_keys=["measurepoint_id", "sample_time", "depth_cm"], wait=True)
            results["frostdepth"] = n
        else:
            if err:
                errors.append(f"frostdepth: {err}")

    # --- 4) SMHI point forecasts for all locations ---
    df_fc, errs = fetch_smhi_forecasts(smhi, locations)
    if df_fc is not None and not df_fc.empty:
        fg_fc = get_or_create_fg(
            fs,
            name="smhi_point_forecast",
            version=3,
            primary_key=["measurepoint_id", "forecast_run_time", "valid_time"],
            event_time="valid_time",
            description="SMHI point forecast time series (numeric measurepoint_id).",
            online_enabled=False,
        )
        print("  Writing FG smhi_point_forecast v3")
        n = insert_fg(
            fg_fc,
            df_fc,
            dedup_keys=["measurepoint_id", "forecast_run_time", "valid_time"],
            wait=True,
        )
        results["forecast"] = n
    else:
        if errs:
            errors.extend([f"smhi: {e}" for e in errs])

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for name, count in results.items():
        print(f"  {name}: {count} rows inserted")

    if errors:
        print("\nErrors encountered:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1 if not results else 0)  # Partial success is OK

    print("\n✓ Daily feature pipeline completed successfully!")


if __name__ == "__main__":
    main()
