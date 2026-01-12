"""Road Risk ML - Batch Inference Pipeline

Generates road hazard predictions for the next 24 hours using
trained model and weather forecasts.
"""
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.settings import settings
from src.feature_store import get_project
from src.clients import SMHIForecastClient
from src.transforms import extract_smhi_point_forecast

# Stockholm measurement points
MEASUREPOINTS = {
    "MP001": {"name": "E4 Norrtull", "lat": 59.357, "lon": 18.05},
    "MP002": {"name": "E4 Häggvik", "lat": 59.433, "lon": 17.933},
    "MP003": {"name": "E18 Jakobsberg", "lat": 59.422, "lon": 17.833},
    "MP004": {"name": "E20 Essingeleden", "lat": 59.327, "lon": 18.0},
    "MP005": {"name": "Rv73 Nynäsvägen", "lat": 59.267, "lon": 18.083},
}


def load_model(model_dir: str):
    """Load model and feature list from directory."""
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, "model.json"))
    with open(os.path.join(model_dir, "features.json"), "r") as f:
        features = json.load(f)
    return model, features


def fetch_smhi_forecasts() -> pd.DataFrame:
    """Fetch SMHI forecasts for all measurement points."""
    print("Fetching SMHI forecasts...")
    smhi = SMHIForecastClient()

    all_dfs = []
    for mp_id, mp_info in MEASUREPOINTS.items():
        try:
            payload = smhi.get_point_forecast(mp_info["lat"], mp_info["lon"])
            df = extract_smhi_point_forecast(payload, mp_info["lat"], mp_info["lon"])
            df["measurepoint_id"] = mp_id
            all_dfs.append(df)
            print(f"  ✓ {mp_info['name']}: {len(df)} hours")
        except Exception as e:
            print(f"  ✗ {mp_info['name']}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def fetch_recent_observations(fs) -> pd.DataFrame:
    """Fetch recent weather observations for lagged features."""
    try:
        weather_fg = fs.get_feature_group(name="tv_weather_observation", version=1)

        # Get last 24 hours of observations
        cutoff = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        df = weather_fg.filter(weather_fg.sample_time >= cutoff).read()

        if not df.empty:
            print(f"  ✓ Got {len(df)} recent observations")
        return df
    except Exception as e:
        print(f"  ✗ Could not fetch recent observations: {e}")
        return pd.DataFrame()


def prepare_forecast_features(
    forecast_df: pd.DataFrame,
    recent_obs_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Prepare features from forecast data for prediction."""
    rows = []
    now = datetime.now()

    for mp_id in MEASUREPOINTS.keys():
        mp_forecast = forecast_df[forecast_df["measurepoint_id"] == mp_id]

        # Get last known observation for this measurepoint
        last_obs = None
        if recent_obs_df is not None and not recent_obs_df.empty:
            mp_obs = recent_obs_df[recent_obs_df["measurepoint_id"] == mp_id]
            if not mp_obs.empty:
                last_obs = mp_obs.sort_values("sample_time").iloc[-1]

        for _, fc in mp_forecast.iterrows():
            valid_time = pd.to_datetime(fc["valid_time"])

            # Skip past times
            if valid_time < now:
                continue

            # Estimate surface temperature (typically 2-3°C below air temp at night)
            air_temp = fc.get("t_air_c", 0) or 0
            hour = valid_time.hour
            night_factor = 2 if hour < 6 or hour > 20 else 1
            surface_temp = air_temp - night_factor

            # Estimate grip based on conditions
            base_grip = 0.85
            if surface_temp < 0:
                base_grip *= 0.6  # Freezing
            if fc.get("precip_mm", 0) and fc.get("precip_mm", 0) > 0:
                base_grip *= 0.8  # Wet

            # Use last observation for lagged features if available
            lag_surface_temp = last_obs["surface_temp_c"] if last_obs is not None else surface_temp
            lag_grip = last_obs["surface_grip"] if last_obs is not None else base_grip

            rows.append({
                "measurepoint_id": mp_id,
                "valid_time": valid_time,
                # Core features
                "surface_temp_c": surface_temp,
                "air_temp_c": air_temp,
                "air_rh": fc.get("rh", 70) or 70,
                "dewpoint_c": air_temp - 5,  # Approximate
                "surface_grip": base_grip,
                # Temporal features
                "hour": valid_time.hour,
                "day_of_week": valid_time.dayofweek,
                "month": valid_time.month,
                "is_rush_hour": int(valid_time.hour in [7, 8, 9, 16, 17, 18]),
                "is_weekend": int(valid_time.dayofweek >= 5),
                # Lagged features (approximated)
                "surface_temp_c_lag_1h": lag_surface_temp,
                "surface_temp_c_lag_3h": lag_surface_temp,
                "surface_temp_c_lag_6h": lag_surface_temp,
                "surface_grip_lag_1h": lag_grip,
                "surface_grip_lag_3h": lag_grip,
                # Forecast metadata
                "precip_mm": fc.get("precip_mm", 0) or 0,
                "wind_ms": fc.get("wind_ms", 0) or 0,
            })

    return pd.DataFrame(rows)


def get_recommendation(risk: float, temp: float) -> str:
    """Generate maintenance recommendation based on risk level."""
    if risk > 0.7:
        if temp < -5:
            return "URGENT: Pre-salt roads immediately, monitor conditions closely"
        return "HIGH: Salt application recommended within 2 hours"
    elif risk > 0.4:
        return "MODERATE: Standby for salting, prepare equipment"
    return "LOW: Normal operations, continue monitoring"


def create_synthetic_forecast(hours: int = 48) -> pd.DataFrame:
    """Create synthetic forecast data for demo purposes."""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    times = [now + timedelta(hours=h) for h in range(hours)]

    # Seasonal base temperature
    month = datetime.now().month
    is_winter = month in [11, 12, 1, 2, 3]
    base_temp = -3 if is_winter else 5

    rows = []
    for mp_id in MEASUREPOINTS.keys():
        for t in times:
            daily_var = 3 * np.sin((t.hour - 6) * np.pi / 12)
            rows.append({
                "measurepoint_id": mp_id,
                "valid_time": t,
                "t_air_c": base_temp + daily_var + np.random.normal(0, 1),
                "rh": 70 + np.random.normal(0, 10),
                "precip_mm": max(0, np.random.normal(-0.5, 0.5)),
                "wind_ms": max(0, np.random.normal(3, 2)),
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Road Risk ML - Batch Inference Pipeline")
    print("=" * 60)

    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    try:
        project = get_project(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        print("  ✓ Connected to feature store")
    except Exception as e:
        print(f"  ✗ Failed to connect to Hopsworks: {e}")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    try:
        retrieved_model = mr.get_model(name="road_risk_xgboost", version=1)
        model_dir = retrieved_model.download()
        model, features = load_model(model_dir)
        print(f"  ✓ Loaded model v{retrieved_model.version} from registry")
    except Exception as e:
        print(f"  ✗ Could not load from registry: {e}")
        print("  Trying local model...")
        try:
            model, features = load_model("road_risk_model")
            print("  ✓ Loaded local model")
        except Exception as e2:
            print(f"  ✗ Could not load local model: {e2}")
            print("  Please run training_pipeline.py first!")
            return

    # Fetch SMHI forecasts
    print("\nFetching forecast data...")
    forecast_df = fetch_smhi_forecasts()

    if forecast_df.empty:
        print("  Using synthetic forecast (SMHI unavailable)")
        forecast_df = create_synthetic_forecast()

    # Fetch recent observations for lagged features
    print("\nFetching recent observations...")
    recent_obs = fetch_recent_observations(fs)

    # Prepare features
    print("\nPreparing prediction features...")
    pred_df = prepare_forecast_features(forecast_df, recent_obs)

    # Filter to next 24 hours
    now = datetime.now()
    pred_df = pred_df[
        (pd.to_datetime(pred_df["valid_time"]) >= now) &
        (pd.to_datetime(pred_df["valid_time"]) <= now + timedelta(hours=24))
    ]

    if pred_df.empty:
        print("  No valid forecast times, using synthetic data")
        forecast_df = create_synthetic_forecast()
        pred_df = prepare_forecast_features(forecast_df, recent_obs)
        pred_df = pred_df[
            (pd.to_datetime(pred_df["valid_time"]) >= now) &
            (pd.to_datetime(pred_df["valid_time"]) <= now + timedelta(hours=24))
        ]

    print(f"  Prepared {len(pred_df)} prediction records")

    # Make predictions
    print("\nGenerating predictions...")

    # Select only features the model expects
    available_features = [f for f in features if f in pred_df.columns]
    missing_features = [f for f in features if f not in pred_df.columns]
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        for f in missing_features:
            pred_df[f] = 0

    X_pred = pred_df[features].fillna(0)
    pred_df["risk_probability"] = model.predict_proba(X_pred)[:, 1]
    pred_df["hazard_predicted"] = (pred_df["risk_probability"] > 0.5).astype(int)
    pred_df["recommendation"] = pred_df.apply(
        lambda r: get_recommendation(r["risk_probability"], r["surface_temp_c"]),
        axis=1
    )
    pred_df["forecast_date"] = date.today()

    # Prepare output
    output_cols = [
        "measurepoint_id", "valid_time", "risk_probability", "hazard_predicted",
        "recommendation", "forecast_date", "surface_temp_c", "air_temp_c"
    ]
    output_df = pred_df[output_cols].copy()
    output_df["risk_probability"] = output_df["risk_probability"].astype("float32")

    # Save to feature store
    print("\nSaving predictions...")
    try:
        predictions_fg = fs.get_or_create_feature_group(
            name="road_risk_predictions",
            version=1,
            primary_key=["measurepoint_id", "valid_time", "forecast_date"],
            event_time="valid_time",
            description="Road hazard predictions for next 24 hours",
        )
        predictions_fg.insert(output_df, write_options={"wait_for_job": True})
        print("  ✓ Predictions saved to Hopsworks feature store")
    except Exception as e:
        print(f"  ✗ Could not save to Hopsworks: {e}")
        output_df.to_csv("predictions.csv", index=False)
        print("  ✓ Predictions saved to predictions.csv")

    # Summary
    print("\n" + "=" * 60)
    print("Prediction Summary:")
    print("=" * 60)

    high_risk = len(pred_df[pred_df["risk_probability"] > 0.7])
    med_risk = len(pred_df[(pred_df["risk_probability"] > 0.4) & (pred_df["risk_probability"] <= 0.7)])
    low_risk = len(pred_df[pred_df["risk_probability"] <= 0.4])

    print(f"  HIGH risk periods: {high_risk}")
    print(f"  MEDIUM risk periods: {med_risk}")
    print(f"  LOW risk periods: {low_risk}")

    # Show highest risk periods
    if high_risk > 0:
        print("\nHighest risk periods:")
        high_risk_df = pred_df[pred_df["risk_probability"] > 0.7].sort_values(
            "risk_probability", ascending=False
        ).head(5)
        for _, row in high_risk_df.iterrows():
            mp_name = MEASUREPOINTS.get(row["measurepoint_id"], {}).get("name", row["measurepoint_id"])
            print(f"  - {mp_name} at {row['valid_time']}: {row['risk_probability']:.0%} risk")

    print("\n✓ Batch inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
