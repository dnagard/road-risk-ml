"""Road Risk ML - Batch Inference Pipeline

Generates calibrated hazard forecasts for 24/48/72 hour horizons using
SMHI point forecasts and trained ensembles.
"""
# Test plan:
# 1) uv run python scripts/batch_inference_pipeline.py
# 2) Verify road_risk_predictions v3 insert and numeric measurepoint_id.
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.settings import settings
from src.feature_store import get_project, get_or_create_fg, insert_fg
from src.clients import SMHIForecastClient
from src.transforms import extract_smhi_point_forecast
from src.utils import to_datetime64ns_utc

HORIZONS = [24, 48, 72]
LOCATIONS_FILE = Path(__file__).parent.parent / "src" / "locations.json"
OBS_FEATURES = [
    "obs_surface_temp_c",
    "obs_surface_grip",
    "obs_air_temp_c",
    "obs_air_rh",
    "obs_dewpoint_c",
    "obs_surface_ice",
    "obs_surface_snow",
    "obs_surface_water",
    "obs_age_minutes",
]


def load_locations() -> dict:
    try:
        with open(LOCATIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def to_utc_naive(series: pd.Series) -> pd.Series:
    return to_datetime64ns_utc(series)


def add_temporal_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[time_col])
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_rush_hour"] = ts.dt.hour.isin([7, 8, 9, 16, 17, 18]).astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def coerce_measurepoint_id(df: pd.DataFrame, col: str = "measurepoint_id") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns or df.empty:
        return df
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df[col].notna()]
    df[col] = df[col].astype(int)
    return df


def build_current_obs_features(obs_df: pd.DataFrame) -> pd.DataFrame:
    if obs_df.empty:
        return obs_df

    obs_df = obs_df.copy()
    if "sample_time" not in obs_df.columns:
        return pd.DataFrame()

    obs_df = coerce_measurepoint_id(obs_df)
    if obs_df.empty:
        return obs_df

    obs_df["sample_time"] = to_datetime64ns_utc(obs_df["sample_time"])

    def _to_bool_int(series: pd.Series) -> pd.Series:
        mapped = series.map(
            {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0, 1: 1, 0: 0}
        )
        numeric = pd.to_numeric(series, errors="coerce")
        combined = mapped.combine_first(numeric)
        return combined.fillna(0).astype(int)

    for col in ["surface_ice", "surface_snow", "surface_water"]:
        if col in obs_df.columns:
            obs_df[col] = _to_bool_int(obs_df[col])

    required = [
        "surface_temp_c",
        "surface_grip",
        "air_temp_c",
        "air_rh",
        "dewpoint_c",
        "surface_ice",
        "surface_snow",
        "surface_water",
    ]
    for col in required:
        if col not in obs_df.columns:
            obs_df[col] = np.nan
        else:
            obs_df[col] = pd.to_numeric(obs_df[col], errors="coerce")

    keep_cols = ["measurepoint_id", "sample_time"] + required
    return obs_df[keep_cols]


def attach_current_obs_features(fc_df: pd.DataFrame, obs_df: pd.DataFrame) -> pd.DataFrame:
    fc_df = fc_df.copy()
    if obs_df.empty:
        for col in OBS_FEATURES:
            fc_df[col] = np.nan
        fc_df["obs_time"] = pd.NaT
        fc_df["obs_age_minutes"] = 1e6
        return fc_df

    fc_df = coerce_measurepoint_id(fc_df)
    if fc_df.empty:
        for col in OBS_FEATURES:
            fc_df[col] = np.nan
        fc_df["obs_time"] = pd.NaT
        fc_df["obs_age_minutes"] = 1e6
        return fc_df

    obs_features = build_current_obs_features(obs_df)
    if obs_features.empty:
        for col in OBS_FEATURES:
            fc_df[col] = np.nan
        fc_df["obs_time"] = pd.NaT
        fc_df["obs_age_minutes"] = 1e6
        return fc_df

    if "forecast_run_time" not in fc_df.columns:
        for col in OBS_FEATURES:
            fc_df[col] = np.nan
        fc_df["obs_time"] = pd.NaT
        fc_df["obs_age_minutes"] = 1e6
        return fc_df

    fc_df["forecast_run_time"] = to_datetime64ns_utc(fc_df["forecast_run_time"])
    obs_features["sample_time"] = to_datetime64ns_utc(obs_features["sample_time"])

    print(
        "  Obs join dtypes:",
        f"forecast_run_time={fc_df['forecast_run_time'].dtype},",
        f"sample_time={obs_features['sample_time'].dtype}",
    )
    print(
        "  Forecast time range:",
        f"{fc_df['forecast_run_time'].min()} -> {fc_df['forecast_run_time'].max()}",
    )
    print(
        "  Observation time range:",
        f"{obs_features['sample_time'].min()} -> {obs_features['sample_time'].max()}",
    )

    assert fc_df["forecast_run_time"].dtype == obs_features["sample_time"].dtype

    fc_df = fc_df.sort_values(["measurepoint_id", "forecast_run_time"]).reset_index(drop=True)
    obs_features = obs_features.sort_values(["measurepoint_id", "sample_time"]).reset_index(drop=True)
    print("  Sorted forecast and observation tables for merge_asof")

    merged = pd.merge_asof(
        fc_df,
        obs_features,
        left_on="forecast_run_time",
        right_on="sample_time",
        by="measurepoint_id",
        direction="backward",
        allow_exact_matches=True,
    )

    rename_map = {
        "surface_temp_c": "obs_surface_temp_c",
        "surface_grip": "obs_surface_grip",
        "air_temp_c": "obs_air_temp_c",
        "air_rh": "obs_air_rh",
        "dewpoint_c": "obs_dewpoint_c",
        "surface_ice": "obs_surface_ice",
        "surface_snow": "obs_surface_snow",
        "surface_water": "obs_surface_water",
        "sample_time": "obs_time",
    }
    merged = merged.rename(columns=rename_map)
    merged["obs_age_minutes"] = (
        (merged["forecast_run_time"] - merged["obs_time"]).dt.total_seconds() / 60.0
    )
    merged["obs_age_minutes"] = merged["obs_age_minutes"].fillna(1e6)
    return merged


def apply_calibrator(calibrator, probs: np.ndarray, method: str) -> np.ndarray:
    if calibrator is None:
        return probs
    method = (method or "platt").lower()
    if method == "isotonic":
        return calibrator.predict(probs)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(probs / (1 - probs)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def load_ensemble(model_dir: str) -> tuple[list, list, list, str]:
    model_dir = Path(model_dir)
    with open(model_dir / "features.json", "r") as f:
        features = json.load(f)

    ensemble_path = model_dir / "ensemble.json"
    models = []
    calibrators = []
    method = "platt"

    if ensemble_path.exists():
        with open(ensemble_path, "r") as f:
            manifest = json.load(f)
        method = manifest.get("calibration_method", "platt")
        members = manifest.get("members", [])
        for member in members:
            model = XGBClassifier()
            model.load_model(str(model_dir / member["model_file"]))
            models.append(model)
            calibrator_path = model_dir / member.get("calibrator_file", "")
            calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None
            calibrators.append(calibrator)
    else:
        model = XGBClassifier()
        model.load_model(str(model_dir / "model.json"))
        models.append(model)
        calibrator_path = model_dir / "calibrator.joblib"
        calibrators.append(joblib.load(calibrator_path) if calibrator_path.exists() else None)

    return models, calibrators, features, method


def predict_ensemble(models: list, calibrators: list, X: pd.DataFrame, method: str) -> pd.DataFrame:
    probs = []
    for model, calibrator in zip(models, calibrators):
        raw = model.predict_proba(X)[:, 1]
        calibrated = apply_calibrator(calibrator, raw, method)
        probs.append(calibrated)

    prob_matrix = np.vstack(probs).T
    mean = prob_matrix.mean(axis=1)
    p10 = np.quantile(prob_matrix, 0.1, axis=1)
    p90 = np.quantile(prob_matrix, 0.9, axis=1)
    return mean, p10, p90


def fetch_smhi_forecasts(locations: dict) -> pd.DataFrame:
    print("Fetching SMHI forecasts...")
    smhi = SMHIForecastClient()

    all_dfs = []
    for loc_id, loc_info in locations.items():
        lat = loc_info["latitude"]
        lon = loc_info["longitude"]
        tv_id = loc_info.get("tv_measurepoint_id")
        if tv_id is None:
            print(f"  ✗ {loc_info.get('name', loc_id)}: missing tv_measurepoint_id")
            continue
        try:
            payload = smhi.get_point_forecast(lat, lon)
            df = extract_smhi_point_forecast(payload, lat, lon, measurepoint_id=tv_id)
            all_dfs.append(df)
            print(f"  ✓ {loc_info['name']}: {len(df)} hours")
        except Exception as e:
            print(f"  ✗ {loc_info['name']}: {e}")

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def prepare_forecast_features(forecast_df: pd.DataFrame) -> pd.DataFrame:
    df = forecast_df.copy()

    df = coerce_measurepoint_id(df)
    if df.empty:
        return df

    df["valid_time"] = to_utc_naive(df["valid_time"])
    if "forecast_run_time" in df.columns:
        df["forecast_run_time"] = to_utc_naive(df["forecast_run_time"])
    else:
        df["forecast_run_time"] = df["valid_time"]

    df["horizon_hours"] = (
        (df["valid_time"] - df["forecast_run_time"]).dt.total_seconds() / 3600.0
    ).round().astype("Int64")

    df = add_temporal_features(df, "valid_time")
    return df


def get_recommendation(risk_mean: float, risk_p90: float | None = None) -> str:
    upper = risk_p90 if risk_p90 is not None else risk_mean
    if risk_mean >= 0.7 or upper >= 0.85:
        return "HIGH: Salting recommended within 2 hours"
    if risk_mean >= 0.4 or upper >= 0.6:
        return "MODERATE: Standby for salting, prepare equipment"
    return "LOW: Normal operations, continue monitoring"


def create_synthetic_forecast(hours: int = 72) -> pd.DataFrame:
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    run_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    times = [run_time + timedelta(hours=h) for h in range(hours + 1)]

    month = run_time.month
    is_winter = month in [11, 12, 1, 2, 3]
    base_temp = -3 if is_winter else 5

    locations = load_locations()
    rows = []
    for mp_id, mp_info in locations.items():
        tv_id = mp_info.get("tv_measurepoint_id", mp_id)
        for t in times:
            daily_var = 3 * np.sin((t.hour - 6) * np.pi / 12)
            rows.append(
                {
                    "measurepoint_id": tv_id,
                    "lat": mp_info["latitude"],
                    "lon": mp_info["longitude"],
                    "forecast_run_time": run_time,
                    "valid_time": t,
                    "t_air_c": base_temp + daily_var + np.random.normal(0, 1),
                    "rh": 70 + np.random.normal(0, 10),
                    "precip_mm": max(0, np.random.normal(-0.2, 0.4)),
                    "wind_ms": max(0, np.random.normal(3, 2)),
                }
            )

    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("Road Risk ML - Batch Inference Pipeline")
    print("=" * 60)

    locations = load_locations()
    if not locations:
        print("  ✗ Missing locations.json; cannot run inference")
        return

    project = None
    fs = None
    mr = None

    if settings.hopsworks_api_key and settings.hopsworks_project:
        print("\nConnecting to Hopsworks...")
        try:
            project = get_project(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
            fs = project.get_feature_store()
            mr = project.get_model_registry()
            print("  ✓ Connected to feature store")
        except Exception as e:
            print(f"  ✗ Failed to connect to Hopsworks: {e}")
    else:
        print("\nHopsworks credentials missing; using local models and CSV output")

    print("\nLoading models...")
    models_by_horizon = {}
    for horizon in HORIZONS:
        model_dir = None
        if mr is not None:
            try:
                retrieved = mr.get_model(name=f"road_risk_xgb_h{horizon}")
                model_dir = retrieved.download()
                print(f"  ✓ Loaded road_risk_xgb_h{horizon} from registry")
            except Exception as e:
                print(f"  WARN: Registry model for {horizon}h not available: {e}")

        if model_dir is None:
            local_dir = Path("road_risk_models") / f"road_risk_xgb_h{horizon}"
            if local_dir.exists():
                model_dir = str(local_dir)
                print(f"  ✓ Loaded local road_risk_xgb_h{horizon}")

        if model_dir is None:
            print(f"  ✗ No model found for {horizon}h")
            continue

        try:
            models, calibrators, features, method = load_ensemble(model_dir)
            models_by_horizon[horizon] = {
                "models": models,
                "calibrators": calibrators,
                "features": features,
                "calibration_method": method,
            }
        except Exception as e:
            print(f"  ✗ Failed to load ensemble for {horizon}h: {e}")

    if not models_by_horizon:
        print("  ✗ No models loaded; aborting")
        return

    print("\nFetching forecast data...")
    forecast_df = fetch_smhi_forecasts(locations)
    if forecast_df.empty:
        print("  Using synthetic forecast (SMHI unavailable)")
        forecast_df = create_synthetic_forecast(hours=72)

    features_df = prepare_forecast_features(forecast_df)
    if features_df.empty:
        print("  ✗ No forecast rows after ID normalization")
        return

    obs_df = pd.DataFrame()
    if fs is not None:
        try:
            print("\nLoading current observations...")
            print("  Reading FG tv_weather_observation v1")
            obs_fg = fs.get_feature_group(name="tv_weather_observation", version=1)
            obs_df = obs_fg.read()
            obs_df = coerce_measurepoint_id(obs_df)
            print(f"  ✓ Loaded {len(obs_df)} observations")
        except Exception as e:
            print(f"  WARN: Could not load observations: {e}")
            obs_df = pd.DataFrame()

    features_df = attach_current_obs_features(features_df, obs_df)
    if "obs_time" in features_df.columns:
        attached = int(features_df["obs_time"].notna().sum())
        print(f"  ✓ Attached current observations to {attached} forecast rows")

    features_df = features_df[features_df["horizon_hours"].isin(HORIZONS)]

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    features_df = features_df[
        (features_df["valid_time"] >= now) &
        (features_df["valid_time"] <= now + timedelta(hours=72))
    ]

    if features_df.empty:
        print("  ✗ No valid forecast rows for requested horizons")
        return

    output_frames = []
    for horizon, bundle in models_by_horizon.items():
        horizon_df = features_df[features_df["horizon_hours"] == horizon].copy()
        if horizon_df.empty:
            continue

        features = bundle["features"]
        for feature in features:
            if feature not in horizon_df.columns:
                horizon_df[feature] = 0

        X_pred = horizon_df[features].fillna(0)
        mean, p10, p90 = predict_ensemble(
            bundle["models"],
            bundle["calibrators"],
            X_pred,
            bundle.get("calibration_method", "platt"),
        )

        horizon_df["risk_mean"] = mean
        horizon_df["risk_p10"] = p10
        horizon_df["risk_p90"] = p90
        horizon_df["hazard_predicted"] = (horizon_df["risk_mean"] >= 0.5).astype(int)
        horizon_df["recommendation"] = horizon_df.apply(
            lambda r: get_recommendation(r["risk_mean"], r["risk_p90"]), axis=1
        )

        output_frames.append(
            horizon_df[[
                "measurepoint_id",
                "valid_time",
                "horizon_hours",
                "risk_mean",
                "risk_p10",
                "risk_p90",
                "hazard_predicted",
                "recommendation",
                "forecast_run_time",
            ]].copy()
        )

    if not output_frames:
        print("  ✗ No predictions generated")
        return

    output_df = pd.concat(output_frames, ignore_index=True)

    print("\nSaving predictions...")
    if fs is not None:
        try:
            predictions_fg = get_or_create_fg(
                fs,
                name="road_risk_predictions",
                version=3,
                primary_key=["measurepoint_id", "forecast_run_time", "valid_time", "horizon_hours"],
                event_time="valid_time",
                description="Road hazard forecasts for 24/48/72h horizons (numeric measurepoint_id).",
                online_enabled=False,
            )
            print("  Writing FG road_risk_predictions v3")
            insert_fg(
                predictions_fg,
                output_df,
                dedup_keys=["measurepoint_id", "forecast_run_time", "valid_time", "horizon_hours"],
                wait=True,
            )
            print("  ✓ Predictions saved to Hopsworks feature store")
        except Exception as e:
            print(f"  ✗ Could not save to Hopsworks: {e}")
            output_df.to_csv("predictions.csv", index=False)
            print("  ✓ Predictions saved to predictions.csv")
    else:
        output_df.to_csv("predictions.csv", index=False)
        print("  ✓ Predictions saved to predictions.csv")

    print("\n" + "=" * 60)
    print("Prediction Summary:")
    print("=" * 60)
    high_risk = len(output_df[output_df["risk_mean"] > 0.7])
    med_risk = len(output_df[(output_df["risk_mean"] > 0.4) & (output_df["risk_mean"] <= 0.7)])
    low_risk = len(output_df[output_df["risk_mean"] <= 0.4])

    print(f"  HIGH risk periods: {high_risk}")
    print(f"  MEDIUM risk periods: {med_risk}")
    print(f"  LOW risk periods: {low_risk}")
    print("\n✓ Batch inference pipeline completed successfully!")


if __name__ == "__main__":
    main()
