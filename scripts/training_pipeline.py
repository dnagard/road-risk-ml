"""Road Risk ML - Training Pipeline

Trains calibrated ensemble XGBoost classifiers to forecast hazardous road
conditions 24/48/72 hours ahead using SMHI forecast features and observed
Trafikverket outcomes.
"""
# Test plan:
# 1) uv run python scripts/training_pipeline.py
# 2) Confirm smhi_point_forecast v3 + tv_weather_observation v1 load and merge without dtype errors.
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from src.settings import settings
from src.feature_store import get_project
from src.utils import to_datetime64ns_utc

HORIZONS = [24, 48, 72]
ENSEMBLE_SIZE = int(os.getenv("ENSEMBLE_SIZE", "10"))
TEST_DAYS = int(os.getenv("TEST_DAYS", "14"))
CALIBRATION_DAYS = int(os.getenv("CALIBRATION_DAYS", "7"))
CALIBRATION_METHOD = os.getenv("CALIBRATION_METHOD", "platt")

LOCATIONS_FILE = Path(__file__).parent.parent / "src" / "locations.json"

FORECAST_FEATURES = [
    "t_air_c",
    "precip_mm",
    "wind_ms",
    "rh",
]
STATIC_FEATURES = [
    "lat",
    "lon",
]
TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_rush_hour",
    "is_weekend",
]
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

    left_nat = int(fc_df["forecast_run_time"].isna().sum())
    right_nat = int(obs_features["sample_time"].isna().sum())
    if left_nat:
        fc_df = fc_df[fc_df["forecast_run_time"].notna()].copy()
    if right_nat:
        obs_features = obs_features[obs_features["sample_time"].notna()].copy()

    left_dupes = 0
    if "valid_time" in fc_df.columns:
        left_dupes = int(fc_df.duplicated(subset=["measurepoint_id", "forecast_run_time", "valid_time"]).sum())
        if left_dupes:
            fc_df = fc_df.drop_duplicates(subset=["measurepoint_id", "forecast_run_time", "valid_time"], keep="last")
    right_dupes = int(obs_features.duplicated(subset=["measurepoint_id", "sample_time"]).sum())
    if right_dupes:
        obs_features = obs_features.drop_duplicates(subset=["measurepoint_id", "sample_time"], keep="last")

    fc_df = fc_df.sort_values(["forecast_run_time", "measurepoint_id"]).reset_index(drop=True)
    obs_features = obs_features.sort_values(["sample_time", "measurepoint_id"]).reset_index(drop=True)

    left_mono = fc_df["forecast_run_time"].is_monotonic_increasing
    right_mono = obs_features["sample_time"].is_monotonic_increasing
    print(
        "  Obs join prep:",
        "left_sort=['forecast_run_time','measurepoint_id']",
        "right_sort=['sample_time','measurepoint_id']",
        f"left_monotonic={left_mono}",
        f"right_monotonic={right_mono}",
        f"left_nat_dropped={left_nat}",
        f"right_nat_dropped={right_nat}",
        f"left_dupes_dropped={left_dupes}",
        f"right_dupes_dropped={right_dupes}",
    )
    if not left_mono or not right_mono:
        print("  WARN: Non-monotonic join keys after sort; re-sorting")
        fc_df = fc_df.sort_values(["forecast_run_time", "measurepoint_id"]).reset_index(drop=True)
        obs_features = obs_features.sort_values(["sample_time", "measurepoint_id"]).reset_index(drop=True)
        left_mono = fc_df["forecast_run_time"].is_monotonic_increasing
        right_mono = obs_features["sample_time"].is_monotonic_increasing

    assert left_mono
    assert right_mono

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


def create_hazard_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary hazard labels based on observed road conditions."""
    df = df.copy()

    for col in ["surface_ice", "surface_snow", "surface_water"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "true": 1, "false": 0}).fillna(0)

    conditions = []
    if "surface_temp_c" in df.columns:
        conditions.append(df["surface_temp_c"].fillna(999) < 0)
    if "surface_ice" in df.columns:
        conditions.append(df["surface_ice"] == 1)
    if "surface_snow" in df.columns:
        conditions.append(df["surface_snow"] == 1)
    if "surface_grip" in df.columns:
        conditions.append(df["surface_grip"].fillna(1) < 0.5)

    if conditions:
        df["hazard"] = np.where(pd.concat(conditions, axis=1).any(axis=1), 1, 0)
    else:
        df["hazard"] = 0

    return df


def build_label_table(obs_df: pd.DataFrame) -> pd.DataFrame:
    obs_df = create_hazard_labels(obs_df)
    obs_df["sample_time"] = to_utc_naive(obs_df["sample_time"])
    obs_df["label_time"] = obs_df["sample_time"].dt.floor("h")
    label_df = (
        obs_df.groupby(["measurepoint_id", "label_time"], as_index=False)["hazard"]
        .max()
        .rename(columns={"label_time": "valid_time"})
    )
    return label_df


def build_training_dataset(fs) -> pd.DataFrame:
    try:
        print("  Reading FG smhi_point_forecast v3")
        fc_fg = fs.get_feature_group(name="smhi_point_forecast", version=3)
        print("  Reading FG tv_weather_observation v1")
        obs_fg = fs.get_feature_group(name="tv_weather_observation", version=1)
        fc_df = fc_fg.read()
        obs_df = obs_fg.read()
        print(f"  ✓ Loaded {len(fc_df)} forecast rows and {len(obs_df)} observations")
    except Exception as e:
        print(f"  ✗ Could not load feature groups: {e}")
        return pd.DataFrame()

    if fc_df.empty or obs_df.empty:
        print("  ✗ Feature groups are empty")
        return pd.DataFrame()

    fc_df = coerce_measurepoint_id(fc_df)
    obs_df = coerce_measurepoint_id(obs_df)
    if fc_df.empty or obs_df.empty:
        print("  ✗ Feature groups are empty after ID normalization")
        return pd.DataFrame()

    fc_df = fc_df.copy()
    if "forecast_run_time" not in fc_df.columns:
        fc_df["forecast_run_time"] = pd.NaT

    fc_df["valid_time"] = to_utc_naive(fc_df["valid_time"])
    fc_df["forecast_run_time"] = to_utc_naive(fc_df["forecast_run_time"])

    if "ingested_at" in fc_df.columns:
        fc_df["ingested_at"] = to_utc_naive(fc_df["ingested_at"])
        fc_df["forecast_run_time"] = fc_df["forecast_run_time"].fillna(fc_df["ingested_at"])

    fc_df["forecast_run_time"] = fc_df["forecast_run_time"].fillna(fc_df["valid_time"])
    fc_df["horizon_hours"] = (
        (fc_df["valid_time"] - fc_df["forecast_run_time"]).dt.total_seconds() / 3600.0
    ).round().astype("Int64")

    fc_df = fc_df[fc_df["horizon_hours"].isin(HORIZONS)].copy()
    if fc_df.empty:
        print("  ✗ No forecast rows matched requested horizons")
        return pd.DataFrame()

    fc_df = attach_current_obs_features(fc_df, obs_df)
    if "obs_time" in fc_df.columns:
        attached = int(fc_df["obs_time"].notna().sum())
        print(f"  ✓ Attached current observations to {attached} forecast rows")

    label_df = build_label_table(obs_df)

    merged = fc_df.merge(label_df, on=["measurepoint_id", "valid_time"], how="inner")
    merged = merged[merged["measurepoint_id"].notna()]
    if merged.empty:
        print("  ✗ No matching labels for forecast rows")
        return pd.DataFrame()

    merged = add_temporal_features(merged, "valid_time")
    return merged


def create_synthetic_training_data(days: int = 45) -> pd.DataFrame:
    np.random.seed(42)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    locations = load_locations()
    measurepoints = [
        (int(info["tv_measurepoint_id"]), info["latitude"], info["longitude"])
        for info in locations.values()
        if info.get("tv_measurepoint_id") is not None
    ]
    if not measurepoints:
        measurepoints = [
            (243, 59.357, 18.05),
            (226, 59.433, 17.933),
            (232, 59.422, 17.833),
            (237, 59.327, 18.0),
            (215, 59.267, 18.083),
        ]

    rows = []
    for mp_id, lat, lon in measurepoints:
        for day in range(days):
            for hour in range(24):
                valid_time = now - timedelta(days=days - day, hours=24 - hour)
                month = valid_time.month
                base_temp = -5 + 15 * np.sin((month - 1) * np.pi / 6)
                daily_var = 3 * np.sin((valid_time.hour - 6) * np.pi / 12)
                for horizon in HORIZONS:
                    forecast_run_time = valid_time - timedelta(hours=horizon)
                    t_air_c = base_temp + daily_var + np.random.normal(0, 2)
                    precip_mm = max(0, np.random.normal(0.2, 0.4))
                    wind_ms = max(0, np.random.normal(4, 2))
                    rh = np.clip(70 + np.random.normal(0, 15), 20, 100)

                    obs_air_temp_c = t_air_c + np.random.normal(0, 0.8)
                    obs_surface_temp_c = obs_air_temp_c - np.random.uniform(0, 2)
                    obs_surface_ice = int(obs_surface_temp_c < 0 and np.random.random() > 0.3)
                    obs_surface_snow = int(obs_surface_temp_c < -1 and np.random.random() > 0.5)
                    obs_surface_water = int(np.random.random() > 0.8)
                    obs_surface_grip = np.clip(0.85 - obs_surface_ice * 0.35 - obs_surface_snow * 0.2, 0.1, 1.0)

                    hazard = int((t_air_c < 0 and precip_mm > 0.1) or rh > 85)

                    rows.append(
                        {
                            "measurepoint_id": mp_id,
                            "lat": lat,
                            "lon": lon,
                            "forecast_run_time": pd.Timestamp(forecast_run_time),
                            "valid_time": pd.Timestamp(valid_time),
                            "horizon_hours": horizon,
                            "t_air_c": t_air_c,
                            "precip_mm": precip_mm,
                            "wind_ms": wind_ms,
                            "rh": rh,
                            "obs_surface_temp_c": obs_surface_temp_c,
                            "obs_surface_grip": obs_surface_grip,
                            "obs_air_temp_c": obs_air_temp_c,
                            "obs_air_rh": np.clip(rh + np.random.normal(0, 5), 20, 100),
                            "obs_dewpoint_c": obs_air_temp_c - 5 + np.random.normal(0, 1),
                            "obs_surface_ice": obs_surface_ice,
                            "obs_surface_snow": obs_surface_snow,
                            "obs_surface_water": obs_surface_water,
                            "obs_age_minutes": np.random.randint(0, 180),
                            "hazard": hazard,
                        }
                    )

    df = pd.DataFrame(rows)
    df = add_temporal_features(df, "valid_time")
    print(f"  ✓ Generated {len(df)} synthetic training rows")
    return df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["forecast_run_time"] = to_utc_naive(df["forecast_run_time"])

    if df["forecast_run_time"].isna().all():
        print("  WARN: Forecast run times missing; using random split")
        return random_split(df)

    cutoff_test = df["forecast_run_time"].max() - timedelta(days=TEST_DAYS)
    test_df = df[df["forecast_run_time"] >= cutoff_test]
    train_df = df[df["forecast_run_time"] < cutoff_test]

    if train_df.empty or test_df.empty:
        print("  WARN: Time split produced empty set; using random split")
        return random_split(df)

    cutoff_cal = train_df["forecast_run_time"].max() - timedelta(days=CALIBRATION_DAYS)
    calib_df = train_df[train_df["forecast_run_time"] >= cutoff_cal]
    train_df = train_df[train_df["forecast_run_time"] < cutoff_cal]

    if train_df.empty or calib_df.empty:
        print("  WARN: Calibration split produced empty set; using random split")
        return random_split(df)

    return train_df, calib_df, test_df


def random_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=42)
    n = len(df)
    test_cut = int(n * 0.2)
    calib_cut = int(n * 0.35)
    test_df = df.iloc[:test_cut]
    calib_df = df.iloc[test_cut:calib_cut]
    train_df = df.iloc[calib_cut:]
    return train_df, calib_df, test_df


def fit_platt_calibrator(probs: np.ndarray, y: pd.Series) -> LogisticRegression:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(probs / (1 - probs)).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs")
    calibrator.fit(logit, y)
    return calibrator


def apply_platt_calibrator(calibrator: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logit = np.log(probs / (1 - probs)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def fit_calibrator(probs: np.ndarray, y: pd.Series):
    if CALIBRATION_METHOD.lower() == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probs, y)
        return calibrator
    return fit_platt_calibrator(probs, y)


def apply_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probs
    if CALIBRATION_METHOD.lower() == "isotonic":
        return calibrator.predict(probs)
    return apply_platt_calibrator(calibrator, probs)


def train_ensemble(X_train: pd.DataFrame, y_train: pd.Series, pos_weight: float) -> list:
    models = []
    base_params = {
        "max_depth": 5,
        "learning_rate": 0.08,
        "n_estimators": 200,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "scale_pos_weight": pos_weight,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    for idx in range(ENSEMBLE_SIZE):
        params = dict(base_params)
        params["random_state"] = 42 + idx
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        models.append(model)

    return models


def evaluate_ensemble(
    models: list,
    calibrators: list,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    preds = []
    for model, calibrator in zip(models, calibrators):
        raw = model.predict_proba(X_test)[:, 1]
        calibrated = apply_calibrator(calibrator, raw)
        preds.append(calibrated)

    pred_matrix = np.vstack(preds).T
    mean_pred = pred_matrix.mean(axis=1)

    metrics = {
        "auc": float(roc_auc_score(y_test, mean_pred)) if len(np.unique(y_test)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_test, mean_pred)) if len(np.unique(y_test)) > 1 else 0.0,
        "brier": float(brier_score_loss(y_test, mean_pred)),
        "accuracy": float(((mean_pred >= 0.5).astype(int) == y_test).mean()),
    }
    return metrics


def save_ensemble_artifacts(
    out_dir: Path,
    horizon: int,
    features: list,
    models: list,
    calibrators: list,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "features.json", "w") as f:
        json.dump(features, f)

    manifest = {
        "horizon_hours": horizon,
        "calibration_method": CALIBRATION_METHOD,
        "members": [],
    }

    for idx, (model, calibrator) in enumerate(zip(models, calibrators)):
        model_file = f"model_{idx}.json"
        calibrator_file = f"calibrator_{idx}.joblib"
        model.save_model(str(out_dir / model_file))
        joblib.dump(calibrator, out_dir / calibrator_file)
        manifest["members"].append(
            {
                "model_file": model_file,
                "calibrator_file": calibrator_file,
            }
        )

    with open(out_dir / "ensemble.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    print("=" * 60)
    print("Road Risk ML - Training Pipeline")
    print("=" * 60)

    project = None
    fs = None

    if settings.hopsworks_api_key and settings.hopsworks_project:
        print("\nConnecting to Hopsworks...")
        try:
            project = get_project(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
            fs = project.get_feature_store()
            print("  ✓ Connected to feature store")
        except Exception as e:
            print(f"  ✗ Failed to connect to Hopsworks: {e}")
    else:
        print("\nHopsworks credentials missing; using synthetic data")

    print("\nBuilding training dataset...")
    if fs is not None:
        df = build_training_dataset(fs)
    else:
        df = pd.DataFrame()

    if df.empty:
        print("  Using synthetic training data...")
        df = create_synthetic_training_data(days=45)

    features_all = FORECAST_FEATURES + STATIC_FEATURES + TEMPORAL_FEATURES + OBS_FEATURES
    available_features = [c for c in features_all if c in df.columns]
    print(f"  Using features: {available_features}")

    df = df.dropna(subset=["hazard", "horizon_hours"])

    model_root = Path("road_risk_models")
    model_root.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        print("\n" + "-" * 50)
        print(f"Training horizon {horizon}h")
        horizon_df = df[df["horizon_hours"] == horizon].copy()

        if horizon_df.empty:
            print("  WARN: No data for this horizon; skipping")
            continue

        train_df, calib_df, test_df = time_split(horizon_df)

        X_train = train_df[available_features].fillna(0)
        y_train = train_df["hazard"].astype(int)
        X_calib = calib_df[available_features].fillna(0)
        y_calib = calib_df["hazard"].astype(int)
        X_test = test_df[available_features].fillna(0)
        y_test = test_df["hazard"].astype(int)

        pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        print(
            f"  Samples - train: {len(train_df)}, calib: {len(calib_df)}, test: {len(test_df)} | pos_weight={pos_weight:.2f}"
        )

        models = train_ensemble(X_train, y_train, pos_weight)
        calibrators = []
        for model in models:
            raw_probs = model.predict_proba(X_calib)[:, 1]
            calibrator = fit_calibrator(raw_probs, y_calib)
            calibrators.append(calibrator)

        metrics = evaluate_ensemble(models, calibrators, X_test, y_test)
        print(
            "  Metrics - "
            f"AUC: {metrics['auc']:.3f} | AP: {metrics['average_precision']:.3f} | "
            f"Brier: {metrics['brier']:.3f} | Acc: {metrics['accuracy']:.3f}"
        )

        out_dir = model_root / f"road_risk_xgb_h{horizon}"
        save_ensemble_artifacts(out_dir, horizon, available_features, models, calibrators)

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        if project is not None:
            try:
                mr = project.get_model_registry()
                model_entry = mr.python.create_model(
                    name=f"road_risk_xgb_h{horizon}",
                    metrics=metrics,
                    description=(
                        f"Calibrated XGBoost ensemble for {horizon}h hazard forecasting "
                        "using SMHI forecast features"
                    ),
                )
                model_entry.save(str(out_dir))
                print(f"  ✓ Registered model: road_risk_xgb_h{horizon} v{model_entry.version}")
            except Exception as e:
                print(f"  ✗ Failed to register model: {e}")

    print("\n" + "=" * 60)
    print("Training pipeline completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
