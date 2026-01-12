"""Road Risk ML - Training Pipeline

Trains an XGBoost classifier to predict hazardous road conditions.
Uses data from the Hopsworks feature store or falls back to synthetic data.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import hopsworks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from xgboost import XGBClassifier, plot_importance

from src.settings import settings
from src.feature_store import get_project

# Feature configuration
NUMERIC_FEATURES = [
    "surface_temp_c",
    "air_temp_c",
    "air_rh",
    "dewpoint_c",
    "surface_grip",
]

TEMPORAL_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_rush_hour",
    "is_weekend",
]

LAGGED_FEATURES = [
    "surface_temp_c_lag_1h",
    "surface_temp_c_lag_3h",
    "surface_temp_c_lag_6h",
    "surface_grip_lag_1h",
    "surface_grip_lag_3h",
]


def create_hazard_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary hazard labels based on road conditions."""
    df = df.copy()

    # Convert boolean columns
    for col in ["surface_ice", "surface_snow", "surface_water"]:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, "true": 1, "false": 0, None: 0}).fillna(0)

    # Define hazard conditions
    conditions = []
    if "surface_temp_c" in df.columns:
        conditions.append(df["surface_temp_c"].fillna(999) < 0)  # Freezing
    if "surface_ice" in df.columns:
        conditions.append(df["surface_ice"] == 1)
    if "surface_snow" in df.columns:
        conditions.append(df["surface_snow"] == 1)
    if "surface_grip" in df.columns:
        conditions.append(df["surface_grip"].fillna(1) < 0.5)  # Low grip

    if conditions:
        df["hazard"] = np.where(pd.concat(conditions, axis=1).any(axis=1), 1, 0)
    else:
        df["hazard"] = 0

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from timestamp."""
    df = df.copy()

    if "sample_time" in df.columns:
        ts = pd.to_datetime(df["sample_time"])
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["month"] = ts.dt.month
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return create_hazard_labels(df)


def create_synthetic_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training data with realistic patterns."""
    np.random.seed(seed)

    # Generate timestamps spanning multiple months
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="h")
    month = dates.month

    # Seasonal temperature (coldest in Jan/Feb)
    base_temp = -5 + 20 * np.sin((month - 1) * np.pi / 6)
    daily_var = 3 * np.sin((dates.hour - 6) * np.pi / 12)

    df = pd.DataFrame({
        "sample_time": dates,
        "measurepoint_id": np.random.choice(
            ["MP001", "MP002", "MP003", "MP004", "MP005"], n_samples
        ),
        "surface_temp_c": base_temp + daily_var + np.random.normal(0, 3, n_samples),
        "air_temp_c": base_temp + daily_var + 2 + np.random.normal(0, 2, n_samples),
        "air_rh": np.clip(70 + np.random.normal(0, 15, n_samples), 20, 100),
        "dewpoint_c": base_temp - 5 + np.random.normal(0, 2, n_samples),
        "surface_grip": np.clip(0.8 + np.random.normal(0, 0.15, n_samples), 0, 1),
    })

    # Create correlated ice/snow based on temperature
    df["surface_ice"] = ((df["surface_temp_c"] < 0) & (np.random.random(n_samples) > 0.4)).astype(int)
    df["surface_snow"] = ((df["surface_temp_c"] < -2) & (np.random.random(n_samples) > 0.5)).astype(int)

    # Reduce grip when ice/snow present
    df.loc[df["surface_ice"] == 1, "surface_grip"] *= 0.4
    df.loc[df["surface_snow"] == 1, "surface_grip"] *= 0.6

    # Add lagged features
    df = df.sort_values(["measurepoint_id", "sample_time"])
    for col in ["surface_temp_c", "surface_grip"]:
        for lag in [1, 3, 6]:
            df[f"{col}_lag_{lag}h"] = df.groupby("measurepoint_id")[col].shift(lag)

    return df.dropna()


def time_based_split(df: pd.DataFrame, test_days: int = 7) -> tuple:
    """Split data using time-based approach to avoid data leakage."""
    df = df.copy()

    if "sample_time" not in df.columns:
        # Fall back to random split
        from sklearn.model_selection import train_test_split
        return train_test_split(df, test_size=0.2, random_state=42)

    df["sample_time"] = pd.to_datetime(df["sample_time"])
    cutoff = df["sample_time"].max() - timedelta(days=test_days)

    train_df = df[df["sample_time"] < cutoff]
    test_df = df[df["sample_time"] >= cutoff]

    print(f"  Train: {len(train_df)} samples (before {cutoff.date()})")
    print(f"  Test: {len(test_df)} samples (from {cutoff.date()})")

    return train_df, test_df


def plot_confusion_matrix(y_true, y_pred, save_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["Safe", "Hazard"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall(y_true, y_proba, save_path: str):
    """Plot and save precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, "b-", linewidth=2)
    ax.axhline(y=y_true.mean(), color="r", linestyle="--", label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP={ap:.3f})")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_from_feature_store(fs) -> pd.DataFrame:
    """Load training data from Hopsworks feature store."""
    try:
        # Try to get existing feature view
        try:
            fv = fs.get_feature_view(name="road_risk_fv", version=1)
            print("  ✓ Using existing feature view")
        except:
            # Create feature view from weather observations
            weather_fg = fs.get_feature_group(name="tv_weather_observation", version=1)

            query = weather_fg.select_all()

            fv = fs.create_feature_view(
                name="road_risk_fv",
                version=1,
                query=query,
                labels=["hazard"] if "hazard" in weather_fg.features else [],
            )
            print("  ✓ Created new feature view")

        # Read training data
        df = fv.get_batch_data()
        print(f"  ✓ Loaded {len(df)} samples from feature store")
        return df

    except Exception as e:
        print(f"  ✗ Could not load from feature store: {e}")
        return pd.DataFrame()


def main():
    print("=" * 60)
    print("Road Risk ML - Training Pipeline")
    print("=" * 60)

    # Connect to Hopsworks
    print("\nConnecting to Hopsworks...")
    try:
        project = get_project(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
        fs = project.get_feature_store()
        print("  ✓ Connected to feature store")
    except Exception as e:
        print(f"  ✗ Failed to connect to Hopsworks: {e}")
        sys.exit(1)

    # Load data
    print("\nLoading training data...")
    try:
        weather_fg = fs.get_feature_group(name="tv_weather_observation", version=1)
        df = weather_fg.read()
        print(f"  ✓ Loaded {len(df)} weather observations from feature store")
    except Exception as e:
        print(f"  ✗ Could not load from feature store: {e}")
        print("  Creating synthetic training data...")
        df = create_synthetic_data(n_samples=10000)

    # Prepare features
    print("\nPreparing features...")
    df = prepare_features(df)

    # Determine available features
    all_features = NUMERIC_FEATURES + TEMPORAL_FEATURES + LAGGED_FEATURES
    available_features = [c for c in all_features if c in df.columns]
    print(f"  Using {len(available_features)} features: {available_features}")

    # Clean data
    df_clean = df.dropna(subset=available_features + ["hazard"])
    print(f"  {len(df_clean)} samples after removing missing values")

    # Fall back to synthetic if not enough data
    if len(df_clean) < 500:
        print("  Insufficient data, using synthetic data...")
        df_clean = prepare_features(create_synthetic_data(10000))
        available_features = [c for c in all_features if c in df_clean.columns]

    # Time-based train/test split
    print("\nSplitting data (time-based)...")
    train_df, test_df = time_based_split(df_clean, test_days=7)

    X_train = train_df[available_features]
    y_train = train_df["hazard"]
    X_test = test_df[available_features]
    y_test = test_df["hazard"]

    # Handle class imbalance
    pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    print(f"\nClass distribution:")
    print(f"  Train - Safe: {len(y_train[y_train==0])}, Hazard: {len(y_train[y_train==1])}")
    print(f"  Test - Safe: {len(y_test[y_test==0])}, Hazard: {len(y_test[y_test==1])}")
    print(f"  Scale pos weight: {pos_weight:.2f}")

    # Train model
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=100,
        scale_pos_weight=pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    print("  ✓ Training complete")

    # Evaluate
    print("\nEvaluation Results:")
    print("-" * 40)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # print(classification_report(y_test, y_pred, target_names=["Safe", "Hazard"]))

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    accuracy = (y_pred == y_test).mean()

    print(f"ROC AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save model locally
    print("\nSaving model artifacts...")
    model_dir = "road_risk_model"
    os.makedirs(f"{model_dir}/images", exist_ok=True)

    model.save_model(f"{model_dir}/model.json")
    with open(f"{model_dir}/features.json", "w") as f:
        json.dump(available_features, f)

    # Generate plots
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=15)
    plt.tight_layout()
    plt.savefig(f"{model_dir}/images/feature_importance.png")
    plt.close()

    # plot_confusion_matrix(y_test, y_pred, f"{model_dir}/images/confusion_matrix.png")
    # plot_precision_recall(y_test, y_pred_proba, f"{model_dir}/images/precision_recall.png")

    print(f"  ✓ Saved model to {model_dir}/")

    # Register model in Hopsworks
    print("\nRegistering model in Hopsworks...")
    mr = project.get_model_registry()

    # Handle NaN values in metrics (replace with 0.0)
    metrics = {
        "accuracy": float(accuracy) if not np.isnan(accuracy) else 0.0,
        "auc": float(auc) if not np.isnan(auc) else 0.0,
        "average_precision": float(ap) if not np.isnan(ap) else 0.0,
    }

    road_risk_model = mr.python.create_model(
        name="road_risk_xgboost",
        metrics=metrics,
        description="XGBoost classifier for road hazard prediction (ice, snow, low grip)",
    )
    road_risk_model.save(model_dir)

    print(f"  ✓ Model registered: road_risk_xgboost v{road_risk_model.version}")
    print("\n" + "=" * 60)
    print("Training pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
