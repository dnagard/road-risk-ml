# Road Risk ML

A machine learning system for predicting hazardous road conditions on Stockholm roads, designed to help municipalities proactively plan road maintenance actions such as salting or snow-ploughing.

## Overview

This project implements a serverless ML pipeline that:
- Collects real-time road weather data from **Trafikverket** (Swedish Transport Administration)
- Integrates weather forecasts from **SMHI** (Swedish Meteorological and Hydrological Institute)
- Trains calibrated XGBoost ensembles to forecast hazardous conditions 24/48/72 hours ahead
- Generates probabilistic predictions with uncertainty bands
- Displays results in an interactive Streamlit dashboard

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Trafikverket   │    │      SMHI       │    │   Historical    │
│  (Road Weather) │    │   (Forecasts)   │    │     Data        │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Feature Pipeline    │
                    │  (Daily/Backfill)     │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Hopsworks        │
                    │    Feature Store      │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
    ┌─────────▼─────────┐ ┌─────▼─────┐ ┌────────▼────────┐
    │ Training Pipeline │ │   Model   │ │ Batch Inference │
    │    (Weekly)       │ │  Registry │ │     (Daily)     │
    └───────────────────┘ └───────────┘ └────────┬────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │     Dashboard       │
                                      │    (Streamlit)      │
                                      └─────────────────────┘
```

## Data Sources

### Trafikverket API
- **WeatherObservation**: Real-time road surface conditions (temperature, ice, snow, grip)
- **Situation**: Traffic incidents and roadworks
- **FrostDepthObservation**: Subsurface temperature measurements

### SMHI API
- **Point Forecast**: 10-day hourly weather forecasts for specific coordinates

## Measurement Points

The system monitors 5 key road segments in the Stockholm area:

| Alias | TV ID (measurepoint_id) | Name              | Road  | Coordinates           |
|-------|--------------------------|-------------------|-------|-----------------------|
| MP001 | 243                      | E4 Norrtull       | E4    | 59.357°N, 18.050°E    |
| MP002 | 226                      | E4 Häggvik        | E4    | 59.433°N, 17.933°E    |
| MP003 | 232                      | E18 Jakobsberg    | E18   | 59.422°N, 17.833°E    |
| MP004 | 237                      | E20 Essingeleden  | E20   | 59.327°N, 18.000°E    |
| MP005 | 215                      | Rv73 Nynäsvägen   | Rv73  | 59.267°N, 18.083°E    |

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys (see Configuration)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/road-risk-ml.git
cd road-risk-ml

# Install dependencies
uv sync

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

```env
# Hopsworks Feature Store
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_project_name

# Trafikverket API
TRAFIKVERKET_API_KEY=your_trafikverket_api_key

# Optional: Default location for SMHI forecasts
DEFAULT_LAT=59.3293
DEFAULT_LON=18.0686
```

### Getting API Keys

1. **Hopsworks**: Sign up at [hopsworks.ai](https://www.hopsworks.ai/) and create a project
2. **Trafikverket**: Register at [trafikinfo.trafikverket.se](https://api.trafikinfo.trafikverket.se/)
3. **SMHI**: No API key required (open data)

## Usage

### Running Pipelines

```bash
# 1. Initial backfill (populate feature store with historical data)
uv run python scripts/backfill_feature_pipeline.py

# 2. Train the models (24/48/72h horizons)
uv run python scripts/training_pipeline.py

# 3. Generate predictions (writes road_risk_predictions with risk_mean/P10/P90)
uv run python scripts/batch_inference_pipeline.py

# 4. Run the dashboard
uv run streamlit run src/dashboard.py
```

Training artifacts are saved to `road_risk_models/road_risk_xgb_h24`, `road_risk_models/road_risk_xgb_h48`, and `road_risk_models/road_risk_xgb_h72`. If Hopsworks credentials are configured, the models are also registered in the model registry.

### Daily Operations

The system is designed to run automatically via GitHub Actions:
- **Daily (05:00 UTC)**: Feature pipeline + batch inference
- **Weekly (Sunday 02:00 UTC)**: Model retraining

## Model

### Features (Forecast + Current Observations)

**Forecast Features:**
- `t_air_c`: Air temperature (SMHI forecast)
- `precip_mm`: Mean precipitation (SMHI forecast)
- `wind_ms`: Wind speed (SMHI forecast)
- `rh`: Relative humidity (SMHI forecast)
- `lat`, `lon`: Measurement point coordinates

**Current Observation Features (latest <= forecast_run_time):**
- `obs_surface_temp_c`, `obs_surface_grip`
- `obs_air_temp_c`, `obs_air_rh`, `obs_dewpoint_c`
- `obs_surface_ice`, `obs_surface_snow`, `obs_surface_water`
- `obs_age_minutes`

**Temporal Features:**
- `hour`, `day_of_week`, `month`
- `is_rush_hour`, `is_weekend`

### Target Variable

Binary classification: **Hazard** (1) vs **Safe** (0) at `valid_time`

A road segment is classified as hazardous if any of:
- Surface temperature < 0°C (freezing)
- Ice detected
- Snow detected
- Grip coefficient < 0.5

### Algorithm

Per-horizon XGBoost ensembles with class imbalance handling (`scale_pos_weight`) and Platt calibration.
Uncertainty intervals (P10/P90) come from the ensemble distribution.

## Data Contract

Canonical identifier: `measurepoint_id` is the Trafikverket numeric ID (int) across observations, forecasts, labels, and predictions. `MP001`-style aliases are display-only.

### tv_weather_observation (v1)
- Primary keys: `measurepoint_id`, `sample_time`
- Event time: `sample_time`
- Core schema: `measurepoint_id`, `sample_time`, `surface_temp_c`, `surface_ice`, `surface_snow`, `surface_grip`, `air_temp_c`, `air_rh`, `dewpoint_c`, `ingested_at`

### smhi_point_forecast (v3)
- Primary keys: `measurepoint_id`, `forecast_run_time`, `valid_time`
- Event time: `valid_time`
- Core schema: `measurepoint_id` (int), `forecast_run_time` (SMHI approvedTime), `valid_time`, `t_air_c`, `precip_mm`, `wind_ms`, `rh`, `lat`, `lon`, `ingested_at`

### road_risk_predictions (v3)
- Primary keys: `measurepoint_id`, `forecast_run_time`, `valid_time`, `horizon_hours`
- Event time: `valid_time`
- Core schema: `measurepoint_id` (int), `forecast_run_time`, `valid_time`, `horizon_hours`, `risk_mean`, `risk_p10`, `risk_p90`, `hazard_predicted`, `recommendation`

### Quick dtype check

```python
import pandas as pd
from src.feature_store import get_project
from src.settings import settings

project = get_project(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="smhi_point_forecast", version=3)
df = fg.read()
print(df[["measurepoint_id"]].dtypes)
```

## Dashboard

The Streamlit dashboard provides:

- **Risk Map**: Interactive map showing current risk levels
- **Timeline**: 72-hour forecast with uncertainty bands (P10-P90)
- **Horizon Filter**: Toggle 24/48/72 hour forecasts
- **Recommendations**: Maintenance action suggestions
- **Metrics**: Summary statistics

Run locally:
```bash
uv run streamlit run src/dashboard.py
```

## Project Structure

```
road-risk-ml/
├── src/
│   ├── clients.py          # Trafikverket & SMHI API clients
│   ├── transforms.py       # Data extraction/transformation
│   ├── feature_store.py    # Hopsworks helper functions
│   ├── settings.py         # Configuration
│   ├── dashboard.py        # Streamlit dashboard
│   └── locations.json      # Measurement point definitions
├── scripts/
│   ├── backfill_feature_pipeline.py   # Historical data loading
│   ├── daily_feature_pipeline.py      # Daily feature updates
│   ├── training_pipeline.py           # Model training
│   └── batch_inference_pipeline.py    # Prediction generation
├── .github/workflows/
│   ├── daily-pipeline.yml    # Daily automation
│   └── weekly-training.yml   # Weekly model retraining
├── pyproject.toml
└── README.md
```

## GitHub Actions Secrets

Configure these secrets in your repository:

- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT`
- `TRAFIKVERKET_API_KEY`

## Development

### Running Tests

```bash
uv run pytest
```

### Code Style

```bash
uv run ruff check .
uv run ruff format .
```

## Future Enhancements

- [ ] Add more measurement points across Sweden
- [ ] Incorporate historical accident data
- [ ] Add elevation and road type features
- [ ] Implement real-time alerts/notifications
- [ ] Deploy dashboard to cloud (Streamlit Cloud, Heroku, etc.)
- [ ] Add model monitoring and drift detection

## License

MIT License

## Acknowledgments

- [Trafikverket](https://www.trafikverket.se/) for road weather data
- [SMHI](https://www.smhi.se/) for weather forecast data
- [Hopsworks](https://www.hopsworks.ai/) for feature store infrastructure
- KTH Royal Institute of Technology - Scalable Machine Learning course
