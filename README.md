# Road Risk ML

A machine learning system for predicting hazardous road conditions on Stockholm roads, designed to help municipalities proactively plan road maintenance actions such as salting or snow-ploughing.

## Overview

This project implements a serverless ML pipeline that:
- Collects real-time road weather data from **Trafikverket** (Swedish Transport Administration)
- Integrates weather forecasts from **SMHI** (Swedish Meteorological and Hydrological Institute)
- Trains an XGBoost classifier to predict hazardous conditions (ice, snow, low grip)
- Generates 24-hour predictions and maintenance recommendations
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

| ID    | Name              | Road  | Coordinates           |
|-------|-------------------|-------|-----------------------|
| MP001 | E4 Norrtull       | E4    | 59.357°N, 18.050°E    |
| MP002 | E4 Häggvik        | E4    | 59.433°N, 17.933°E    |
| MP003 | E18 Jakobsberg    | E18   | 59.422°N, 17.833°E    |
| MP004 | E20 Essingeleden  | E20   | 59.327°N, 18.000°E    |
| MP005 | Rv73 Nynäsvägen   | Rv73  | 59.267°N, 18.083°E    |

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

# 2. Train the model
uv run python scripts/training_pipeline.py

# 3. Generate predictions
uv run python scripts/batch_inference_pipeline.py

# 4. Run the dashboard
uv run streamlit run src/dashboard.py
```

### Daily Operations

The system is designed to run automatically via GitHub Actions:
- **Daily (05:00 UTC)**: Feature pipeline + batch inference
- **Weekly (Sunday 02:00 UTC)**: Model retraining

## Model

### Features

**Numeric Features:**
- `surface_temp_c`: Road surface temperature
- `air_temp_c`: Air temperature
- `air_rh`: Relative humidity
- `dewpoint_c`: Dewpoint temperature
- `surface_grip`: Road grip coefficient (0-1)

**Temporal Features:**
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `is_rush_hour`: Rush hour indicator
- `is_weekend`: Weekend indicator

**Lagged Features:**
- Previous 1h, 3h, 6h values for temperature and grip

### Target Variable

Binary classification: **Hazard** (1) vs **Safe** (0)

A road segment is classified as hazardous if any of:
- Surface temperature < 0°C (freezing)
- Ice detected
- Snow detected
- Grip coefficient < 0.5

### Algorithm

XGBoost classifier with class imbalance handling (`scale_pos_weight`)

## Dashboard

The Streamlit dashboard provides:

- **Risk Map**: Interactive map showing current risk levels
- **Timeline**: 24-hour forecast for each road segment
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
