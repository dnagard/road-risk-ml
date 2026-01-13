import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)

@dataclass(frozen=True)
class Settings:
    hopsworks_host: str = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    hopsworks_project: str = os.getenv("HOPSWORKS_PROJECT", "")
    hopsworks_api_key: str = os.getenv("HOPSWORKS_API_KEY", "")

    trafikverket_api_key: str = os.getenv("TRAFIKVERKET_API_KEY", "")
    trafikverket_url: str = os.getenv("TRAFIKVERKET_URL", "https://api.trafikinfo.trafikverket.se/v2/data.json")

    default_lat: float = _env_float("DEFAULT_LAT", 59.3293)
    default_lon: float = _env_float("DEFAULT_LON", 18.0686)

settings = Settings()
