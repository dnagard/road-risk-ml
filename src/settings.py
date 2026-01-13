import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    hopsworks_host: str = os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    hopsworks_project: str = os.getenv("HOPSWORKS_PROJECT", "")
    hopsworks_api_key: str = os.getenv("HOPSWORKS_API_KEY", "")

    trafikverket_api_key: str = os.getenv("TRAFIKVERKET_API_KEY", "")
    trafikverket_url: str = os.getenv("TRAFIKVERKET_URL", "https://api.trafikinfo.trafikverket.se/v2/data.json")

    default_lat: float = float(os.getenv("DEFAULT_LAT", "59.3293"))
    default_lon: float = float(os.getenv("DEFAULT_LON", "18.0686"))

settings = Settings()
