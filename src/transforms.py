from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime, timezone
import pandas as pd

def _utc_now_naive() -> pd.Timestamp:
    # store as tz-naive UTC timestamp (often easiest for feature stores)
    return pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)

def _to_ts(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.tz_convert(None)

def extract_weather_observations(payload: Dict[str, Any]) -> pd.DataFrame:
    results = payload.get("RESPONSE", {}).get("RESULT", [])
    rows: List[dict] = []

    for res in results:
        for obs in res.get("WeatherObservation", []) or []:
            mp = obs.get("Measurepoint", {}) or {}
            geom = mp.get("Geometry", {}) or {}
            wgs84 = geom.get("WGS84")

            surface = obs.get("Surface") or {}
            air = obs.get("Air") or {}

            rows.append({
                "measurepoint_id": mp.get("Id"),
                "measurepoint_name": mp.get("Name"),
                "wgs84": wgs84,
                "sample_time": obs.get("Sample"),
                "surface_temp_c": (surface.get("Temperature") or {}).get("Value"),
                "surface_water": surface.get("Water"),
                "surface_ice": surface.get("Ice"),
                "surface_snow": surface.get("Snow"),
                "surface_grip": (surface.get("Grip") or {}).get("Value"),
                "air_temp_c": (air.get("Temperature") or {}).get("Value"),
                "air_rh": (air.get("RelativeHumidity") or {}).get("Value"),
                "dewpoint_c": (air.get("Dewpoint") or {}).get("Value"),
                "ingested_at": _utc_now_naive(),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["sample_time"] = _to_ts(df["sample_time"])
        df["measurepoint_id"] = pd.to_numeric(df["measurepoint_id"], errors="coerce").astype("Int64")
    return df

def extract_situations(payload: Dict[str, Any]) -> pd.DataFrame:
    results = payload.get("RESPONSE", {}).get("RESULT", [])
    rows: List[dict] = []

    for res in results:
        for s in res.get("Situation", []) or []:
            for dev in s.get("Deviation", []) or []:
                geom = dev.get("Geometry", {}) or {}
                point = geom.get("Point", {}) or {}

                rows.append({
                    "situation_id": s.get("Id"),
                    "publication_time": s.get("PublicationTime"),
                    "version_time": s.get("VersionTime"),
                    "message_type": dev.get("MessageType"),
                    "message_type_value": dev.get("MessageTypeValue"),
                    "message_code": dev.get("MessageCode"),
                    "message_code_value": dev.get("MessageCodeValue"),
                    "severity_code": dev.get("SeverityCode"),
                    "start_time": dev.get("StartTime"),
                    "end_time": dev.get("EndTime"),
                    "suspended": dev.get("Suspended"),
                    "wgs84": point.get("WGS84"),
                    "ingested_at": _utc_now_naive(),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        for col in ["publication_time", "version_time", "start_time", "end_time"]:
            if col in df.columns:
                df[col] = _to_ts(df[col])
    return df

def extract_frostdepth_observations(payload: Dict[str, Any]) -> pd.DataFrame:
    results = payload.get("RESPONSE", {}).get("RESULT", [])
    rows: List[dict] = []

    for res in results:
        for o in res.get("FrostDepthObservation", []) or []:
            mp = o.get("Measurepoint", {}) or {}
            geom = mp.get("Geometry", {}) or {}
            wgs84 = geom.get("WGS84")

            fd = o.get("FrostDepth", {}) or {}
            sample = fd.get("Sample")

            for t in fd.get("Temperature", []) or []:
                rows.append({
                    "measurepoint_id": mp.get("Id"),
                    "measurepoint_name": mp.get("Name"),
                    "wgs84": wgs84,
                    "sample_time": sample,
                    "sensor_name": t.get("SensorName"),
                    "depth_cm": t.get("Depth"),
                    "temp_c": t.get("Value"),
                    "ingested_at": _utc_now_naive(),
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["sample_time"] = _to_ts(df["sample_time"])
        df["measurepoint_id"] = pd.to_numeric(df["measurepoint_id"], errors="coerce").astype("Int64")
    return df

def extract_smhi_point_forecast(
    payload: Dict[str, Any],
    lat: float,
    lon: float,
    measurepoint_id: int | str | None = None,
) -> pd.DataFrame:
    run_time_raw = payload.get("approvedTime") or payload.get("referenceTime")
    run_time = pd.to_datetime(run_time_raw, utc=True, errors="coerce")
    if pd.isna(run_time):
        forecast_run_time = _utc_now_naive()
    else:
        forecast_run_time = run_time.tz_convert(None)

    if measurepoint_id is not None:
        try:
            measurepoint_id = int(measurepoint_id)
        except (TypeError, ValueError):
            pass

    rows: List[dict] = []
    for ts in payload.get("timeSeries", []) or []:
        params = {p["name"]: (p.get("values") or [None])[0] for p in ts.get("parameters", [])}
        rows.append({
            "measurepoint_id": measurepoint_id,
            "lat": lat,
            "lon": lon,
            "forecast_run_time": forecast_run_time,
            "valid_time": ts.get("validTime"),
            "t_air_c": params.get("t"),
            "precip_mm": params.get("pmean"),
            "wind_ms": params.get("ws"),
            "rh": params.get("r"),
            "ingested_at": _utc_now_naive(),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["valid_time"] = _to_ts(df["valid_time"])
        df["measurepoint_id"] = pd.to_numeric(df["measurepoint_id"], errors="coerce").astype("Int64")
    return df
