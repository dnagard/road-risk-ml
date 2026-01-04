from src.settings import settings
from src.clients import TrafikverketClient, SMHIForecastClient
from src.transforms import (
    extract_weather_observations,
    extract_situations,
    extract_frostdepth_observations,
    extract_smhi_point_forecast,
)
from src.feature_store import get_fs, get_or_create_fg, insert_fg

def main():
    fs = get_fs(settings.hopsworks_project, settings.hopsworks_api_key, settings.hopsworks_host)

    tv = TrafikverketClient(settings.trafikverket_api_key, settings.trafikverket_url)
    smhi = SMHIForecastClient()

    # --- 1) Trafikverket WeatherObservation ---
    xml = tv.build_request("WeatherObservation", "2.1", limit=1000, filter_xml="")
    df_weather = extract_weather_observations(tv.post_xml(xml))

    fg_weather = get_or_create_fg(
        fs,
        name="tv_weather_observation",
        version=1,
        primary_key=["measurepoint_id", "sample_time"],
        event_time="sample_time",
        description="Trafikverket road weather observations (rolling retention).",
        online_enabled=False,
    )
    n_weather = insert_fg(fg_weather, df_weather, dedup_keys=["measurepoint_id", "sample_time"], wait=True)

    # --- 2) Trafikverket Situations ---
    xml = tv.build_request("Situation", "1.6", namespace="road.trafficinfo.new", limit=1000, filter_xml="")
    df_sit = extract_situations(tv.post_xml(xml))

    fg_sit = get_or_create_fg(
        fs,
        name="tv_situations",
        version=1,
        primary_key=["situation_id", "version_time"],
        event_time="version_time",
        description="Trafikverket situations (incidents/roadworks/etc).",
        online_enabled=False,
    )
    n_sit = insert_fg(fg_sit, df_sit, dedup_keys=["situation_id", "version_time"], wait=True)

    # --- 3) Trafikverket FrostDepthObservation ---
    xml = tv.build_request("FrostDepthObservation", "1.0", namespace="Road.WeatherInfo", limit=5000, filter_xml="")
    df_fd = extract_frostdepth_observations(tv.post_xml(xml))

    fg_fd = get_or_create_fg(
        fs,
        name="tv_frostdepth_observation",
        version=1,
        primary_key=["measurepoint_id", "sample_time", "depth_cm"],
        event_time="sample_time",
        description="Trafikverket frost depth observations.",
        online_enabled=False,
    )
    n_fd = insert_fg(
        fg_fd,
        df_fd,
        dedup_keys=["measurepoint_id", "sample_time", "depth_cm"],
        wait=True,
    )

    # --- 4) SMHI point forecast (baseline for Stockholm) ---
    payload_fc = smhi.get_point_forecast(settings.default_lat, settings.default_lon)
    df_fc = extract_smhi_point_forecast(payload_fc, settings.default_lat, settings.default_lon)

    fg_fc = get_or_create_fg(
        fs,
        name="smhi_point_forecast",
        version=1,
        primary_key=["lat", "lon", "valid_time"],
        event_time="valid_time",
        description="SMHI point forecast time series for reference location (baseline).",
        online_enabled=False,
    )
    n_fc = insert_fg(fg_fc, df_fc, dedup_keys=["lat", "lon", "valid_time"], wait=True)

    print(f"Inserted rows: weather={n_weather}, situations={n_sit}, frostdepth={n_fd}, forecast={n_fc}")

if __name__ == "__main__":
    main()
