import hopsworks
import pandas as pd

def get_project(project: str, api_key: str, host_url: str):
    """Login to Hopsworks and return the project handle."""
    return hopsworks.login(project=project, api_key_value=api_key, host=host_url)

def get_fs(project: str, api_key: str, host_url: str):
    """Login to Hopsworks and return the feature store."""
    proj = get_project(project, api_key, host_url)
    return proj.get_feature_store()

def get_or_create_fg(fs, *, name, version, primary_key, event_time, description, online_enabled=False):
    return fs.get_or_create_feature_group(
        name=name,
        version=version,
        primary_key=list(primary_key),
        event_time=event_time,
        description=description,
        online_enabled=online_enabled,
    )

def insert_fg(fg, df: pd.DataFrame, *, dedup_keys=None, wait=True) -> int:
    if df is None or df.empty:
        return 0

    if dedup_keys:
        df = df.drop_duplicates(subset=list(dedup_keys), keep="last")

    # Support both insert APIs
    try:
        fg.insert(df, wait=wait)
    except TypeError:
        fg.insert(df, write_options={"wait_for_job": wait})

    return len(df)
