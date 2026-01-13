import hopsworks
import pandas as pd


def get_fg_feature_names(fg) -> list[str]:
    """Best-effort retrieval of feature names from a Hopsworks feature group."""
    if hasattr(fg, "get_feature_names"):
        try:
            names = fg.get_feature_names()
            return list(names) if names else []
        except Exception:
            pass

    for attr in ("features", "schema"):
        if hasattr(fg, attr):
            try:
                items = getattr(fg, attr) or []
                names = []
                for item in items:
                    if hasattr(item, "name"):
                        names.append(item.name)
                    else:
                        names.append(str(item))
                return names
            except Exception:
                continue

    if hasattr(fg, "feature_names"):
        try:
            return list(fg.feature_names)
        except Exception:
            pass

    return []


def align_df_to_fg_schema(
    fg,
    df: pd.DataFrame,
    *,
    fg_label: str | None = None,
    fallback_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Drop dataframe columns not present in the feature group schema."""
    if df is None or df.empty:
        return df

    feature_names = get_fg_feature_names(fg)
    if not feature_names:
        if fallback_columns:
            label = fg_label or getattr(fg, "name", "feature group")
            print(f"  WARN: Could not read schema for {label}; using fallback columns")
            feature_names = list(fallback_columns)
        else:
            label = fg_label or getattr(fg, "name", "feature group")
            print(f"  WARN: Could not read schema for {label}; skipping column drop")
            return df

    feature_set = set(feature_names)
    keep_cols = [c for c in df.columns if c in feature_set]
    dropped = [c for c in df.columns if c not in feature_set]
    if dropped:
        label = fg_label or getattr(fg, "name", "feature group")
        print(f"  Dropping columns not in {label} schema: {', '.join(dropped)}")
    return df[keep_cols]

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
