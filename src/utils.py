from __future__ import annotations

import pandas as pd


def to_datetime64ns_utc(series: pd.Series) -> pd.Series:
    """Normalize timestamps to tz-naive UTC datetime64[ns]."""
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    dt = dt.dt.tz_convert(None)
    return dt.astype("datetime64[ns]")
