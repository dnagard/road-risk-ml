"""Road Risk ML - Streamlit Dashboard"""
from datetime import datetime, timedelta
from pathlib import Path
import json
import hopsworks
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hopsworks_api_key: str = ""
    hopsworks_project: str = ""
    hopsworks_host: str = "c.app.hopsworks.ai"
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')


LOCATIONS_FILE = Path(__file__).parent / "locations.json"


def load_locations() -> dict:
    try:
        with open(LOCATIONS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def build_measurepoint_map() -> dict:
    locations = load_locations()
    points = {}
    for info in locations.values():
        tv_id = info.get("tv_measurepoint_id")
        if tv_id is None:
            continue
        points[int(tv_id)] = {
            "name": info.get("name", str(tv_id)),
            "lat": info.get("latitude"),
            "lon": info.get("longitude"),
            "alias": info.get("id"),
        }

    if points:
        return points

    return {
        243: {"name": "E4 Norrtull", "lat": 59.357, "lon": 18.05, "alias": "MP001"},
        226: {"name": "E4 Häggvik", "lat": 59.433, "lon": 17.933, "alias": "MP002"},
        232: {"name": "E18 Jakobsberg", "lat": 59.422, "lon": 17.833, "alias": "MP003"},
        237: {"name": "E20 Essingeleden", "lat": 59.327, "lon": 18.0, "alias": "MP004"},
        215: {"name": "Rv73 Nynäsvägen", "lat": 59.267, "lon": 18.083, "alias": "MP005"},
    }


MEASUREPOINTS = build_measurepoint_map()
ALIAS_TO_TV_ID = {v.get("alias"): k for k, v in MEASUREPOINTS.items() if v.get("alias")}


def get_risk_color(risk):
    return "#FF0000" if risk > 0.7 else "#FFA500" if risk > 0.4 else "#00CC00"


def get_risk_level(risk):
    return "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"


@st.cache_data(ttl=300)
def load_predictions():
    settings = Settings()
    if not settings.hopsworks_api_key:
        return create_demo_predictions()

    try:
        project = hopsworks.login(
            api_key_value=settings.hopsworks_api_key,
            host=settings.hopsworks_host,
            project=settings.hopsworks_project if settings.hopsworks_project else None,
        )
        fs = project.get_feature_store()
        pred_fg = fs.get_feature_group(name="road_risk_predictions", version=3)
        df = pred_fg.read()
    except Exception:
        return create_demo_predictions()

    if df.empty:
        return create_demo_predictions()

    if "forecast_run_time" in df.columns:
        df["forecast_run_time"] = pd.to_datetime(df["forecast_run_time"], errors="coerce")
        latest_run = df["forecast_run_time"].max()
        if pd.notna(latest_run):
            df = df[df["forecast_run_time"] == latest_run]

    if "measurepoint_id" in df.columns:
        df["measurepoint_id"] = df["measurepoint_id"].map(ALIAS_TO_TV_ID).fillna(df["measurepoint_id"])
        df["measurepoint_id"] = pd.to_numeric(df["measurepoint_id"], errors="coerce")

    df["valid_time"] = pd.to_datetime(df["valid_time"], errors="coerce")
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    df = df[(df["valid_time"] >= now) & (df["valid_time"] <= now + timedelta(hours=72))]

    if df.empty:
        return create_demo_predictions()

    if "risk_mean" not in df.columns and "risk_probability" in df.columns:
        df["risk_mean"] = df["risk_probability"]
    if "risk_p10" not in df.columns:
        df["risk_p10"] = np.nan
    if "risk_p90" not in df.columns:
        df["risk_p90"] = np.nan

    return df


def create_demo_predictions():
    np.random.seed(42)
    rows = []
    run_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    for mp_id, mp_info in MEASUREPOINTS.items():
        for horizon in [24, 48, 72]:
            valid_time = run_time + timedelta(hours=horizon)
            base_risk = 0.35 + (0.2 if valid_time.hour in [0, 1, 2, 3, 4, 5, 6] else 0)
            risk = np.clip(base_risk + np.random.normal(0, 0.12), 0, 1)
            spread = np.clip(np.random.normal(0.15, 0.05), 0.05, 0.3)
            rows.append(
                {
                    "measurepoint_id": mp_id,
                    "valid_time": valid_time,
                    "forecast_run_time": run_time,
                    "horizon_hours": horizon,
                    "risk_mean": risk,
                    "risk_p10": max(0, risk - spread),
                    "risk_p90": min(1, risk + spread),
                    "hazard_predicted": int(risk > 0.5),
                    "recommendation": "LOW: Normal" if risk < 0.4 else "MODERATE: Standby",
                }
            )
    return pd.DataFrame(rows)


def create_risk_map(pred_df):
    fig = go.Figure()
    for _, row in pred_df.iterrows():
        mp_id = row["measurepoint_id"]
        if mp_id in MEASUREPOINTS:
            mp = MEASUREPOINTS[mp_id]
            fig.add_trace(
                go.Scattermapbox(
                    lat=[mp["lat"]],
                    lon=[mp["lon"]],
                    mode="markers+text",
                    marker=dict(size=30, color=get_risk_color(row["risk_mean"]), opacity=0.8),
                    text=mp["name"],
                    textposition="top center",
                    hovertemplate=(
                        f"<b>{mp['name']}</b><br>Risk: {row['risk_mean']:.0%}"
                        f"<br>P10-P90: {row.get('risk_p10', np.nan):.0%} - {row.get('risk_p90', np.nan):.0%}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=59.33, lon=18.07), zoom=10),
        margin=dict(r=0, t=0, l=0, b=0),
        height=500,
    )
    return fig


def create_timeline(pred_df, mp_id):
    mp_df = pred_df[pred_df["measurepoint_id"] == mp_id].sort_values("valid_time")
    if mp_df.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mp_df["valid_time"],
            y=mp_df["risk_mean"],
            mode="lines+markers",
            name="Risk Mean",
            line=dict(color="#FF6B6B", width=3),
        )
    )

    if mp_df["risk_p10"].notna().any() and mp_df["risk_p90"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pd.concat([mp_df["valid_time"], mp_df["valid_time"][::-1]]),
                y=pd.concat([mp_df["risk_p90"], mp_df["risk_p10"][::-1]]),
                fill="toself",
                fillcolor="rgba(255, 107, 107, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="P10-P90",
            )
        )

    fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Medium")
    fig.update_layout(
        title=f"72h Forecast: {MEASUREPOINTS.get(mp_id, {}).get('name', mp_id)}",
        yaxis=dict(title="Risk", range=[0, 1]),
        height=400,
    )
    return fig


def main():
    st.set_page_config(page_title="Road Risk Dashboard", page_icon="🚗", layout="wide")
    st.title("🚗 Road Risk Forecast - Stockholm")
    st.markdown("Forecasted hazardous road conditions for 24/48/72 hour horizons")

    with st.spinner("Loading..."):
        pred_df = load_predictions()
    if pred_df is None or pred_df.empty:
        st.error("No data. Run batch_inference_pipeline.py first.")
        return

    horizon_options = ["All", 24, 48, 72]
    selected_horizon = st.sidebar.selectbox("Horizon", horizon_options, index=0)

    filtered_df = pred_df.copy()
    if selected_horizon != "All" and "horizon_hours" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["horizon_hours"] == selected_horizon]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk", len(filtered_df[filtered_df["risk_mean"] > 0.7]))
    col2.metric("📊 Avg Risk", f"{filtered_df['risk_mean'].mean() * 100:.1f}%")
    col3.metric("Points", filtered_df["measurepoint_id"].nunique())

    st.subheader("🗺️ Risk Map")
    st.plotly_chart(create_risk_map(filtered_df), use_container_width=True)

    st.subheader("📋 Recommendations")
    for _, row in filtered_df.sort_values("risk_mean", ascending=False).iterrows():
        mp_name = MEASUREPOINTS.get(row["measurepoint_id"], {}).get("name", row["measurepoint_id"])
        color = get_risk_color(row["risk_mean"])
        st.markdown(
            f"**{mp_name}** | <span style='color:{color}'>"
            f"{get_risk_level(row['risk_mean'])} ({row['risk_mean']:.0%})</span> | "
            f"_{row.get('recommendation', 'N/A')}_",
            unsafe_allow_html=True,
        )

    st.subheader("📈 Timeline")
    mp = st.selectbox(
        "Segment",
        list(MEASUREPOINTS.keys()),
        format_func=lambda x: MEASUREPOINTS[x]["name"],
    )
    fig = create_timeline(filtered_df, mp)
    if fig:
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
