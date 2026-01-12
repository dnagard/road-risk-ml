"""Road Risk ML - Streamlit Dashboard"""
from datetime import datetime, date, timedelta
import hopsworks
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    hopsworks_api_key: str
    hopsworks_project: str = ""
    hopsworks_host: str = "c.app.hopsworks.ai"
    model_config = SettingsConfigDict(env_file=".env", extra='ignore')

STOCKHOLM_MEASUREPOINTS = {
    "MP001": {"name": "E4 Norrtull", "lat": 59.357, "lon": 18.05},
    "MP002": {"name": "E4 Häggvik", "lat": 59.433, "lon": 17.933},
    "MP003": {"name": "E18 Jakobsberg", "lat": 59.422, "lon": 17.833},
    "MP004": {"name": "E20 Essingeleden", "lat": 59.327, "lon": 18.0},
    "MP005": {"name": "Rv73 Nynäsvägen", "lat": 59.267, "lon": 18.083},
}

def get_risk_color(risk): return "#FF0000" if risk > 0.7 else "#FFA500" if risk > 0.4 else "#00CC00"
def get_risk_level(risk): return "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"

@st.cache_data(ttl=300)
def load_predictions():
    try:
        settings = Settings()
        project = hopsworks.login(api_key_value=settings.hopsworks_api_key, host=settings.hopsworks_host,
                                   project=settings.hopsworks_project if settings.hopsworks_project else None)
        fs = project.get_feature_store()
        pred_fg = fs.get_feature_group(name="road_risk_predictions", version=1)
        return pred_fg.filter(pred_fg.forecast_date >= date.today().strftime("%Y-%m-%d")).read()
    except:
        return create_demo_predictions()

def create_demo_predictions():
    np.random.seed(42)
    rows = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    for mp_id, mp_info in STOCKHOLM_MEASUREPOINTS.items():
        for h in range(24):
            valid_time = now + timedelta(hours=h)
            base_risk = 0.3 + (0.3 if valid_time.hour in [0,1,2,3,4,5,6] else 0)
            risk = np.clip(base_risk + np.random.normal(0, 0.15), 0, 1)
            temp = -2 + np.sin(valid_time.hour * np.pi / 12) * 4
            rec = "URGENT: Pre-salt" if risk > 0.7 else "MODERATE: Standby" if risk > 0.4 else "LOW: Normal"
            rows.append({'measurepoint_id': mp_id, 'valid_time': valid_time, 'risk_probability': risk,
                        'hazard_predicted': int(risk > 0.5), 'recommendation': rec,
                        'forecast_date': date.today(), 'surface_temp_c': temp, 'air_temp_c': temp + 2})
    return pd.DataFrame(rows)

def create_risk_map(pred_df, selected_hour):
    pred_df['hour'] = pd.to_datetime(pred_df['valid_time']).dt.hour
    hourly_df = pred_df[pred_df['hour'] == selected_hour]
    if hourly_df.empty: hourly_df = pred_df.head(len(STOCKHOLM_MEASUREPOINTS))
    
    fig = go.Figure()
    for _, row in hourly_df.iterrows():
        mp_id = row['measurepoint_id']
        if mp_id in STOCKHOLM_MEASUREPOINTS:
            mp = STOCKHOLM_MEASUREPOINTS[mp_id]
            fig.add_trace(go.Scattermapbox(
                lat=[mp['lat']], lon=[mp['lon']], mode='markers+text',
                marker=dict(size=30, color=get_risk_color(row['risk_probability']), opacity=0.8),
                text=mp['name'], textposition="top center",
                hovertemplate=f"<b>{mp['name']}</b><br>Risk: {row['risk_probability']:.0%}<br>Temp: {row.get('surface_temp_c', 0):.1f}°C<extra></extra>",
                showlegend=False))
    
    fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=59.33, lon=18.07), zoom=10),
                      margin=dict(r=0, t=0, l=0, b=0), height=500)
    return fig

def create_timeline(pred_df, mp_id):
    mp_df = pred_df[pred_df['measurepoint_id'] == mp_id].sort_values('valid_time')
    if mp_df.empty: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mp_df['valid_time'], y=mp_df['risk_probability'], mode='lines+markers',
                             name='Risk', line=dict(color='#FF6B6B', width=3), fill='tozeroy'))
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="High")
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Medium")
    fig.update_layout(title=f"24h Forecast: {STOCKHOLM_MEASUREPOINTS.get(mp_id, {}).get('name', mp_id)}",
                      yaxis=dict(title="Risk", range=[0, 1]), height=400)
    return fig

def main():
    st.set_page_config(page_title="Road Risk Dashboard", page_icon="🚗", layout="wide")
    st.title("🚗 Road Risk Prediction - Stockholm")
    st.markdown("Predicted hazardous road conditions for the next 24 hours")
    
    with st.spinner("Loading..."): pred_df = load_predictions()
    if pred_df is None or pred_df.empty:
        st.error("No data. Run batch_inference_pipeline.py first.")
        return
    
    now = datetime.now()
    selected_hour = st.sidebar.selectbox("Hour", range(24), index=now.hour,
        format_func=lambda x: (now.replace(hour=0) + timedelta(hours=x)).strftime("%H:00"))
    
    current = pred_df[pd.to_datetime(pred_df['valid_time']).dt.hour == selected_hour]
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High Risk", len(current[current['risk_probability'] > 0.7]))
    col2.metric("📊 Avg Risk", f"{current['risk_probability'].mean()*100:.1f}%")
    col3.metric("🌡️ Avg Temp", f"{current['surface_temp_c'].mean():.1f}°C")
    
    st.subheader("🗺️ Risk Map")
    st.plotly_chart(create_risk_map(pred_df, selected_hour), use_container_width=True)
    
    st.subheader("📋 Recommendations")
    for _, row in current.sort_values('risk_probability', ascending=False).iterrows():
        mp_name = STOCKHOLM_MEASUREPOINTS.get(row['measurepoint_id'], {}).get('name', row['measurepoint_id'])
        color = get_risk_color(row['risk_probability'])
        st.markdown(f"**{mp_name}** | <span style='color:{color}'>{get_risk_level(row['risk_probability'])} ({row['risk_probability']:.0%})</span> | _{row.get('recommendation', 'N/A')}_", unsafe_allow_html=True)
    
    st.subheader("📈 Timeline")
    mp = st.selectbox("Segment", list(STOCKHOLM_MEASUREPOINTS.keys()),
                      format_func=lambda x: STOCKHOLM_MEASUREPOINTS[x]['name'])
    fig = create_timeline(pred_df, mp)
    if fig: st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()