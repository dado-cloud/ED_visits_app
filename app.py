import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.load_model import load_tft_model, load_training_dataset
from src.predict import predict_daily_forecast
from src.preprocess import load_input_data, prepare_future_dataframe
from src.risk_rules import (
    calculate_risk_level,
    generate_alerts,
    get_peak_periods_table,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="ED Operations Dashboard",
    page_icon="🏥",
    layout="wide",
)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH = MODEL_DIR / "tft_model.ckpt"
TRAINING_DATASET_PATH = MODEL_DIR / "dataset.pkl"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_tft_model(MODEL_PATH)
    training_dataset = load_training_dataset(TRAINING_DATASET_PATH)

    feature_config = {}
    if FEATURE_CONFIG_PATH.exists():
        with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
            feature_config = json.load(f)

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, training_dataset, feature_config, metrics


def monthly_aggregate(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("month", as_index=False)["predicted_visits"]
        .sum()
        .rename(columns={"predicted_visits": "monthly_predicted_visits"})
    )
    return monthly


def style_metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            min-height: 120px;
        ">
            <div style="font-size: 16px; color: #6b7280; margin-bottom: 8px;">{title}</div>
            <div style="font-size: 32px; font-weight: 700; color: #111827;">{value}</div>
            <div style="font-size: 14px; color: #2563eb; margin-top: 8px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_risk_badge(risk_level: str) -> str:
    colors = {
        "Low": "#dcfce7",
        "Moderate": "#fef3c7",
        "High": "#fee2e2",
    }
    text_colors = {
        "Low": "#166534",
        "Moderate": "#92400e",
        "High": "#991b1b",
    }
    bg = colors.get(risk_level, "#e5e7eb")
    fg = text_colors.get(risk_level, "#374151")

    return f"""
    <div style="
        display:inline-block;
        padding:8px 14px;
        border-radius:999px;
        background-color:{bg};
        color:{fg};
        font-weight:600;
        font-size:15px;
    ">
        {risk_level}
    </div>
    """


def render_alerts(alerts: list[str]):
    st.subheader("Alerts")
    if not alerts:
        st.success("No critical alerts at the moment.")
    else:
        for alert in alerts:
            st.warning(alert)


def render_daily_chart(df: pd.DataFrame):
    fig = px.line(
        df,
        x="date",
        y="predicted_visits",
        markers=True,
        title="Daily Forecast",
        labels={"date": "Date", "predicted_visits": "Expected ED Visits"},
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Expected ED Visits",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_monthly_chart(df: pd.DataFrame):
    fig = px.bar(
        df,
        x="month",
        y="monthly_predicted_visits",
        title="Monthly Forecast",
        labels={
            "month": "Month",
            "monthly_predicted_visits": "Expected ED Visits"
        },
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Expected ED Visits",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Title
# -----------------------------
st.markdown("## ED Operations Dashboard")
st.caption("Interactive forecasting dashboard for Emergency Department visit demand.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Forecast Settings")

forecast_view = st.sidebar.radio(
    "Select forecast horizon",
    ["Daily", "Monthly", "Hourly"],
    index=0,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload input CSV",
    type=["csv"],
    help="Upload the latest prepared dataset used for inference.",
)

default_horizon = st.sidebar.slider(
    "Forecast length (days)",
    min_value=7,
    max_value=60,
    value=30,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.info("Config 5 is currently used for production forecasting.")

# -----------------------------
# Load model artifacts
# -----------------------------
try:
    model, training_dataset, feature_config, metrics = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# -----------------------------
# Load input data
# -----------------------------
try:
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = load_input_data()
except Exception as e:
    st.error(f"Failed to load input data: {e}")
    st.stop()

# -----------------------------
# Prepare inference data
# -----------------------------
try:
    future_df = prepare_future_dataframe(
        input_df=input_df,
        forecast_days=default_horizon,
        feature_config=feature_config,
    )

    daily_forecast = predict_daily_forecast(
        model=model,
        training_dataset=training_dataset,
        future_df=future_df,
    )
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Expected format of daily_forecast:
# columns = ["date", "predicted_visits"]

if daily_forecast.empty:
    st.warning("No forecast was generated.")
    st.stop()

# -----------------------------
# Derived summaries
# -----------------------------
today_prediction = int(round(daily_forecast.iloc[0]["predicted_visits"]))
total_period_prediction = int(round(daily_forecast["predicted_visits"].sum()))
avg_prediction = float(daily_forecast["predicted_visits"].mean())

risk_level = calculate_risk_level(avg_prediction)
alerts = generate_alerts(daily_forecast)
peak_periods_df = get_peak_periods_table(daily_forecast)

# simple baseline comparison
historical_mean = input_df["ED_visits"].mean() if "ED_visits" in input_df.columns else None
if historical_mean and historical_mean > 0:
    delta_pct = ((today_prediction - historical_mean) / historical_mean) * 100
    comparison_text = f"{delta_pct:+.1f}% vs historical average"
else:
    comparison_text = "Baseline comparison unavailable"

# -----------------------------
# Top KPI row
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    style_metric_card(
        "Next Forecasted Day",
        str(today_prediction),
        comparison_text,
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            min-height: 120px;
        ">
            <div style="font-size: 16px; color: #6b7280; margin-bottom: 8px;">Current Risk Level</div>
            <div style="margin-top: 12px;">{style_risk_badge(risk_level)}</div>
            <div style="font-size: 14px; color: #6b7280; margin-top: 14px;">
                Based on predicted demand level.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    style_metric_card(
        "Forecast Period Total",
        str(total_period_prediction),
        f"Average: {avg_prediction:.1f} visits/day",
    )

st.markdown("---")

# -----------------------------
# Main content
# -----------------------------
left_col, right_col = st.columns([2, 1])

with left_col:
    if forecast_view == "Daily":
        st.subheader("Daily Forecast")
        render_daily_chart(daily_forecast)

    elif forecast_view == "Monthly":
        st.subheader("Monthly Forecast")
        monthly_df = monthly_aggregate(daily_forecast)
        render_monthly_chart(monthly_df)

    else:
        st.subheader("Hourly Forecast")
        st.info("Hourly forecasting will be added later using a separate model within the same website.")

with right_col:
    render_alerts(alerts)

    st.subheader("Upcoming Peak Periods")
    if peak_periods_df is not None and not peak_periods_df.empty:
        st.dataframe(peak_periods_df, use_container_width=True, hide_index=True)
    else:
        st.write("No peak periods identified.")

# -----------------------------
# Model info
# -----------------------------
with st.expander("Model Information"):
    st.write("Production model: TFT Config 5")

    if metrics:
        metric_cols = st.columns(3)
        metric_cols[0].metric("MAE", metrics.get("mae", "N/A"))
        metric_cols[1].metric("RMSE", metrics.get("rmse", "N/A"))
        metric_cols[2].metric("MAPE", metrics.get("mape", "N/A"))

    if feature_config:
        st.json(feature_config)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("ED forecasting prototype for operational planning and demand monitoring.")
