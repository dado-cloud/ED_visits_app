import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="ED Operations Dashboard", layout="wide")

# Custom CSS to mimic the prototype's clean look
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header Section ---
st.title("🏥 ED Operations Dashboard")
st.caption("Near-real-time forecast of ED visits, overcrowding risk, and key drivers.")

# --- Forecast Selection Bar ---
# Replicating your [Hourly, Daily, Weekly, Monthly] buttons
col_btn1, col_btn2, col_btn3, col_btn4, _ = st.columns([1,1,1,1,4])
with col_btn1:
    st.button("Hourly", type="primary") # Highlighting hourly as requested
with col_btn2:
    st.button("Daily")
with col_btn3:
    st.button("Weekly")
with col_btn4:
    st.button("Monthly")

st.divider()

# --- Top Metrics Row ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Today's Forecast")
    st.metric(label="Total expected visits today", value="212", delta="+8% vs typical Tuesday")

with col2:
    st.subheader("Current risk level")
    st.warning("Moderate Overcrowding Risk")
    st.write("Peak expected between 18:00–21:00")

with col3:
    st.subheader("Next hour (forecast)")
    st.error("24")
    st.caption("High inflow")

# --- Main Content Area ---
main_col, side_col = st.columns([2, 1])

with main_col:
    st.write("### Hourly forecast (next 24 hours)")
    
    # Generate dummy data for the plot to match your prototype curve
    chart_data = pd.DataFrame({
        'Hour': pd.date_range("2026-04-20 23:00", periods=24, freq="H"),
        'Visits': [18, 15, 12, 10, 8, 9, 11, 15, 20, 24, 26, 28, 25, 23, 22, 24, 28, 30, 28, 24, 21, 20, 19, 16]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['Hour'], y=chart_data['Visits'], 
                             mode='lines+markers', line=dict(color='#3b82f6', width=3),
                             fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=350, 
                      xaxis_title="Time", yaxis_title="Expected Visits per Hour")
    st.plotly_chart(fig, use_container_width=True)

with side_col:
    # Alerts Section
    st.write("### Alerts")
    st.info("⚠️ **Alert:** Predicted demand exceeds 120% of normal between 19:00–21:00.")
    
    # Upcoming Peak Periods Table
    st.write("### Upcoming Peak Periods")
    peak_data = pd.DataFrame({
        "Time Window": ["16:00–17:00", "18:00–19:00", "19:00–20:00"],
        "Expected": [21, 27, 30],
        "Risk": ["Normal", "Elevated", "High"]
    })
    st.table(peak_data)
