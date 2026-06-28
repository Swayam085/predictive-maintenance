# dashboard/app.py
# Author      : Vrushabh (Data Engineer)
# Branch      : feature/vrushabh-data
# Description : Streamlit dashboard — live predictive maintenance monitor
# Week 4 Status:
#   Day 23 — main dashboard, live feed, alert display
# Run        : streamlit run dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

sys.path.append("src")
from data_feed import load_sensor_data, get_single_reading, simulate_stream
from inference import preprocess_input, load_feature_cols
from alert_system import process_alert, get_alert_summary

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title = "Predictive Maintenance",
    page_icon  = "🔧",
    layout     = "wide"
)

# ── Title ──────────────────────────────────────────────
st.title("Predictive Maintenance Dashboard")
st.caption("IoT Edge AI ")
st.divider()

# ── Load data ──────────────────────────────────────────
@st.cache_data
def load_data():
    return load_sensor_data()

df = load_data()

# ── Sidebar ────────────────────────────────────────────
st.sidebar.header("Controls")
n_readings  = st.sidebar.slider("Number of readings", 5, 50, 10)
delay       = st.sidebar.slider("Delay between readings (sec)", 0.0, 2.0, 0.5)
random_seed = st.sidebar.number_input("Random seed", value=42)
run_button  = st.sidebar.button("Run Live Simulation", type="primary")
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {len(df):,} sensor readings")
st.sidebar.caption("Model: Simulated probability")

# ── Metric cards (top) ─────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", f"{len(df):,}")
col2.metric("Failure Rate",   "3.39%")
col3.metric("Features",       "18")
col4.metric("Tests Passing",  "27/27")

st.divider()

# ── Live simulation ────────────────────────────────────
if run_button:
    np.random.seed(int(random_seed))
    feature_cols = load_feature_cols()

    st.subheader("Live Sensor Stream")

    # Alert banner placeholder
    alert_banner = st.empty()

    # Metrics row
    m1, m2, m3 = st.columns(3)
    green_count  = m1.empty()
    yellow_count = m2.empty()
    red_count    = m3.empty()

    # Live table
    st.subheader("Reading History")
    table_placeholder = st.empty()

    # Progress bar
    progress = st.progress(0)

    # Results store karo
    results  = []
    g_cnt = y_cnt = r_cnt = 0

    for i in range(n_readings):
        reading      = get_single_reading(df)
        sensor_input = {k: v for k, v in reading.items()
                        if k not in ["idx", "actual_failure"]}

        # Probability simulate karo
        wear_ratio   = reading["Tool_wear_min"] / 250.0
        torque_ratio = reading["Torque_Nm"] / 80.0
        temp_ratio   = (reading["Process_temperature_K"] -
                        reading["Air_temperature_K"]) / 15.0
        prob = min(0.99, (wear_ratio * 0.5 +
                          torque_ratio * 0.3 +
                          temp_ratio * 0.2))

        pred_label = 1 if prob >= 0.30 else 0
        alert      = process_alert(
            sensor_input, prob, pred_label,
            save_history=False
        )

        level = alert["alert_level"]

        # Count update karo
        if level == "GREEN":  g_cnt += 1
        elif level == "YELLOW": y_cnt += 1
        else: r_cnt += 1

        # Alert banner
        if level == "GREEN":
            alert_banner.success(
                f"Reading {i+1}: {alert['message']}"
            )
        elif level == "YELLOW":
            alert_banner.warning(
                f"Reading {i+1}: {alert['message']}"
            )
        else:
            alert_banner.error(
                f"Reading {i+1}: {alert['message']}"
            )

        # Metrics update
        green_count.metric("GREEN (safe)", g_cnt)
        yellow_count.metric("YELLOW (warn)", y_cnt)
        red_count.metric("RED (critical)", r_cnt)

        # Table update
        results.append({
            "Reading" : i + 1,
            "Type"    : reading["Type"],
            "Torque"  : f"{reading['Torque_Nm']:.1f} Nm",
            "Tool wear": f"{reading['Tool_wear_min']} min",
            "Temp diff": f"{reading['Process_temperature_K'] - reading['Air_temperature_K']:.1f} K",
            "Prob"    : f"{prob:.4f}",
            "Alert"   : level,
            "Actual"  : reading["actual_failure"]
        })

        table_placeholder.dataframe(
            pd.DataFrame(results),
            use_container_width=True,
            hide_index=True
        )

        # Progress bar
        progress.progress((i + 1) / n_readings)

        time.sleep(delay)

    # ── Final summary ───────────────────────────────────
    st.divider()
    st.subheader("Session Summary")

    total    = len(results)
    accuracy = sum(
        1 for r in results
        if (r["Alert"] in ["YELLOW", "RED"]) == bool(r["Actual"])
    ) / total * 100

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total", total)
    s2.metric("GREEN", g_cnt)
    s3.metric("YELLOW", y_cnt)
    s4.metric("RED", r_cnt)

    st.info(f"Session accuracy: {accuracy:.1f}%")
    st.success("Simulation complete!")

else:
    # ── Default view ────────────────────────────────────
    st.subheader("Sensor Data Preview")
    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.subheader("Dataset Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure cases",    f"{df['Machine_failure'].sum()}")
    c2.metric("Normal cases",     f"{(df['Machine_failure']==0).sum()}")
    c3.metric("Failure rate",     "3.39%")

    st.info("Click 'Run Live Simulation' in sidebar to start!")