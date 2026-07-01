# dashboard/app.py
# Author      : Vrushabh (Data Engineer) + Swayam Arya (ML)
# Branch      : feature/vrushabh-data + feature/swayam-ml
# Description : Streamlit dashboard — live predictive maintenance monitor
# Week 4 Status:
#   Day 23 — main dashboard, live feed, alert display
#   Day 24 — ML Analysis tabs added (Swayam)
#   Day 25 — Real ONNX model connected (Swayam)

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import json

sys.path.append("src")
from data_feed import load_sensor_data, get_single_reading, simulate_stream
from inference import preprocess_input, load_feature_cols
from alert_system import process_alert, get_alert_summary

# ── ONNX Model Load (ek baar) ──────────────────────────
import onnxruntime as ort

FINAL_MODEL_PATH = "models/final_model.onnx"
EXT_DEFAULTS_PATH = "models/ext_feature_defaults.json"

FEATURE_ORDER = [
    'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm',
    'Torque_Nm', 'Tool_wear_min', 'ambient_temp', 'load_density',
    'humidity', 'shift_encoded', 'temp_load_stress', 'humidity_wear', 'night_load'
]

@st.cache_resource
def load_onnx():
    session = ort.InferenceSession(FINAL_MODEL_PATH)
    return session

@st.cache_data
def load_ext_defaults():
    with open(EXT_DEFAULTS_PATH, 'r') as f:
        return json.load(f)

onnx_session   = load_onnx()
onnx_input_name = onnx_session.get_inputs()[0].name
ext_defaults   = load_ext_defaults()

def get_real_prob(reading: dict) -> float:
    # 5 raw sensor values + 7 external defaults se 12-feature vector banao
    features = {
        'Air_temperature_K'     : reading['Air_temperature_K'],
        'Process_temperature_K' : reading['Process_temperature_K'],
        'Rotational_speed_rpm'  : reading['Rotational_speed_rpm'],
        'Torque_Nm'             : reading['Torque_Nm'],
        'Tool_wear_min'         : reading['Tool_wear_min'],
        'ambient_temp'          : ext_defaults['ambient_temp'],
        'load_density'          : ext_defaults['load_density'],
        'humidity'              : ext_defaults['humidity'],
        'shift_encoded'         : ext_defaults['shift_encoded'],
        'temp_load_stress'      : ext_defaults['temp_load_stress'],
        'humidity_wear'         : ext_defaults['humidity_wear'],
        'night_load'            : ext_defaults['night_load'],
    }
    X = np.array([[features[col] for col in FEATURE_ORDER]], dtype=np.float32)
    result = onnx_session.run(None, {onnx_input_name: X})
    prob = float(result[1][0][1])  # class 1 (failure) probability
    return prob

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title = "Predictive Maintenance",
    page_icon  = "🔧",
    layout     = "wide"
)

# ── Title ──────────────────────────────────────────────
st.title("Predictive Maintenance Dashboard")
st.caption("IoT Edge AI")
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
st.sidebar.caption("Model: LightGBM (ONNX) — Real predictions")

# ── Metric cards (top) ─────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", f"{len(df):,}")
col2.metric("Failure Rate",   "3.39%")
col3.metric("Features",       "12")
col4.metric("Tests Passing",  "27/27")

st.divider()

# ── Live simulation ────────────────────────────────────
if run_button:
    np.random.seed(int(random_seed))
    feature_cols = load_feature_cols()

    st.subheader("Live Sensor Stream")

    alert_banner = st.empty()

    m1, m2, m3 = st.columns(3)
    green_count  = m1.empty()
    yellow_count = m2.empty()
    red_count    = m3.empty()

    st.subheader("Reading History")
    table_placeholder = st.empty()

    progress = st.progress(0)

    results  = []
    g_cnt = y_cnt = r_cnt = 0

    for i in range(n_readings):
        reading      = get_single_reading(df)
        sensor_input = {k: v for k, v in reading.items()
                        if k not in ["idx", "actual_failure"]}

        # Real ONNX model se probability
        prob = get_real_prob(reading)

        pred_label = 1 if prob >= 0.30 else 0
        alert      = process_alert(
            sensor_input, prob, pred_label,
            save_history=False
        )

        level = alert["alert_level"]

        if level == "GREEN":  g_cnt += 1
        elif level == "YELLOW": y_cnt += 1
        else: r_cnt += 1

        if level == "GREEN":
            alert_banner.success(f"Reading {i+1}: {alert['message']}")
        elif level == "YELLOW":
            alert_banner.warning(f"Reading {i+1}: {alert['message']}")
        else:
            alert_banner.error(f"Reading {i+1}: {alert['message']}")

        green_count.metric("GREEN (safe)", g_cnt)
        yellow_count.metric("YELLOW (warn)", y_cnt)
        red_count.metric("RED (critical)", r_cnt)

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
            width=700,
            hide_index=True
        )

        progress.progress((i + 1) / n_readings)
        time.sleep(delay)

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
    st.subheader("Sensor Data Preview")
    st.dataframe(
        df.head(20),
        width=700,
        hide_index=True
    )

    st.divider()
    st.subheader("Dataset Statistics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Failure cases",    f"{df['Machine_failure'].sum()}")
    c2.metric("Normal cases",     f"{(df['Machine_failure']==0).sum()}")
    c3.metric("Failure rate",     "3.39%")

    st.info("Click 'Run Live Simulation' in sidebar to start!")

# ── ML Analysis Tabs ────────────────────────────────────
st.divider()
st.subheader("ML Model Analysis — Swayam Arya")

tab1, tab2, tab3, tab4 = st.tabs([
    "SHAP Analysis",
    "Precision-Recall Curve",
    "Noise Analysis",
    "Model Metrics"
])

# Tab 1 — SHAP
with tab1:
    st.markdown("### SHAP Feature Importance")
    shap_summary = "reports/figures/shap_summary.png"
    shap_bar     = "reports/figures/shap_bar.png"
    shap_wfall   = "reports/figures/shap_waterfall.png"

    if os.path.exists(shap_summary):
        col1, col2 = st.columns(2)
        with col1:
            st.image(shap_summary, caption="SHAP Summary Plot", width=700)
        with col2:
            st.image(shap_bar, caption="SHAP Feature Importance", width=700)
        st.image(shap_wfall, caption="SHAP Waterfall — Single Failure Explanation", width=700)
    else:
        st.warning("SHAP plots not found!")

# Tab 2 — PR Curve
with tab2:
    st.markdown("### Precision-Recall Curve")
    pr_curve = "reports/figures/pr_curve.png"

    if os.path.exists(pr_curve):
        st.image(pr_curve, caption="PR Curve — Best Threshold = 0.40", width=700)
        col1, col2, col3 = st.columns(3)
        col1.metric("Default Threshold", "0.50")
        col2.metric("Tuned Threshold", "0.3988")
        col3.metric("F1 Improvement", "0.8639 → 0.8753")
    else:
        st.warning("PR curve not found!")

# Tab 3 — Noise Analysis
with tab3:
    st.markdown("### Noise Sensitivity Analysis")
    noise_plot = "reports/figures/week4_analysis.png"

    if os.path.exists(noise_plot):
        st.image(noise_plot, caption="Model Robustness — Noise vs Macro F1", width=700)

    noise_data = {
        "Noise Level": [0.01, 0.05, 0.10, 0.20],
        "Mean Macro F1": [0.8637, 0.8593, 0.8572, 0.8485],
        "Status": ["✅ Above Target", "✅ Above Target", 
                   "✅ Above Target", "✅ Above Target"]
    }
    st.dataframe(pd.DataFrame(noise_data), width=700, hide_index=True)

# Tab 4 — Model Metrics
with tab4:
    st.markdown("### Final Model Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Macro F1 (CV)", "0.8819", "+0.0624 vs baseline")
    col2.metric("ROC-AUC", "0.9644")
    col3.metric("Tuned Threshold", "0.3988")
    col4.metric("Noise Robust upto", "0.20")

    st.divider()

    metrics_data = {
        "Metric": [
            "Macro F1 — Baseline",
            "Macro F1 — With External",
            "Macro F1 — CV + SMOTE",
            "Tuned Macro F1",
            "ROC-AUC"
        ],
        "Value": ["0.8195", "0.8715", "0.8819", "0.8753", "0.9644"],
        "Week": ["Week 2", "Week 2", "Week 3", "Week 3", "Week 3"]
    }
    st.dataframe(pd.DataFrame(metrics_data), width=700, hide_index=True)

    st.markdown("**Top Features (SHAP):** Torque_Nm > Rotational_speed_rpm > Air_temperature_K")