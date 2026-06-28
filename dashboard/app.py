import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# ---------------- TITLE ----------------
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("### Machine Failure Analysis & Insights")

# ---------------- STYLE (MAKE IMAGES SAME SIZE) ----------------
st.markdown("""
<style>
img {
    height: 300px !important;
    object-fit: contain;
    border-radius: 10px;
}
.block-container {
    padding-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BASE URL ----------------
BASE_URL = "https://raw.githubusercontent.com/Swayam085/predictive-maintenance/main/reports/figures/"

# ---------------- IMAGE FUNCTION ----------------
def show_image(filename, title):
    url = BASE_URL + filename

    st.markdown(f"""
    <div style="background-color:#111; padding:10px; border-radius:10px; margin-bottom:10px;">
        <h5 style="text-align:center; color:white;">{title}</h5>
    </div>
    """, unsafe_allow_html=True)

    st.image(url, use_container_width=True)

# ---------------- DASHBOARD ----------------

# 📊 Overview
st.subheader("📊 Overview")
col1, col2 = st.columns(2)

with col1:
    show_image("01_class_distribution.png", "Failure Distribution")

with col2:
    show_image("10_feature_importance.png", "Feature Importance")

# 🔍 Key Insights
st.subheader("🔍 Key Insights")
col3, col4 = st.columns(2)

with col3:
    show_image("07_speed_vs_torque.png", "Speed vs Torque")

with col4:
    show_image("04_features_by_failure.png", "Features by Failure")

# ⚙️ Additional Analysis
st.subheader("⚙️ Additional Analysis")
col5, col6 = st.columns(2)

with col5:
    show_image("05_failure_types.png", "Failure Types")

with col6:
    show_image("03_correlation_heatmap.png", "Correlation Heatmap")

# ---------------- FOOTER ----------------
st.markdown("---")
st.success("✅ Dashboard Fully Working & Styled")