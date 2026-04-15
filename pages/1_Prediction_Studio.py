import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from dashboard_core import (
    inject_global_styles,
    load_model_bundle,
    prediction_input_frame,
    predict_probabilities,
    risk_class_name,
)

st.set_page_config(
    page_title="Prediction Studio",
    page_icon="🌍",
    layout="wide",
)

sns.set_theme(style="whitegrid")

# Apply styles
inject_global_styles()

st.title("🌍 Prediction Studio")

model, label_encoder = load_model_bundle()

left, right = st.columns([1.03, 0.97])

# 🔹 LEFT PANEL (INPUTS)
with left:
    st.markdown("### Input Parameters")

    latitude = st.slider("Latitude", -90.0, 90.0, 34.05, 0.05)
    longitude = st.slider("Longitude", -180.0, 180.0, -118.25, 0.05)
    depth = st.slider("Depth (km)", 0.0, 700.0, 12.5, 0.5)

    year = st.slider("Year", 2020, 2030, 2025)
    month = st.slider("Month", 1, 12, 7)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 14)

    day_of_week = st.selectbox(
        "Day of Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )

# 🔹 PREP INPUT
sample_input = prediction_input_frame(
    latitude, longitude, depth, year, month, day, hour, day_of_week
)

predicted_risk, probability_df = predict_probabilities(
    model, label_encoder, sample_input
)

# 🔹 RIGHT PANEL (OUTPUT)
with right:
    st.markdown("### Prediction Result")

    st.success(f"Predicted Risk: {predicted_risk}")

    palette = {"Low": "green", "Medium": "orange", "High": "red"}

    fig, ax = plt.subplots()
    sns.barplot(
        data=probability_df,
        x="Probability",
        y="Risk Level",
        palette=palette,
        ax=ax
    )

    st.pyplot(fig)

    st.dataframe(probability_df)