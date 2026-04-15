import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

from dashboard_core import (
    inject_global_styles,
    load_model_bundle,
    predict_probabilities,
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

# ✅ LOAD MODEL
model, label_encoder = load_model_bundle()

# 🔹 INPUT UI
left, right = st.columns([1, 1])

with left:
    st.subheader("Input Parameters")

    latitude = st.slider("Latitude", -90.0, 90.0, 34.05, 0.05)
    longitude = st.slider("Longitude", -180.0, 180.0, -118.25, 0.05)
    depth = st.slider("Depth (km)", 0.0, 700.0, 12.5, 0.5)

    year = st.slider("Year", 2020, 2030, 2025)
    month = st.slider("Month", 1, 12, 7)
    day = st.slider("Day", 1, 31, 15)
    hour = st.slider("Hour", 0, 23, 14)

    # ✅ FIXED (numeric encoding)
    day_of_week = st.selectbox(
        "Day of Week",
        [
            (0, "Monday"),
            (1, "Tuesday"),
            (2, "Wednesday"),
            (3, "Thursday"),
            (4, "Friday"),
            (5, "Saturday"),
            (6, "Sunday"),
        ],
        format_func=lambda x: x[1]
    )[0]

# 🔹 CREATE MODEL INPUT (VERY IMPORTANT)
sample_input = pd.DataFrame([{
    "latitude": latitude,
    "longitude": longitude,
    "depth": depth,
    "year": year,
    "month": month,
    "day": day,
    "hour": hour,
    "day_of_week": day_of_week
}])

# 🔍 DEBUG (remove later if you want)
# st.write(sample_input)
# st.write(sample_input.dtypes)

# 🔹 PREDICTION
try:
    predicted_risk, probability_df = predict_probabilities(
        model, label_encoder, sample_input
    )

    with right:
        st.subheader("Prediction Result")

        st.success(f"Predicted Risk: {predicted_risk}")

        # 📊 CHART
        palette = {"Low": "green", "Medium": "orange", "High": "red"}

        fig, ax = plt.subplots()
        sns.barplot(
            data=probability_df,
            x="Probability",
            y="Risk Level",
            palette=palette,
            ax=ax
        )

        ax.set_xlim(0, 1)
        st.pyplot(fig)

        # 📋 TABLE
        display_df = probability_df.copy()
        display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df)

except Exception as e:
    st.error("Prediction failed. Please retrain the model or check input format.")
    st.exception(e)