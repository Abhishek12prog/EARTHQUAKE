import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from dashboard_core import (
    inject_global_styles,
    load_model_bundle,
    prediction_input_frame,
    predict_probabilities,
    render_app_shell,
    risk_class_name,
)


st.set_page_config(
    page_title="Prediction Studio",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown('<div class="page-title">Prediction Studio</div>', unsafe_allow_html=True)

    model, label_encoder = load_model_bundle()

    left, right = st.columns([1.03, 0.97])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        latitude = st.slider("Latitude", -90.0, 90.0, 34.05, 0.05)
        longitude = st.slider("Longitude", -180.0, 180.0, -118.25, 0.05)
        depth = st.slider("Depth (km)", 0.0, 700.0, 12.5, 0.5)
        year = st.slider("Year", 2020, 2030, 2025, 1)
        month = st.select_slider("Month", options=list(range(1, 13)), value=7)
        day = st.select_slider("Day", options=list(range(1, 32)), value=15)
        hour = st.select_slider("Hour", options=list(range(0, 24)), value=14)
        day_of_week = st.selectbox(
            "Day of Week",
            options=[
                (0, "Monday"),
                (1, "Tuesday"),
                (2, "Wednesday"),
                (3, "Thursday"),
                (4, "Friday"),
                (5, "Saturday"),
                (6, "Sunday"),
            ],
            index=1,
            format_func=lambda item: item[1],
        )[0]
        st.markdown("</div>", unsafe_allow_html=True)

    sample_input = prediction_input_frame(
        latitude, longitude, depth, year, month, day, hour, day_of_week
    )
    predicted_risk, probability_df = predict_probabilities(model, label_encoder, sample_input)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(
            f'<span class="risk-pill {risk_class_name(predicted_risk)}">{predicted_risk} Band</span>',
            unsafe_allow_html=True,
        )
        st.write("")

        palette = {"Low": "#167c59", "Medium": "#ca6702", "High": "#a62c2b"}
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        sns.barplot(
            data=probability_df,
            x="Probability",
            y="Risk Level",
            hue="Risk Level",
            dodge=False,
            palette=palette,
            ax=ax,
            legend=False,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("")
        ax.set_title("Prediction Breakdown", fontsize=13, weight="bold")
        for idx, value in enumerate(probability_df["Probability"]):
            ax.text(value + 0.015, idx, f"{value:.1%}", va="center", fontsize=10)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        display_df = probability_df.copy()
        display_df["Probability"] = display_df["Probability"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
