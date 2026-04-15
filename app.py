import streamlit as st

from dashboard_core import inject_global_styles, render_app_shell


st.set_page_config(
    page_title="Earthquake Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown(
        """
        <section class="hero-panel hero-compact">
            <div class="hero-kicker">Earthquake Prediction</div>
            <h1 class="hero-title">Seismic analysis and prediction dashboard</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
from data_collection import fetch_earthquake_data
from eda import run_eda
from model_building import train_and_evaluate
from prediction import predict_sample
from preprocessing import preprocess_earthquake_data


def main():
    print("Step 1/5: Fetching earthquake data...")
    fetch_earthquake_data()
    print("Step 2/5: Preprocessing data...")
    preprocess_earthquake_data()
    print("Step 3/5: Running EDA and saving plots...")
    run_eda()
    print("Step 4/5: Training and evaluating models...")
    train_and_evaluate()
    print("Step 5/5: Running sample prediction...")
    predict_sample()
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()

import streamlit as st

from dashboard_core import inject_global_styles, render_app_shell


st.set_page_config(
    page_title="Earthquake Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown(
        """
        <section class="hero-panel hero-compact">
            <div class="hero-kicker">Earthquake Prediction</div>
            <h1 class="hero-title">Seismic analysis and prediction dashboard</h1>
        </section>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
z