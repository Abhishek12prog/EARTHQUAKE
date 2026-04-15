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
