from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    CUSTOM_RAW_DATA_PATH,
    LABEL_ENCODER_PATH,
    MODEL_RESULTS_PATH,
    PROCESSED_DATA_PATH,
    TRAINING_SETTINGS_PATH,
)
from dashboard_core import (
    default_training_settings,
    get_active_raw_data_path,
    inject_global_styles,
    load_training_settings,
    render_app_shell,
    save_training_settings,
)
from data_collection import fetch_earthquake_data
from eda import run_eda
from model_building import train_and_evaluate
from preprocessing import preprocess_earthquake_data


st.set_page_config(
    page_title="Training Studio",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def retrain_pipeline(settings):
    active_raw_data_path = get_active_raw_data_path(settings)

    if active_raw_data_path == "earthquakes_raw.csv" and not Path(active_raw_data_path).exists():
        fetch_earthquake_data()

    preprocess_earthquake_data(input_path=active_raw_data_path, training_options=settings)
    run_eda()
    results_df, best_model_name, _, _, _, _, _ = train_and_evaluate(training_options=settings)

    st.cache_data.clear()
    st.cache_resource.clear()
    return results_df, best_model_name


def main():
    inject_global_styles()
    render_app_shell()

    st.markdown('<div class="page-title">Training Studio</div>', unsafe_allow_html=True)

    settings = load_training_settings()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    with st.form("training_studio_form"):
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Dataset")
            dataset_mode = st.radio(
                "Dataset Source",
                options=["default", "custom"],
                format_func=lambda value: "USGS Dataset" if value == "default" else "Upload CSV",
                index=0 if settings.get("dataset_mode") == "default" else 1,
            )
            uploaded_file = st.file_uploader(
                "Upload earthquake CSV",
                type=["csv"],
                help="CSV should include time, latitude, longitude, depth, and mag or magnitude columns.",
            )

            st.markdown("### Feature Engineering")
            include_quarter = st.checkbox(
                "Quarter",
                value=settings["feature_engineering"].get("include_quarter", False),
            )
            include_is_weekend = st.checkbox(
                "Weekend Flag",
                value=settings["feature_engineering"].get("include_is_weekend", False),
            )
            include_depth_bucket = st.checkbox(
                "Depth Bucket",
                value=settings["feature_engineering"].get("include_depth_bucket", False),
            )

        with right:
            st.markdown("### Algorithms")
            algorithms = st.multiselect(
                "Select Models",
                options=[
                    "Logistic Regression",
                    "Random Forest",
                    "Decision Tree",
                    "Gradient Boosting",
                    "KNN",
                    "XGBoost",
                ],
                default=settings.get("algorithms", ["Logistic Regression", "Random Forest"]),
            )

            st.markdown("### Parameter Tuning")
            rf_estimators = st.slider(
                "Random Forest Trees",
                min_value=100,
                max_value=600,
                value=settings["random_forest"].get("n_estimators", 200),
                step=50,
            )
            rf_max_depth_raw = st.slider(
                "Random Forest Max Depth",
                min_value=0,
                max_value=30,
                value=settings["random_forest"].get("max_depth") or 0,
                step=1,
                help="Set to 0 for no max depth.",
            )
            rf_min_split = st.slider(
                "Random Forest Min Samples Split",
                min_value=2,
                max_value=10,
                value=settings["random_forest"].get("min_samples_split", 2),
                step=1,
            )
            logistic_c = st.slider(
                "Logistic Regression C",
                min_value=0.1,
                max_value=5.0,
                value=float(settings["logistic_regression"].get("C", 1.0)),
                step=0.1,
            )
            knn_neighbors = st.slider(
                "KNN Neighbors",
                min_value=3,
                max_value=25,
                value=settings["knn"].get("n_neighbors", 9),
                step=2,
            )

        save_clicked = st.form_submit_button("Save Settings")
    st.markdown("</div>", unsafe_allow_html=True)

    if save_clicked:
        if dataset_mode == "custom":
            if uploaded_file is None and not Path(CUSTOM_RAW_DATA_PATH).exists():
                st.error("Upload a custom CSV before saving custom mode.")
                return
            if uploaded_file is not None:
                Path(CUSTOM_RAW_DATA_PATH).write_bytes(uploaded_file.getvalue())

        updated_settings = default_training_settings()
        updated_settings["dataset_mode"] = dataset_mode
        updated_settings["feature_engineering"] = {
            "include_quarter": include_quarter,
            "include_is_weekend": include_is_weekend,
            "include_depth_bucket": include_depth_bucket,
        }
        updated_settings["algorithms"] = algorithms or ["Random Forest"]
        updated_settings["logistic_regression"]["C"] = logistic_c
        updated_settings["random_forest"]["n_estimators"] = rf_estimators
        updated_settings["random_forest"]["max_depth"] = None if rf_max_depth_raw == 0 else rf_max_depth_raw
        updated_settings["random_forest"]["min_samples_split"] = rf_min_split
        updated_settings["knn"]["n_neighbors"] = knn_neighbors
        save_training_settings(updated_settings)
        st.success(f"Saved training settings to {TRAINING_SETTINGS_PATH}.")

    if st.button("Retrain Model", type="primary", use_container_width=True):
        settings = load_training_settings()
        for artifact in [BEST_MODEL_PATH, LABEL_ENCODER_PATH, MODEL_RESULTS_PATH, PROCESSED_DATA_PATH]:
            artifact_path = Path(artifact)
            if artifact_path.exists():
                artifact_path.unlink()
        with st.spinner("Training models with selected settings..."):
            results_df, best_model_name = retrain_pipeline(settings)
        st.success(f"Retraining complete. Best model: {best_model_name}")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("### Dataset Preview")
    active_raw_data_path = Path(get_active_raw_data_path(load_training_settings()))
    if active_raw_data_path.exists():
        preview_df = pd.read_csv(active_raw_data_path).head(10)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
    else:
        st.info("No dataset is available yet.")


if __name__ == "__main__":
    main()
