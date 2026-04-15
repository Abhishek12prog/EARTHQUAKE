import pandas as pd
import streamlit as st
from pathlib import Path

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
)

# Apply styles
inject_global_styles()

st.title("⚙️ Training Studio")


# 🔁 PIPELINE
def retrain_pipeline(settings):
    active_raw_data_path = get_active_raw_data_path(settings)

    if active_raw_data_path == "earthquakes_raw.csv" and not Path(active_raw_data_path).exists():
        fetch_earthquake_data()

    preprocess_earthquake_data(input_path=active_raw_data_path, training_options=settings)
    run_eda()
    results_df, best_model_name, *_ = train_and_evaluate(training_options=settings)

    st.cache_data.clear()
    st.cache_resource.clear()
    return results_df, best_model_name


# 🔹 LOAD SETTINGS
settings = load_training_settings()


# 🔹 FORM
with st.form("training_form"):
    left, right = st.columns(2)

    # LEFT SIDE
    with left:
        st.subheader("Dataset")

        dataset_mode = st.radio(
            "Dataset Source",
            ["default", "custom"],
            format_func=lambda x: "USGS Dataset" if x == "default" else "Upload CSV",
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        st.subheader("Feature Engineering")
        include_quarter = st.checkbox("Quarter", value=True)
        include_is_weekend = st.checkbox("Weekend Flag", value=True)
        include_depth_bucket = st.checkbox("Depth Bucket", value=True)

    # RIGHT SIDE
    with right:
        st.subheader("Algorithms")

        algorithms = st.multiselect(
            "Select Models",
            [
                "Logistic Regression",
                "Random Forest",
                "Decision Tree",
                "Gradient Boosting",
                "KNN",
                "XGBoost",
            ],
            default=["Random Forest"],
        )

        st.subheader("Tuning")

        rf_estimators = st.slider("RF Trees", 100, 600, 200, step=50)
        rf_depth = st.slider("RF Depth", 0, 30, 10)
        knn_neighbors = st.slider("KNN Neighbors", 3, 25, 9)

    save_clicked = st.form_submit_button("Save Settings")


# 🔹 SAVE SETTINGS
if save_clicked:
    if dataset_mode == "custom" and uploaded_file:
        Path(CUSTOM_RAW_DATA_PATH).write_bytes(uploaded_file.getvalue())

    new_settings = default_training_settings()
    new_settings["dataset_mode"] = dataset_mode
    new_settings["algorithms"] = algorithms

    save_training_settings(new_settings)

    st.success("Settings saved!")


# 🔹 TRAIN BUTTON
if st.button("🚀 Retrain Model"):
    settings = load_training_settings()

    for path in [BEST_MODEL_PATH, LABEL_ENCODER_PATH, MODEL_RESULTS_PATH, PROCESSED_DATA_PATH]:
        p = Path(path)
        if p.exists():
            p.unlink()

    with st.spinner("Training..."):
        results_df, best_model = retrain_pipeline(settings)

    st.success(f"Best Model: {best_model}")
    st.dataframe(results_df)


# 🔹 PREVIEW
st.subheader("Dataset Preview")

data_path = Path(get_active_raw_data_path(load_training_settings()))

if data_path.exists():
    df = pd.read_csv(data_path).head(10)
    st.dataframe(df)
else:
    st.info("No dataset available yet.")