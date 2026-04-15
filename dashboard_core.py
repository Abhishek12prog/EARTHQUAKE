import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import (
    BEST_MODEL_PATH,
    CUSTOM_RAW_DATA_PATH,
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    MODEL_RESULTS_PATH,
    PLOTS_DIR,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TRAINING_SETTINGS_PATH,
)
from data_collection import fetch_earthquake_data
from eda import run_eda
from model_building import train_and_evaluate
from preprocessing import build_feature_columns, preprocess_earthquake_data


def default_training_settings():
    return {
        "dataset_mode": "default",
        "feature_engineering": {
            "include_quarter": False,
            "include_is_weekend": False,
            "include_depth_bucket": False,
        },
        "algorithms": ["Logistic Regression", "Random Forest"],
        "logistic_regression": {"max_iter": 1000, "C": 1.0},
        "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
        "decision_tree": {"max_depth": 8, "min_samples_split": 2},
        "gradient_boosting": {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 3},
        "knn": {"n_neighbors": 9},
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    }


def load_training_settings():
    settings_path = Path(TRAINING_SETTINGS_PATH)
    if not settings_path.exists():
        settings = default_training_settings()
        settings_path.write_text(json.dumps(settings, indent=2))
        return settings

    try:
        stored = json.loads(settings_path.read_text())
    except Exception:
        stored = {}

    settings = default_training_settings()
    for key, value in stored.items():
        if key not in settings:
            continue
        if isinstance(settings[key], dict) and isinstance(value, dict):
            settings[key].update(value)
        else:
            settings[key] = value
    return settings


def save_training_settings(settings):
    Path(TRAINING_SETTINGS_PATH).write_text(json.dumps(settings, indent=2))


def get_active_raw_data_path(settings=None):
    settings = settings or load_training_settings()
    if settings.get("dataset_mode") == "custom" and Path(CUSTOM_RAW_DATA_PATH).exists():
        return CUSTOM_RAW_DATA_PATH
    return RAW_DATA_PATH


def inject_global_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg-top: #f4eadf;
            --bg-bottom: #f7fafc;
            --ink: #172033;
            --muted: #617186;
            --panel: rgba(255, 255, 255, 0.74);
            --stroke: rgba(23, 32, 51, 0.08);
            --accent: #ca6702;
            --accent-dark: #005f73;
            --good: #167c59;
            --warn: #a15c00;
            --danger: #a62c2b;
            --shadow: 0 20px 60px rgba(15, 23, 42, 0.10);
        }

        html, body, [class*="css"] {
            font-family: "Plus Jakarta Sans", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(202, 103, 2, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(0, 95, 115, 0.14), transparent 20%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
            position: fixed;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 0.55rem;
            padding-bottom: 2.75rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #172033 0%, #0d4f5d 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        .sidebar-brand {
            padding: 0.75rem 0 1rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.10);
            margin-bottom: 1rem;
        }

        .sidebar-brand-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .sidebar-brand-copy {
            display: none;
        }

        .hero-panel, .glass-card {
            background: var(--panel);
            backdrop-filter: blur(14px);
            border: 1px solid var(--stroke);
            box-shadow: var(--shadow);
        }

        .hero-panel {
            position: relative;
            overflow: hidden;
            border-radius: 32px;
            padding: 1.05rem 1.35rem;
            background:
                linear-gradient(135deg, rgba(23, 32, 51, 0.98), rgba(0, 95, 115, 0.93));
            color: #f8fafc;
            margin-bottom: 0.55rem;
        }

        .hero-panel::after {
            content: "";
            position: absolute;
            width: 320px;
            height: 320px;
            right: -80px;
            top: -100px;
            background: radial-gradient(circle, rgba(255, 183, 77, 0.28), transparent 62%);
        }

        .hero-kicker {
            display: inline-block;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
        }

        .hero-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.8rem;
            line-height: 1.02;
            margin: 0;
            max-width: 820px;
        }

        .hero-subtitle {
            display: none;
        }

        .glass-card {
            border-radius: 26px;
            padding: 1.2rem 1.2rem 1.15rem 1.2rem;
            margin-top: 0.95rem;
        }

        .hero-compact {
            min-height: auto;
        }

        .tall-card { min-height: 132px; }
        .stat-card { min-height: 132px; }
        .nav-card { margin-top: 1rem; }

        .card-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.45rem;
        }

        .card-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.42rem;
            color: var(--ink);
            margin-bottom: 0.55rem;
        }

        .metric-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 1.9rem;
            color: var(--ink);
            margin-bottom: 0.35rem;
        }

        .card-copy {
            color: var(--muted);
            font-size: 0.89rem;
            line-height: 1.45;
            margin: 0;
        }

        .page-title {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .page-subtitle {
            display: none;
        }

        [data-testid="stAppViewContainer"] .stSlider label,
        [data-testid="stAppViewContainer"] .stSelectbox label,
        [data-testid="stAppViewContainer"] .stSelectSlider label,
        [data-testid="stAppViewContainer"] .stNumberInput label,
        [data-testid="stAppViewContainer"] .stRadio label,
        [data-testid="stAppViewContainer"] .stCheckbox label,
        [data-testid="stAppViewContainer"] .stFileUploader label,
        [data-testid="stAppViewContainer"] [data-baseweb="radio"] *,
        [data-testid="stAppViewContainer"] [data-baseweb="checkbox"] *,
        [data-testid="stAppViewContainer"] [data-baseweb="slider"] *,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] small {
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"] div[data-baseweb="select"] *,
        [data-testid="stAppViewContainer"] div[data-baseweb="base-input"] * {
            color: var(--ink) !important;
        }

        [data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {
            background: #172033 !important;
            color: #f8fafc !important;
            border-color: rgba(23, 32, 51, 0.16) !important;
        }

        [data-testid="stAppViewContainer"] div[data-baseweb="select"] svg,
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] input,
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] span {
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }

        [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[role="button"],
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[role="button"] *,
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[role="combobox"],
        [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[role="combobox"] * {
            color: #f8fafc !important;
            fill: #f8fafc !important;
        }

        .risk-pill {
            display: inline-block;
            padding: 0.5rem 0.92rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.94rem;
            letter-spacing: 0.02em;
        }

        .risk-low { background: rgba(22,124,89,0.14); color: var(--good); }
        .risk-medium { background: rgba(202,103,2,0.14); color: var(--warn); }
        .risk-high { background: rgba(166,44,43,0.14); color: var(--danger); }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_shell():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Earthquake Prediction</div>
            <div class="sidebar-brand-copy"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_pipeline_artifacts():
    settings = load_training_settings()
    required_files = [
        Path(get_active_raw_data_path(settings)),
        Path(PROCESSED_DATA_PATH),
        Path(BEST_MODEL_PATH),
        Path(LABEL_ENCODER_PATH),
        Path(MODEL_RESULTS_PATH),
    ]

    if all(path.exists() for path in required_files):
        return

    with st.spinner("Preparing data and model assets..."):
        active_raw_data_path = get_active_raw_data_path(settings)
        if active_raw_data_path == RAW_DATA_PATH and not Path(RAW_DATA_PATH).exists():
            fetch_earthquake_data()
        if not Path(PROCESSED_DATA_PATH).exists():
            preprocess_earthquake_data(
                input_path=active_raw_data_path,
                training_options=settings,
            )
        run_eda()
        train_and_evaluate(training_options=settings)


def rebuild_model_artifacts():
    settings = load_training_settings()
    with st.spinner("Rebuilding model artifacts for this environment..."):
        active_raw_data_path = get_active_raw_data_path(settings)
        if active_raw_data_path == RAW_DATA_PATH and not Path(RAW_DATA_PATH).exists():
            fetch_earthquake_data()
        preprocess_earthquake_data(
            input_path=active_raw_data_path,
            training_options=settings,
        )
        train_and_evaluate(training_options=settings)


@st.cache_resource
def load_model_bundle():
    ensure_pipeline_artifacts()
    try:
        return joblib.load(BEST_MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)
    except Exception:
        for artifact_path in [Path(BEST_MODEL_PATH), Path(LABEL_ENCODER_PATH), Path(MODEL_RESULTS_PATH)]:
            if artifact_path.exists():
                artifact_path.unlink()
        rebuild_model_artifacts()
        return joblib.load(BEST_MODEL_PATH), joblib.load(LABEL_ENCODER_PATH)


@st.cache_data
def load_processed_data():
    ensure_pipeline_artifacts()
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).copy()


@st.cache_data
def load_model_results():
    ensure_pipeline_artifacts()
    return pd.read_csv(MODEL_RESULTS_PATH)


def load_plot_path(filename):
    return Path(PLOTS_DIR) / filename


def predict_probabilities(model, label_encoder, sample_input):
    if hasattr(model, "classes_"):
        predicted_risk = model.predict(sample_input)[0]
        probabilities = model.predict_proba(sample_input)[0]
        class_names = list(model.classes_)
    else:
        encoded_prediction = model.predict(sample_input)[0]
        predicted_risk = label_encoder.inverse_transform([encoded_prediction])[0]
        probabilities = model.predict_proba(sample_input)[0]
        class_names = list(label_encoder.classes_)

    probability_df = pd.DataFrame(
        {"Risk Level": class_names, "Probability": probabilities}
    ).sort_values(by="Probability", ascending=False)
    return predicted_risk, probability_df


def risk_class_name(risk_level):
    return f"risk-{risk_level.lower()}"


def prediction_input_frame(latitude, longitude, depth, year, month, day, hour, day_of_week):
    settings = load_training_settings()
    feature_columns = build_feature_columns(training_options=settings)

    record = {
        "latitude": latitude,
        "longitude": longitude,
        "depth": depth,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "day_of_week": day_of_week,
    }

    if "quarter" in feature_columns:
        record["quarter"] = ((month - 1) // 3) + 1
    if "is_weekend" in feature_columns:
        record["is_weekend"] = int(day_of_week in [5, 6])
    if "depth_bucket" in feature_columns:
        if depth <= 70:
            record["depth_bucket"] = 0
        elif depth <= 300:
            record["depth_bucket"] = 1
        else:
            record["depth_bucket"] = 2

    return pd.DataFrame([record], columns=feature_columns)
