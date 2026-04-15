USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"

QUERY_PARAMS = {
    "format": "csv",
    "starttime": "2023-01-01",
    "endtime": "2026-04-16",
    "orderby": "time-asc",
    "minmagnitude": 2.5,
    "limit": 5000,
}

RAW_DATA_PATH = "earthquakes_raw.csv"
PROCESSED_DATA_PATH = "earthquakes_processed.csv"
MODEL_RESULTS_PATH = "model_results.csv"
BEST_MODEL_PATH = "best_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
PLOTS_DIR = "plots"
REQUEST_TIMEOUT_SECONDS = 30
TRAINING_SETTINGS_PATH = "training_settings.json"
CUSTOM_RAW_DATA_PATH = "custom_earthquakes_raw.csv"

FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "depth",
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
]

RISK_ORDER = ["Low", "Medium", "High"]
