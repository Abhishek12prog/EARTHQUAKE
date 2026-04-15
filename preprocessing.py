import pandas as pd

from config import FEATURE_COLUMNS, PROCESSED_DATA_PATH, RAW_DATA_PATH


def magnitude_to_risk(magnitude):
    if magnitude < 3:
        return "Low"
    if magnitude < 5:
        return "Medium"
    return "High"


def build_feature_columns(training_options=None):
    training_options = training_options or {}
    feature_flags = training_options.get("feature_engineering", {})

    feature_columns = FEATURE_COLUMNS.copy()

    if feature_flags.get("include_quarter"):
        feature_columns.append("quarter")
    if feature_flags.get("include_is_weekend"):
        feature_columns.append("is_weekend")
    if feature_flags.get("include_depth_bucket"):
        feature_columns.append("depth_bucket")

    return feature_columns


def preprocess_earthquake_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH, training_options=None):
    df = pd.read_csv(input_path)

    if "mag" not in df.columns and "magnitude" in df.columns:
        df = df.rename(columns={"magnitude": "mag"})

    earthquakes = df[["time", "latitude", "longitude", "depth", "mag"]].copy()
    earthquakes.rename(columns={"mag": "magnitude"}, inplace=True)
    earthquakes.dropna(inplace=True)

    earthquakes["time"] = pd.to_datetime(earthquakes["time"], errors="coerce")
    earthquakes.dropna(subset=["time"], inplace=True)

    earthquakes["year"] = earthquakes["time"].dt.year
    earthquakes["month"] = earthquakes["time"].dt.month
    earthquakes["day"] = earthquakes["time"].dt.day
    earthquakes["hour"] = earthquakes["time"].dt.hour
    earthquakes["day_of_week"] = earthquakes["time"].dt.dayofweek

    feature_columns = build_feature_columns(training_options=training_options)

    if "quarter" in feature_columns:
        earthquakes["quarter"] = earthquakes["time"].dt.quarter
    if "is_weekend" in feature_columns:
        earthquakes["is_weekend"] = earthquakes["day_of_week"].isin([5, 6]).astype(int)
    if "depth_bucket" in feature_columns:
        earthquakes["depth_bucket"] = pd.cut(
            earthquakes["depth"],
            bins=[-1, 70, 300, 1000],
            labels=[0, 1, 2],
        ).astype(int)

    earthquakes["risk_level"] = earthquakes["magnitude"].apply(magnitude_to_risk)

    required_columns = ["time", "magnitude", "risk_level", *feature_columns]
    earthquakes = earthquakes[required_columns]
    earthquakes.to_csv(output_path, index=False)

    print(f"Saved processed data to {output_path} with shape {earthquakes.shape}")
    return earthquakes


if __name__ == "__main__":
    preprocess_earthquake_data()
