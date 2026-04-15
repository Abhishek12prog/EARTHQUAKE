import joblib
import pandas as pd

from config import BEST_MODEL_PATH, LABEL_ENCODER_PATH


def predict_sample():
    model = joblib.load(BEST_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    sample_input = pd.DataFrame(
        [
            {
                "latitude": 34.05,
                "longitude": -118.25,
                "depth": 12.5,
                "year": 2025,
                "month": 7,
                "day": 15,
                "hour": 14,
                "day_of_week": 1,
            }
        ]
    )

    if hasattr(model, "classes_"):
        predicted_risk = model.predict(sample_input)[0]
        probabilities = model.predict_proba(sample_input)[0]
        class_names = model.classes_
    else:
        predicted_class_encoded = model.predict(sample_input)[0]
        predicted_risk = label_encoder.inverse_transform([predicted_class_encoded])[0]
        probabilities = model.predict_proba(sample_input)[0]
        class_names = label_encoder.classes_

    probability_df = pd.DataFrame(
        {"Risk Level": class_names, "Probability": probabilities}
    ).sort_values(by="Probability", ascending=False)

    print("Sample input:")
    print(sample_input)
    print(f"\nPredicted earthquake risk level: {predicted_risk}")
    print("\nPredicted probabilities:")
    print(probability_df)

    return predicted_risk, probability_df


if __name__ == "__main__":
    predict_sample()
