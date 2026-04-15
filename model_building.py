import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    BEST_MODEL_PATH,
    LABEL_ENCODER_PATH,
    MODEL_RESULTS_PATH,
    PROCESSED_DATA_PATH,
    RISK_ORDER,
)


def get_feature_columns(df):
    return [column for column in df.columns if column not in {"time", "magnitude", "risk_level"}]


def get_models(label_encoder=None, training_options=None):
    training_options = training_options or {}
    selected_algorithms = training_options.get("algorithms", ["Logistic Regression", "Random Forest"])
    logistic_params = training_options.get("logistic_regression", {})
    random_forest_params = training_options.get("random_forest", {})
    decision_tree_params = training_options.get("decision_tree", {})
    gradient_boosting_params = training_options.get("gradient_boosting", {})
    knn_params = training_options.get("knn", {})
    xgboost_params = training_options.get("xgboost", {})

    models = {}

    if "Logistic Regression" in selected_algorithms:
        models["Logistic Regression"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=logistic_params.get("max_iter", 1000),
                        class_weight="balanced",
                        C=logistic_params.get("C", 1.0),
                        random_state=42,
                    ),
                ),
            ]
        )

    if "Random Forest" in selected_algorithms:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=random_forest_params.get("n_estimators", 200),
            max_depth=random_forest_params.get("max_depth"),
            min_samples_split=random_forest_params.get("min_samples_split", 2),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    if "Decision Tree" in selected_algorithms:
        models["Decision Tree"] = DecisionTreeClassifier(
            max_depth=decision_tree_params.get("max_depth"),
            min_samples_split=decision_tree_params.get("min_samples_split", 2),
            random_state=42,
        )

    if "Gradient Boosting" in selected_algorithms:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=gradient_boosting_params.get("n_estimators", 150),
            learning_rate=gradient_boosting_params.get("learning_rate", 0.1),
            random_state=42,
        )

    if "KNN" in selected_algorithms:
        models["KNN"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=knn_params.get("n_neighbors", 9))),
            ]
        )

    if "XGBoost" in selected_algorithms and label_encoder is not None:
        try:
            from xgboost import XGBClassifier

            models["XGBoost"] = XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_encoder.classes_),
                n_estimators=xgboost_params.get("n_estimators", 200),
                max_depth=xgboost_params.get("max_depth", 6),
                learning_rate=xgboost_params.get("learning_rate", 0.1),
                subsample=xgboost_params.get("subsample", 0.9),
                colsample_bytree=xgboost_params.get("colsample_bytree", 0.9),
                eval_metric="mlogloss",
                random_state=42,
            )
        except Exception as exc:
            print(f"Skipping XGBoost because it is not usable in this environment: {exc}")

    if not models:
        models["Random Forest"] = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    return models


def train_and_evaluate(input_path=PROCESSED_DATA_PATH, training_options=None):
    df = pd.read_csv(input_path)
    feature_columns = get_feature_columns(df)
    X = df[feature_columns].copy()
    y = df["risk_level"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    models = get_models(label_encoder=label_encoder, training_options=training_options)
    results = []
    trained_models = {}

    for model_name, model in models.items():
        if model_name == "XGBoost":
            model.fit(X_train, y_train_encoded)
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        trained_models[model_name] = model
        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )

        print(f"\nModel: {model_name}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=RISK_ORDER))

    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    results_df.to_csv(MODEL_RESULTS_PATH, index=False)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    print(f"\nSaved model comparison to {MODEL_RESULTS_PATH}")
    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Best model: {best_model_name}")

    return results_df, best_model_name, best_model, X_test, y_test, y_test_encoded, label_encoder


if __name__ == "__main__":
    train_and_evaluate()
