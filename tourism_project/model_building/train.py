# train.py
# Trains and compares Logistic Regression, Random Forest and XGBoost models
# for predicting ProdTaken, tracks every run with MLflow, and persists the
# best-performing pipeline (preprocessing + model) to disk.

import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/model_building"
EXPERIMENT_NAME = "Tourism_Wellness_Package_Prediction"

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]


def load_splits():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.08,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(1 - 0.193) / 0.193,  # counter class imbalance
        ),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_splits()
    preprocessor = build_preprocessor()
    candidates = get_candidate_models()

    results = {}
    best_model_name, best_pipeline, best_score = None, None, -1

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_params({"model_type": name, **{
                k: v for k, v in model.get_params().items()
                if isinstance(v, (int, float, str, bool)) or v is None
            }})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, artifact_path="model", serialization_format="pickle"
            )

            results[name] = metrics
            print(f"{name}: {json.dumps(metrics, indent=2)}")

            # Selection metric: F1 on the positive (purchase) class, since both
            # false positives (wasted sales calls) and false negatives (missed
            # customers) are costly for the marketing team.
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_model_name = name
                best_pipeline = pipe

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f"{MODEL_DIR}/best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    with open(f"{MODEL_DIR}/results.json", "w") as f:
        json.dump({"results": results, "best_model": best_model_name}, f, indent=2)

    print(f"\nBest model: {best_model_name} (F1 = {best_score:.4f}) saved to {model_path}")
    return results, best_model_name


if __name__ == "__main__":
    main()
