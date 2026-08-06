"""
WattWise - Model Training Pipeline

Compares Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
Selects the best model by R² on test set and saves to models/model.pkl.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = os.path.join("dataset", "electricity_data.csv")
MODEL_PATH = os.path.join("models", "model.pkl")
METADATA_PATH = os.path.join("models", "model_metadata.json")

CATEGORICAL_FEATURES = [
    "HouseType", "Occupation", "IncomeLevel", "WorkFromHome", "Season"
]
NUMERIC_FEATURES = [
    "FamilyMembers", "Temperature", "Humidity",
    "ACCount", "ACHours", "FanCount", "LightCount", "LEDPercentage",
    "Cooler", "Geyser", "Refrigerator", "WashingMachine",
    "LaptopHours", "TVHours", "PreviousMonthUnits", "EstimatedBaseLoad",
]
TARGET = "MonthlyUnits"

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y, df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate_model(name: str, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")

    return {
        "name": name,
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "cv_r2_mean": round(cv_scores.mean(), 4),
        "cv_r2_std": round(cv_scores.std(), 4),
        "pipeline": pipeline,
    }


def get_feature_names(preprocessor):
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + NUMERIC_FEATURES


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Run generate_dataset.py first."
        )

    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    X, y, df = load_and_prepare_data(DATASET_PATH)
    print(f"Samples: {len(X)}, Features: {len(FEATURE_COLUMNS)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=18, min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }

    results = []
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("regressor", estimator),
        ])
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)
        print(f"\n{name}:")
        print(f"  R² Score:    {result['r2']}")
        print(f"  MAE:         {result['mae']} kWh")
        print(f"  RMSE:        {result['rmse']} kWh")
        print(f"  CV R² (5-fold): {result['cv_r2_mean']} ± {result['cv_r2_std']}")

    best = max(results, key=lambda r: r["r2"])
    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best['name']} (R² = {best['r2']})")
    print("=" * 60)

    best_pipeline = best["pipeline"]
    best_pipeline.fit(X, y)

    joblib.dump(best_pipeline, MODEL_PATH)

    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor_fitted)

    metadata = {
        "best_model": best["name"],
        "r2_score": best["r2"],
        "mae": best["mae"],
        "rmse": best["rmse"],
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "all_results": [
            {"name": r["name"], "r2": r["r2"], "mae": r["mae"], "rmse": r["rmse"]}
            for r in results
        ],
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
