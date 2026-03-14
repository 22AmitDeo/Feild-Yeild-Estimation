"""
train_model.py - Full ML pipeline: train all models, evaluate, select best, save.

Run with:
    python src/train_model.py
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.evaluate_model import (
    compute_metrics,
    plot_correlation_heatmap,
    plot_prediction_vs_actual,
    plot_residuals,
    save_metrics,
)

from src.feature_engineering import ENGINEERED_FEATURES, add_features


MODEL_PATH = "models/yield_model.pkl"
PLOTS_DIR = "results/plots"


# ───────────────────────────────────────────────
# Model definitions
# ───────────────────────────────────────────────

def build_models():

    linear_pipe = lambda estimator: Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    return {

        "Linear Regression": linear_pipe(LinearRegression()),

        "Ridge Regression": linear_pipe(Ridge(alpha=1.0)),

        "Lasso Regression": linear_pipe(Lasso(alpha=1.0, max_iter=5000)),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
    }


# ───────────────────────────────────────────────
# Feature Importance
# ───────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if not hasattr(estimator, "feature_importances_"):
        print(f"[skip] {model_name} has no feature_importances_")
        return

    importances = estimator.feature_importances_

    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(range(len(importances)), importances[indices])

    ax.set_xticks(range(len(importances)))

    ax.set_xticklabels(
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right"
    )

    ax.set_title(f"Feature Importance — {model_name}")

    ax.set_ylabel("Importance")

    fig.tight_layout()

    path = f"{PLOTS_DIR}/feature_importance_{model_name.replace(' ', '_')}.png"

    fig.savefig(path, dpi=150)

    plt.close(fig)

    print(f"Saved: {path}")


# ───────────────────────────────────────────────
# SHAP
# ───────────────────────────────────────────────

def run_shap(model, X_test, model_name):

    os.makedirs(PLOTS_DIR, exist_ok=True)

    try:

        estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model

        if not hasattr(estimator, "feature_importances_"):
            print(f"[SHAP skip] {model_name} is not a tree model")
            return

        explainer = shap.TreeExplainer(estimator)

        shap_values = explainer.shap_values(X_test)

        shap.summary_plot(shap_values, X_test, show=False)

        plt.title(f"SHAP Summary — {model_name}")

        plt.tight_layout()

        path = f"{PLOTS_DIR}/shap_summary_{model_name.replace(' ', '_')}.png"

        plt.savefig(path, dpi=150, bbox_inches="tight")

        plt.close()

        print(f"Saved: {path}")

    except Exception as e:
        print(f"[SHAP skip] {model_name}: {e}")


# ───────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────

def main():

    print("=" * 60)
    print("Field Yield Estimation — Training Pipeline")
    print("=" * 60)

    # 1️⃣ Load data
    df = load_data()

    # 2️⃣ Feature engineering
    df = add_features(df)

    if "yield" not in df.columns:
        raise ValueError("Column 'yield' not found in dataset")

    # 3️⃣ Feature selection
    available_features = [f for f in ENGINEERED_FEATURES if f in df.columns]

    print(f"Using {len(available_features)} features")

    # Correlation heatmap
    plot_correlation_heatmap(df[available_features + ["yield"]])

    # 4️⃣ Train-test split
    X = df[available_features]

    y = df["yield"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 5️⃣ Train models
    models = build_models()

    best_model = None
    best_name = ""
    best_r2 = -np.inf

    all_metrics = []

    for name, model in models.items():

        print(f"\nTraining: {name}")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, name)

        all_metrics.append(metrics)

        print(
            f"R²={metrics['r2']}  RMSE={metrics['rmse']}  MAE={metrics['mae']}"
        )

        if metrics["r2"] > best_r2:

            best_r2 = metrics["r2"]

            best_model = model

            best_name = name

    # 6️⃣ Save metrics
    save_metrics(all_metrics)

    print(f"\nBest model: {best_name} (R²={best_r2})")

    # 7️⃣ Best model plots
    y_pred_best = best_model.predict(X_test)

    plot_prediction_vs_actual(y_test, y_pred_best, best_name)

    plot_residuals(y_test, y_pred_best, best_name)

    plot_feature_importance(best_model, available_features, best_name)

    run_shap(best_model, X_test, best_name)

    # 8️⃣ Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "features": available_features
        },
        MODEL_PATH
    )

    print(f"\nModel saved to {MODEL_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()