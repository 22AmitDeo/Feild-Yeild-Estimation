"""
evaluate_model.py - Compute metrics and generate evaluation plots.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLOTS_DIR = "results/plots"
METRICS_PATH = "results/metrics.json"


def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    metrics = {
        "model": model_name,
        "r2": round(r2_score(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
    }
    return metrics


def save_metrics(all_metrics: list):
    os.makedirs("results", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


def plot_correlation_heatmap(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved: correlation_heatmap.png")


def plot_prediction_vs_actual(y_true, y_pred, model_name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Prediction vs Actual — {model_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/pred_vs_actual_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: pred_vs_actual_{model_name}.png")


def plot_residuals(y_true, y_pred, model_name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Predicted Yield")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residual Plot — {model_name}")
    fig.tight_layout()
    fig.savefig(f"{PLOTS_DIR}/residuals_{model_name.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: residuals_{model_name}.png")
