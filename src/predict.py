"""
predict.py - Load saved model and predict yield for a single input.
"""

import joblib
import pandas as pd

from src.feature_engineering import add_features

MODEL_PATH = "models/yield_model.pkl"


def load_model(path: str = MODEL_PATH):
    artifact = joblib.load(path)
    return artifact["model"], artifact["features"]


def predict(input_data: dict) -> float:
    model, features = load_model()
    df = pd.DataFrame([input_data])
    df_eng = add_features(df)
    X = df_eng[features]
    return float(model.predict(X)[0])
