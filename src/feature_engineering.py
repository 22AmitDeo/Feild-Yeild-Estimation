"""
feature_engineering.py - Add interaction features and domain-specific indices.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Interaction features
    df["rainfall_x_temperature"] = df["rainfall"] * df["temperature"]
    df["humidity_x_temperature"] = df["humidity"] * df["temperature"]
    df["organic_matter_x_rainfall"] = df["organic_matter"] * df["rainfall"]
    df["sand_pct_x_rainfall"] = df["sand_pct"] * df["rainfall"]

    # Ratio features
    df["rainfall_to_temp_ratio"] = df["rainfall"] / (df["temperature"] + 1e-6)

    # Soil quality index: higher organic matter and optimal pH (6-7) = better soil
    df["soil_quality_index"] = df["organic_matter"] / (abs(df["soil_ph"] - 6.5) + 1)

    return df


ENGINEERED_FEATURES = [
    "soil_ph", "organic_matter", "sand_pct", "temperature", "humidity", "rainfall", "ndvi",
    "rainfall_x_temperature", "humidity_x_temperature",
    "organic_matter_x_rainfall", "sand_pct_x_rainfall",
    "rainfall_to_temp_ratio", "soil_quality_index",
]
