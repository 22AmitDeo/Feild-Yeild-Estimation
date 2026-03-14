"""
data_loader.py - Load and split the raw dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/train.csv"
TARGET = "yield"


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "field_id" in df.columns:
        df = df.drop(columns=["field_id"])

    return df


def get_splits(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )