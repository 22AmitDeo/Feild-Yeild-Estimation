import os
import sys
import joblib
import pandas as pd

# Allow project root imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import add_features

MODEL_PATH = "models/yield_model.pkl"
TEST_PATH = "data/raw/test.csv"
SAMPLE_PATH = "data/raw/sample_submission.csv"

OUTPUT_PATH = "results/submission.csv"


def main():

    print("=" * 50)
    print("Generating Kaggle Submission")
    print("=" * 50)

    # Load trained model
    model_data = joblib.load(MODEL_PATH)

    model = model_data["model"]
    features = model_data["features"]

    print("Loaded model:", model)

    # Load test data
    test_df = pd.read_csv(TEST_PATH)

    print("Test shape:", test_df.shape)

    # Apply feature engineering
    test_df = add_features(test_df)

    # Select features used during training
    X_test = test_df[features]

    # Predict
    predictions = model.predict(X_test)

    # Load sample submission
    submission = pd.read_csv(SAMPLE_PATH)

    # Replace yield column
    submission["yield"] = predictions

    # Save submission
    os.makedirs("results", exist_ok=True)

    submission.to_csv(OUTPUT_PATH, index=False)

    print("Submission saved to:", OUTPUT_PATH)

    print("=" * 50)


if __name__ == "__main__":
    main()