# src/inference.py

import joblib
import pandas as pd

MODEL_PATH = "models/lightgbm_model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)

def predict_failure(input_df):
    input_df = input_df[features]
    probability = model.predict_proba(input_df)[:, 1]
    return probability

if __name__ == "__main__":
    print("Inference module ready.")