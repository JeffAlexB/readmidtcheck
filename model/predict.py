# model/predict.py

import joblib
import pandas as pd
from utils.preprocessing import preprocess_features

# Load the tuned Random Forest model
model = joblib.load("../model/model/readmission_rf_model_tuned.pkl")

def predict_risk(input_data: pd.DataFrame):
    """
    Takes a raw input dataframe with columns matching the input form,
    preprocesses it, and returns readmission probability.
    """
    processed_data = preprocess_features(input_data)
    probability = model.predict_proba(processed_data)[:, 1][0]
    return probability
