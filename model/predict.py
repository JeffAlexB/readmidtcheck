import pandas as pd
import joblib
from utils.preprocessing import preprocess_features


def predict_readmission(input_df):
    model = joblib.load('model/readmission_model.pkl')

    X, _ = preprocess_features(input_df)

    predictions = model.predict(X)
    return predictions
