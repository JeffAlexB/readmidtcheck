import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    df = df.copy()

    # Encode all object (categorical) columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Optional: save encoders for inverse_transform if needed

    # Handle missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']

    return X, y
