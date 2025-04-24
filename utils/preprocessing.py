# === utils/preprocessing.py ===
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Mirrors the preprocessing in train_model.py
def preprocess_features(df):
    df = df.copy()

    le = LabelEncoder()
    for col in ['gender', 'primary_diagnosis', 'discharge_to']:
        df[col] = le.fit_transform(df[col].astype(str))

    # Derived Features
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['stay_group'] = pd.cut(df['days_in_hospital'], bins=[0, 3, 7, 14, 100], labels=[0, 1, 2, 3]).astype(int)
    df['comorbidity_group'] = pd.cut(df['comorbidity_score'], bins=[-1, 0, 2, 10], labels=[0, 1, 2]).astype(int)

    # Flags and Interactions
    df['elderly_flag'] = (df['age'] >= 75).astype(int)
    df['short_stay'] = (df['days_in_hospital'] <= 3).astype(int)
    df['complex_case'] = (df['comorbidity_score'] >= 3).astype(int)
    df['facility_flag'] = df['discharge_to'].isin([1, 2]).astype(int)
    df['proc_density'] = df['num_procedures'] / (df['days_in_hospital'] + 1e-6)
    df['comorb_procedures'] = df['comorbidity_score'] * df['num_procedures']
    df['complex_density'] = df['comorbidity_score'] * df['proc_density']

    # Drop raw columns if needed to match training data
    return df.drop(columns=['age', 'days_in_hospital', 'comorbidity_score'])
