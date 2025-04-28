# test_preprocessing.py

import pandas as pd
from utils.preprocessing import preprocess_features

def test_preprocess_features_output_columns():
    # Minimal test input
    data = {
        'age': [50],
        'gender': ['Male'],
        'primary_diagnosis': ['Heart Disease'],
        'num_procedures': [1],
        'days_in_hospital': [2],
        'comorbidity_score': [2],
        'discharge_to': ['Home']
    }
    df = pd.DataFrame(data)
    processed = preprocess_features(df)

    # Check that new derived features exist
    assert 'age_group' in processed.columns
    assert 'stay_group' in processed.columns
    assert 'comorbidity_group' in processed.columns
    assert 'complex_case' in processed.columns

def test_preprocess_features_output_type():
    data = {
        'age': [30],
        'gender': ['Female'],
        'primary_diagnosis': ['Diabetes'],
        'num_procedures': [2],
        'days_in_hospital': [5],
        'comorbidity_score': [1],
        'discharge_to': ['Home Health Care']
    }
    df = pd.DataFrame(data)
    processed = preprocess_features(df)

    # Check that output is still a DataFrame
    assert isinstance(processed, pd.DataFrame)
