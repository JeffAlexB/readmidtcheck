# train_model.py
# Imports
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from utils.preprocessing import preprocess_features

# Load and Prepare Data
df = pd.read_csv("../data/sample_data.csv")
X_raw = df.drop(columns=["readmitted"])
y = df["readmitted"]

X = preprocess_features(X_raw)

# Handle Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled,
    test_size=0.3,
    random_state=42,
    stratify=y_resampled
)

# Define Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize Random Forest
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

# Set Up Grid Search
grid_search = GridSearchCV(
    rf,
    param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit the Model
grid_search.fit(X_train, y_train)

# Select Best Model
best_model = grid_search.best_estimator_

# Evaluate on Test Set
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n=== Final Evaluation ===")
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Test ROC AUC Score: {auc_score:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))

# Save Tuned Model
joblib.dump(best_model, "../model/model/readmission_rf_model_tuned.pkl")
print("\n[✓] Tuned model saved as readmission_rf_model_tuned.pkl")

# Generate New Risk Tiered Predictions
print("\n=== Generating Updated Risk Tiered Predictions ===")

df_raw = pd.read_csv("../data/sample_data.csv")
X_raw = df_raw.drop(columns=["readmitted"])
y_raw = df_raw["readmitted"]
X_processed = preprocess_features(X_raw)

# Predict probabilities
readmission_probs = best_model.predict_proba(X_processed)[:, 1]

# Assign risk tiers
risk_tiers = pd.cut(
    readmission_probs,
    bins=[-np.inf, 0.4, 0.6, np.inf],
    labels=["Low", "Medium", "High"]
)

# Create and Save Output
df_results = df_raw.copy()
df_results["readmission_probability"] = readmission_probs
df_results["risk_tier"] = risk_tiers
df_results["actual_readmission"] = y_raw

df_results.to_csv("../model/output/risk_tiered_predictions.csv", index=False)

print("[✓] New risk_tiered_predictions.csv generated and saved.")

# Final Confirmation
print("\n=== Training and Prediction Generation Complete! ===")
