import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score, precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ========== 1. LOAD & PREPROCESS DATA ==========
df = pd.read_csv("../data/sample_data.csv")

# Encode categorical variables
le = LabelEncoder()
for col in ['gender', 'primary_diagnosis', 'discharge_to']:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature Engineering
# Bins and groupings
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
df['stay_group'] = pd.cut(df['days_in_hospital'], bins=[0, 3, 7, 14, 100], labels=[0, 1, 2, 3]).astype(int)
df['comorbidity_group'] = pd.cut(df['comorbidity_score'], bins=[-1, 0, 2, 10], labels=[0, 1, 2]).astype(int)

# New diagnostic features
df['elderly_flag'] = (df['age'] >= 75).astype(int)
df['short_stay'] = (df['days_in_hospital'] <= 3).astype(int)
df['complex_case'] = (df['comorbidity_score'] >= 3).astype(int)
df['facility_flag'] = df['discharge_to'].isin([1, 2]).astype(int)
df['proc_density'] = df['num_procedures'] / (df['days_in_hospital'] + 1e-6)
df['comorb_procedures'] = df['comorbidity_score'] * df['num_procedures']
df['complex_density'] = df['comorbidity_score'] * df['proc_density']

# Drop raw columns
X = df.drop(columns=['readmitted', 'age', 'days_in_hospital', 'comorbidity_score'])
y = df['readmitted']

# ========== 2. TRAIN/TEST SPLIT + SMOTE ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ========== 3. TRAIN & CALIBRATE MODELS ==========
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
logreg.fit(X_resampled, y_resampled)
rf.fit(X_resampled, y_resampled)

# Calibrate Random Forest
calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
calibrated_rf.fit(X_resampled, y_resampled)

# ========== 4. EVALUATE MODELS ==========
y_pred_logreg = logreg.predict(X_test)
y_pred_rf = calibrated_rf.predict(X_test)
report_logreg = classification_report(y_test, y_pred_logreg, output_dict=True)
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)

# Combine results into a DataFrame
metrics = ['precision', 'recall', 'f1-score']
classes = ['0', '1']
comparison = pd.DataFrame({
    'Model': ['Logistic Regression'] * 2 + ['Random Forest'] * 2,
    'Class': ['0', '1'] * 2,
    'Precision': [report_logreg[c]['precision'] for c in classes] + [report_rf[c]['precision'] for c in classes],
    'Recall': [report_logreg[c]['recall'] for c in classes] + [report_rf[c]['recall'] for c in classes],
    'F1-score': [report_logreg[c]['f1-score'] for c in classes] + [report_rf[c]['f1-score'] for c in classes],
})

# ========== 5. VISUALIZE COMPARISON ==========
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, metric in enumerate(metrics):
    sns.barplot(ax=axes[i], data=comparison, x='Class', y=metric.capitalize(), hue='Model')
    axes[i].set_title(f"{metric.capitalize()} Comparison by Model")
    axes[i].set_ylim(0, 1)
    axes[i].legend(loc='lower right')
plt.tight_layout()
plt.show()

# ========== 6. RISK TIERING WITH CUSTOM THRESHOLD ==========
y_scores = calibrated_rf.predict_proba(X_test)[:, 1]

# Use fixed thresholds instead of quantiles for better control
def assign_risk_tier(score):
    if score > 0.85:
        return 'High'
    elif score > 0.50:
        return 'Medium'
    else:
        return 'Low'

df_results = X_test.copy()
df_results['readmission_probability'] = y_scores
df_results['risk_tier'] = df_results['readmission_probability'].apply(assign_risk_tier)
df_results['actual_readmission'] = y_test.values

# Save and report
df_results.to_csv("output/risk_tiered_predictions.csv", index=False)
print("\n[✓] Risk Tier Breakdown (Readmission Rates):")
print(df_results.groupby('risk_tier')['actual_readmission'].value_counts(normalize=True).unstack().fillna(0))

sns.countplot(data=df_results, x='risk_tier', hue='actual_readmission')
plt.title("Readmission Distribution by Risk Tier")
plt.xlabel("Predicted Risk Tier")
plt.ylabel("Patient Count")
plt.tight_layout()
plt.show()

# ========== 7. PRECISION/RECALL THRESHOLD PLOT ==========
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.axvline(0.85, color='red', linestyle='--', label='High Risk Cutoff')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========== 8. FEATURE IMPORTANCE ==========
importances = rf.feature_importances_
feature_names = X.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='Importance', y='Feature')
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.show()

# ========== 9. SAVE BEST MODEL ==========
os.makedirs("model", exist_ok=True)
joblib.dump(calibrated_rf, "model/readmission_rf_model.pkl")