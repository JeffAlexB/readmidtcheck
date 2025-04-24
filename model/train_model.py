# # Hospital Readmission Prediction Notebook
# Phase 1: Jupyter-based Interactive Application

# ## 1. Imports & Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ## 2. Load Data

df = pd.read_csv("data/sample_data.csv")
df.head()

# ## 3. Feature Engineering
le = LabelEncoder()
for col in ['gender', 'primary_diagnosis', 'discharge_to']:
    df[col] = le.fit_transform(df[col].astype(str))

df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 65, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
df['stay_group'] = pd.cut(df['days_in_hospital'], bins=[0, 3, 7, 14, 100], labels=[0, 1, 2, 3]).astype(int)
df['comorbidity_group'] = pd.cut(df['comorbidity_score'], bins=[-1, 0, 2, 10], labels=[0, 1, 2]).astype(int)

# New clinical features
df['elderly_flag'] = (df['age'] >= 75).astype(int)
df['short_stay'] = (df['days_in_hospital'] <= 3).astype(int)
df['complex_case'] = (df['comorbidity_score'] >= 3).astype(int)
df['facility_flag'] = df['discharge_to'].isin([1, 2]).astype(int)
df['proc_density'] = df['num_procedures'] / (df['days_in_hospital'] + 1e-6)
df['comorb_procedures'] = df['comorbidity_score'] * df['num_procedures']
df['complex_density'] = df['comorbidity_score'] * df['proc_density']

X = df.drop(columns=['readmitted', 'age', 'days_in_hospital', 'comorbidity_score'])
y = df['readmitted']

# ## 4. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cal_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
cal_rf.fit(X_resampled, y_resampled)

# ## 5. Evaluate Model
y_scores = cal_rf.predict_proba(X_test)[:, 1]
y_pred = cal_rf.predict(X_test)
print(classification_report(y_test, y_pred))

# ## 6. Visualizations
# ROC Threshold Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.axvline(0.85, color='red', linestyle='--', label='High-Risk Threshold')
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision/Recall vs Threshold")
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance
importances = rf.feature_importances_
features = X.columns
imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ## 7. Risk Tiering Function
def assign_risk_tier(score):
    if score > 0.85:
        return 'High'
    elif score > 0.5:
        return 'Medium'
    else:
        return 'Low'

# ## 8. Single-Patient Interface
# Sample interface - input a new patient's data
new_patient = {
    'gender': 'Male',
    'primary_diagnosis': 'COPD',
    'num_procedures': 2,
    'discharge_to': 'Home',
    'age': 70,
    'days_in_hospital': 5,
    'comorbidity_score': 2
}

# Process input
input_df = pd.DataFrame([new_patient])
for col in ['gender', 'primary_diagnosis', 'discharge_to']:
    input_df[col] = le.transform(input_df[col].astype(str))

input_df['age_group'] = pd.cut(input_df['age'], bins=[0, 40, 55, 65, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
input_df['stay_group'] = pd.cut(input_df['days_in_hospital'], bins=[0, 3, 7, 14, 100], labels=[0, 1, 2, 3]).astype(int)
input_df['comorbidity_group'] = pd.cut(input_df['comorbidity_score'], bins=[-1, 0, 2, 10], labels=[0, 1, 2]).astype(int)

input_df['elderly_flag'] = (input_df['age'] >= 75).astype(int)
input_df['short_stay'] = (input_df['days_in_hospital'] <= 3).astype(int)
input_df['complex_case'] = (input_df['comorbidity_score'] >= 3).astype(int)
input_df['facility_flag'] = input_df['discharge_to'].isin([1, 2]).astype(int)
input_df['proc_density'] = input_df['num_procedures'] / (input_df['days_in_hospital'] + 1e-6)
input_df['comorb_procedures'] = input_df['comorbidity_score'] * input_df['num_procedures']
input_df['complex_density'] = input_df['comorbidity_score'] * input_df['proc_density']

input_final = input_df[X.columns]
y_pred_score = cal_rf.predict_proba(input_final)[0][1]
y_pred_tier = assign_risk_tier(y_pred_score)

print(f"Predicted Readmission Risk Score: {y_pred_score:.2f}")
print(f"Predicted Risk Tier: {y_pred_tier}")

# ## 9. Save Model for Later Use
joblib.dump(cal_rf, "model/readmission_rf_model.pkl")