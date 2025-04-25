import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance

# Load results if not already loaded
df_results = pd.read_csv("../model/output/risk_tiered_predictions.csv")

display(Markdown("""
# Model Evaluation Visualizations

**Visualizations:**
- [ ROC Curve](#roc-curve)
- [ Feature Importance](#feature-importance)
- [ Risk Score Distribution](#risk-score-distribution)
- [ Risk Tier Breakdown](#risk-tier-breakdown)

---

"""))


# ROC Curve ---
# Shows model's ability to distinguish readmitted vs. non-readmitted patients
display(Markdown("""
### Visualization: ROC Curve
This curve shows the model’s ability to distinguish between patients who will and won't be readmitted.
Higher AUC (Area Under Curve) means better model skill.
"""))
fpr, tpr, thresholds = roc_curve(df_results['actual_readmission'], df_results['readmission_probability'])
auc_score = roc_auc_score(df_results['actual_readmission'], df_results['readmission_probability'])

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: 30-Day Readmission Prediction')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
display(Markdown("""
**Why this matters:**
If the model reliably separates high-risk from low-risk patients, clinicians can better decide who needs extra follow-up or interventions after discharge.
"""))

# Permutation Feature Importance ---
# Shows which features had the strongest impact on the prediction model
display(Markdown("""
### Visualization: Feature Importance
This chart shows which patient features had the greatest impact on the model's predictions.
"""))
df_raw = pd.read_csv("../data/sample_data.csv")
X_raw = df_raw.drop(columns=["readmitted"])
y_raw = df_raw["readmitted"]
X_processed = preprocess_features(X_raw)

perm = permutation_importance(model, X_processed, y_raw, n_repeats=5, random_state=42)

importance_df = pd.DataFrame({
    'Feature': X_processed.columns,
    'Importance': perm.importances_mean
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Estimated Feature Importance via Permutation")
plt.xlabel("Mean Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
display(Markdown("""
**Why this matters:**
Understanding which factors (e.g., age, diagnosis) influence readmission risk helps clinicians validate the model and focus interventions more effectively.
"""))

# Boxplot of Predicted Risk Scores by Actual Readmission ---
# Shows distribution of predicted scores for patients grouped by true outcomes
display(Markdown("""
### Visualization: Risk Score Distribution
This boxplot shows how predicted risk scores differ between patients who were and were not readmitted.
"""))
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_results, x='actual_readmission', y='readmission_probability')
plt.title('Predicted Risk Score by Actual Readmission Status')
plt.xlabel('Actual Readmission (0 = No, 1 = Yes)')
plt.ylabel('Predicted Risk Score')
plt.grid(True)
plt.tight_layout()
plt.show()
display(Markdown("""
### Visualization: Risk Tier Breakdown
This chart shows how many patients in each risk tier (Low, Medium, High) were actually readmitted or not.
"""))

# Countplot of Risk Tiers vs Actual Readmissions ---
# Compares how well risk tiers separate patients who were actually readmitted
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df_results, x='risk_tier', hue='actual_readmission')
plt.title("True Readmissions by Predicted Risk Tier")
plt.xlabel("Predicted Risk Tier")
plt.ylabel("Number of Patients")
plt.legend(title="Actual Readmission")
plt.tight_layout()
plt.grid(True)
plt.show()
display(Markdown("""
**Why this matters:**
A clear separation where more readmissions are concentrated in higher tiers validates that the risk scoring system aligns with real patient outcomes.
"""))
