# 30-Day Hospital Readmission Risk Prediction App
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffAlexB/readmidtcheck/HEAD?urlpath=voila/render/app/notebook_interface.ipynb&fresh=true)

## Intended Audience
This tool is designed for clinicians, case managers, or hospital staff who need a fast, interpretable way to identify patients at higher risk of 30-day hospital readmission and prioritize follow-up care.

---
## Project Overview
This interactive machine learning application predicts the likelihood of a patient being readmitted to the hospital within 30 days. It uses a trained Random Forest model calibrated on synthetic healthcare data to provide:

- A **numerical risk score** (0.00 to 1.00)
- A corresponding **risk tier** (Low, Medium, High)

The interface was built using Jupyter Notebook and `ipywidgets`, and is designed for ease-of-use by clinicians or hospital staff.

## Table of Contents
- [Purpose](#purpose)
- [Features](#features)
- [Visualizations](#visualizations)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run-locally)
- [Troubleshooting Notes](#troubleshooting-notes)
- [Evaluation Summary](#evaluation-summary)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Purpose

Hospital readmissions are costly and often preventable. This tool provides a **decision-support aid** to help healthcare professionals identify patients at risk of early readmission and improve follow-up care planning.

---

## Features

- Input form with sliders and dropdowns for patient info (age, comorbidities, procedures, etc.)
- Prediction output: risk score and tier label
- Data visualizations (see below)
- Synthetic test cases for model behavior analysis
- Final summary + discussion of results, limitations, and future work

---

## Visualizations
Below are examples of the model evaluation charts produced by the application. These visualizations help explain the model's behavior and effectiveness to users.

### 1. ROC Curve
![ROC Curve Placeholder](https://github.com/JeffAlexB/readmidtcheck/blob/main/kaggle/visualizations/ROC_curve.png)
- **What it shows**: Trade-off between sensitivity and specificity across thresholds.
- **Why it matters**: Provides a single score (AUC) to evaluate classifier performance.
### 2. Feature Importance (Permutation)
![Feature Importance Placeholder](https://github.com/JeffAlexB/readmidtcheck/blob/main/kaggle/visualizations/prem_features.png)
- **What it shows**: Highlights which patient features (e.g., age, diagnosis, hospital stay length) had the strongest impact on the model’s predictions.
- **Why it matters**: Helps clinicians understand what factors drive readmission risk, supporting model transparency and potential clinical action.
### 3. Boxplot of Predicted Risk Score by Actual Readmission
![Boxplot Placeholder](https://github.com/JeffAlexB/readmidtcheck/blob/main/kaggle/visualizations/risk_score.png)
- **What it shows**: Distribution of risk scores for patients grouped by true readmission outcome (0 = No, 1 = Yes).
- **Why it matters**: Helps understand model calibration and how scores separate between groups.
### 4. Risk Tier Breakdown Heatmap
![Risk Tiers Placeholder](https://github.com/JeffAlexB/readmidtcheck/blob/main/kaggle/visualizations/risktier_heatmap.png)
- **What it shows**: Percentage of true readmissions within each assigned tier.
- **Why it matters**: Indicates how well the model isolates high-risk patients.

---

## Technologies Used

- Python
- scikit-learn
- pandas
- matplotlib / seaborn
- ipywidgets
- Voilà
- Binder

---

## How to Run (Locally)

1. Clone the repository
2. Set up a Python environment (e.g., `venv` or `conda`)
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter Notebook and run `app/notebook_interface.ipynb`

OR launch directly using Binder:  
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffAlexB/readmidtcheck/HEAD?urlpath=voila/render/app/notebook_interface.ipynb&fresh=true)

### UI and Styling
The final application uses Voilà for deployment with optional use of the Vuetify template, providing a clean, responsive web-app appearance suitable for clinical settings.


---

## Troubleshooting Notes

> If you see a "GitHub refused to connect" message or widgets not responding:
> - Open the Binder/Voilà link in a new tab.
> - Wait 1–2 minutes on first launch (Binder builds the environment).
> - If using classic Jupyter interface, widgets may need re-running from the top.

---

## Evaluation Summary

- ROC AUC: ~0.70–0.88
- High tier readmission rate: ~28.6%
- Model tested against synthetic edge cases
- Visual and interactive feedback for predictions

---

## Limitations

- Model trained on synthetic/limited data — not yet real-world validated
- High false positives in some clinical edge cases (e.g., elderly with many comorbidities)
- Feature weighting and score calibration could be improved for production

---

## Future Improvements

- Integrate patient history and lab values
- Add post-discharge follow-up planning tools
- Improve UI for tablet/clinical settings
- Log user input for retraining / continuous learning

---

## License
MIT License 
