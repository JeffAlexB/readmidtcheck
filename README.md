# 30-Day Hospital Readmission Risk Prediction App

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffAlexB/readmidtcheck/main?urlpath=%2Fdoc%2Ftree%2Fvoila%2Frender%2Fapp%2Fnotebook_interface.ipynb)
## Project Overview
This interactive machine learning application predicts the likelihood of a patient being readmitted to the hospital within 30 days. It uses a trained Random Forest model calibrated on synthetic healthcare data to provide:

- A **numerical risk score** (0.00 to 1.00)
- A corresponding **risk tier** (Low, Medium, High)

The interface was built using Jupyter Notebook and `ipywidgets`, and is designed for ease-of-use by clinicians or hospital staff.

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

### 1. Boxplot of Predicted Risk Score by Actual Readmission
![Boxplot Placeholder](https://via.placeholder.com/500x250?text=Boxplot+Risk+vs+Outcome)
- **What it shows**: Distribution of risk scores for patients grouped by true readmission outcome (0 = No, 1 = Yes).
- **Why it matters**: Helps understand model calibration and how scores separate between groups.

### 2. ROC Curve (Optional)
![ROC Curve Placeholder](https://via.placeholder.com/500x250?text=ROC+AUC+Curve)
- **What it shows**: Trade-off between sensitivity and specificity across thresholds.
- **Why it matters**: Provides a single score (AUC) to evaluate classifier performance.

### 3. Risk Tier Breakdown Heatmap
![Risk Tiers Placeholder](https://via.placeholder.com/500x250?text=Risk+Tier+Breakdown)
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
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JeffAlexB/readmidtcheck/main?urlpath=%2Fdoc%2Ftree%2Fvoila%2Frender%2Fapp%2Fnotebook_interface.ipynb)
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
