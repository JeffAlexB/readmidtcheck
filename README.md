# 30-Day Hospital Readmission Risk Prediction App

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/readmission-risk-app/HEAD?urlpath=voila%2Frender%2Fapp%2Fnotebook_interface.ipynb)

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
- Data visualizations:
  - Predicted score distribution by outcome
  - Risk tier breakdown
  - Feature importance analysis
- Synthetic test cases for model behavior analysis
- Final summary + discussion of results, limitations, and future work

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
4. Launch Jupyter Notebook and run app/notebook_interface.ipynb 

OR launch directly using Binder:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/readmission-risk-app/HEAD?urlpath=voila%2Frender%2Fapp%2Fnotebook_interface.ipynb)

---

## License

MIT License
