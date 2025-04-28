# Changesheet - 30-Day Hospital Readmission Risk Prediction App

---

## Project Initialization
**Date:** April 19, 2025  
**Tasks Completed:**
- Set up project repository structure (`/app`, `/model`, `/utils`, `/data`).
- Installed necessary libraries (scikit-learn, pandas, matplotlib, ipywidgets, voila).
- Prepared synthetic dataset (`sample_data.csv`) for initial development.

---

## Data Preparation & Preprocessing
**Date:** April 21, 2025  
**Tasks Completed:**
- Developed `preprocessing.py` script for feature engineering.
- Implemented handling for categorical features, binning, and derived risk factors.
- Verified preprocessing through manual inspection of feature outputs.

---

## Model Development
**Date:** April 22, 2025  
**Tasks Completed:**
- Trained initial Random Forest model.
- Evaluated performance using ROC AUC, precision, recall, and F1-score metrics.
- Applied SMOTE balancing for minority class improvement.

---

## Hyperparameter Tuning and Model Optimization
**Date:** April 24, 2025  
**Tasks Completed:**
- Conducted randomized search and grid search for hyperparameter tuning.
- Identified best hyperparameters achieving ROC AUC > 0.88.
- Saved final model as `readmission_rf_model_tuned.pkl`.

---

## Application Interface Development
**Date:** April 25, 2025  
**Tasks Completed:**
- Built Jupyter Notebook interface (`notebook_interface.ipynb`) using `ipywidgets`.
- Integrated risk score prediction and tier assignment.
- Added interactive predict button and user-friendly layout.

---

## Visualization and Validation
**Date:** April 26, 2025  
**Tasks Completed:**
- Implemented 4 visualizations: ROC curve, boxplot, risk tier heatmap, feature importance.
- Validated application outputs against expected model behavior.
- Confirmed ROC AUC results matched tuned model outputs.

---

## Deployment Setup
**Date:** April 26, 2025  
**Tasks Completed:**
- Configured Voilà for frontend display.
- Prepared Binder deployment settings.
- Tested application access via Binder and verified functionality.

---

## Documentation and Finalization
**Date:** April 29, 2025  
**Tasks Completed:**
- Created README.md with detailed usage instructions and visualization explanations.
- Wrote User Guide and full project documentation (Part A–D).
- Completed changesheet and unit test coverage for preprocessing functions.

---
