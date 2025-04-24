import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from utils.preprocessing import preprocess_data

import os

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Load dataset
df = pd.read_csv("../data/sample_data.csv")

# Preprocess
X, y = preprocess_data(df)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'model/readmission_model.pkl')
