"""Train and save the student-result demonstration model."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, "student_results.csv"))
X = data[["study_hours", "attendance", "previous_score"]]
y = data["passed"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, os.path.join(BASE_DIR, "student_pass_model.joblib"))
print("Model trained and saved to student_pass_model.joblib")
