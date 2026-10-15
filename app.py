"""
Week 4: AI Project Deployment & Capstone
Student: Binnala Esikha

A simple end-to-end ML app:
- Loads a sample student-results dataset
- Trains a Random Forest model
- Saves it with joblib
- Exposes a Flask prediction API

Run:
  pip install -r requirements.txt
  python train_model.py
  python app.py

Then visit http://127.0.0.1:5000/ in a browser.
"""
import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_pass_model.joblib")
DATA_PATH = os.path.join(BASE_DIR, "student_results.csv")

app = Flask(__name__)

def train_and_save_model():
    data = pd.read_csv(DATA_PATH)
    X = data[["study_hours", "attendance", "previous_score"]]
    y = data["passed"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_and_save_model()

@app.route("/", methods=["GET"])
def home():
    return render_template_string("""
    <!doctype html>
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Student Result Predictor</title>
    <style>
    body { font-family: Arial, sans-serif; max-width: 680px; margin: 32px auto; padding: 16px; }
    input { width: 100%; padding: 10px; margin: 6px 0 14px; box-sizing: border-box; }
    button { padding: 11px 16px; cursor: pointer; }
    .note { color: #555; font-size: 0.95rem; }
    </style></head><body>
    <h1>Student Result Predictor</h1>
    <p>Enter sample study information to get a demo prediction.</p>
    <form action="/predict-form" method="post">
      <label>Study hours per week</label><input name="study_hours" type="number" min="0" max="80" required>
      <label>Attendance percentage</label><input name="attendance" type="number" min="0" max="100" required>
      <label>Previous score (0–100)</label><input name="previous_score" type="number" min="0" max="100" required>
      <button type="submit">Predict</button>
    </form>
    <p class="note">Educational demo using synthetic sample data; predictions are not official academic results.</p>
    </body></html>
    """)

@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or {}
    required = ["study_hours", "attendance", "previous_score"]
    missing = [key for key in required if key not in payload]
    if missing:
        return jsonify({"error": "Missing fields", "missing": missing}), 400
    try:
        values = [[float(payload["study_hours"]), float(payload["attendance"]),
                   float(payload["previous_score"])]]
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric"}), 400
    prediction = int(load_or_train_model().predict(values)[0])
    label = "Likely to pass" if prediction == 1 else "May need additional support"
    return jsonify({"prediction": prediction, "result": label})

@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        values = [[float(request.form["study_hours"]), float(request.form["attendance"]),
                   float(request.form["previous_score"])]]
    except (KeyError, TypeError, ValueError):
        return "Please enter valid numeric values.", 400
    prediction = int(load_or_train_model().predict(values)[0])
    label = "Likely to pass" if prediction == 1 else "May need additional support"
    return render_template_string("""
      <html><head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
      <body style="font-family:Arial,sans-serif;margin:32px auto;max-width:680px;padding:16px">
      <h1>Prediction Result</h1><p><strong>{{ label }}</strong></p>
      <p class="note">This is only an educational demonstration based on synthetic data.</p>
      <a href="/">Try another prediction</a></body></html>
    """, label=label)

if __name__ == "__main__":
    # For local learning only; use a production WSGI server for real deployments.
    app.run(host="127.0.0.1", port=5000, debug=False)
