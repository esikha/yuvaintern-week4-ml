# Week 4: AI Project Deployment & Capstone

**Student:** Binnala Esikha

## Project: Student Result Predictor
This end-to-end educational demo trains a Random Forest classifier on a small synthetic dataset and serves predictions through a Flask web app and JSON API.

## Files
- `app.py` — Flask web page and `/predict` JSON API
- `train_model.py` — trains and serializes the model with joblib
- `student_results.csv` — synthetic sample data
- `requirements.txt` — required Python packages
- `Week4_Report.docx` — project report

## Run locally
1. Install Python 3.
2. Install packages: `pip install -r requirements.txt`
3. Train/save the model: `python train_model.py`
4. Start the app: `python app.py`
5. Open `http://127.0.0.1:5000/` in a browser.

## API example
Send a POST request to `/predict` with JSON:
`{"study_hours": 6, "attendance": 80, "previous_score": 72}`

The response contains a demo prediction and a short label.

## Notes
The dataset is synthetic and only for educational demonstration. The prediction is not an official academic outcome. The Flask server binds to localhost and is not publicly deployed. For an internet-facing deployment, use a production WSGI server and follow the hosting provider's security guidance.
