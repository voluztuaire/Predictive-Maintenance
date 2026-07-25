```markdown
### Project Overview

Winteq Predictive Maintenance is a web-based dashboard application built with Flask and Chart.js. It monitors industrial induction motors by tracking sensor telemetry, predicting failures using multiple trained machine learning models, and generating automated maintenance alerts.

---

### Key Features

- **Dashboard**: Displays real-time health scores, remaining useful life (RUL), failure probability, and multi-parameter sensor trend analysis.
- **Forecast**: Projects historical vs forecasted sensor values (temperature, vibration, voltage, current, RPM) 48h ahead per motor.
- **Motor Assets**: Provides a fleet-wide overview with filtering options and dynamic health progress bars.
- **Sensor Data**: Tracks live telemetry for temperature, vibration, voltage, current, and rotational speed per motor.
- **AI Alerts**: Logs automated diagnostic insights categorized by severity level, generated from real model predictions.
- **Condition Alerts**: Independent, non-ML threshold-based monitoring that flags sensor readings the AI model may not yet recognize, used to build new training data.
- **Data Training (admin only)**: Expert Review Queue to approve/reject condition alerts as new labeled training data, plus a Retrain button with gated model deployment and retraining history.
- **Reports**: Generates downloadable PDF maintenance reports with customizable motor and field selection.
- **Maintenance Assistant Chatbot**: In-app chat widget backed by a local Ollama LLM, grounded with live fleet/motor data.
- **Authentication**: Login/register/forgot-password system with admin and standard user roles.
- **Settings (admin only)**: Customization of alert thresholds and alarm rules.
- **Light/Dark mode** and mobile-responsive layout.

---

### Prerequisites

Python 3.10+ and:
```
pip install -r requirements.txt
```

On Windows, `lightgbm` and `xgboost` may require the Microsoft Visual C++ Build Tools if a prebuilt wheel isn't available for your Python version.

For the chatbot widget, also install Ollama (https://ollama.com/download) and pull a model:
```
ollama pull llama3.2:1b
```
If you use a different model, update `DEFAULT_MODEL` in `chatbot_llm.py` to match — the app does not automatically fall back to another installed model when running through Flask.

---

### Project Structure

```
predictive-maintenance/
│
├── app.py                              # Flask backend server & API routes
├── auth.py                             # Login/register/forgot-password/logout routes
├── models.py                           # Database models (User, AlarmRule)
├── astra_anomaly_detection.py          # Condition/fault model training script (source of truth)
├── astra_config.py                     # Shared constants/pure functions (no side effects)
├── feature_utils.py                    # Shared feature engineering for RUL/Health models
├── forecast_engine.py                  # Holt-Winters sensor forecasting + RUL/Health projection
├── expert_validation.py                # Threshold alert -> review queue -> labeled training data
├── retrain_pipeline.py                 # Versioned retraining with gated deployment
├── chatbot_llm.py                      # Ollama-based maintenance assistant chatbot
├── requirements.txt
├── datasets/
│   ├── client_training_dataset.csv     # Labeled dataset used for training
│   └── raw_sensor_new_data.csv         # Unlabeled dataset simulating live sensor input
├── models/                             # All trained model artifacts live here
│   ├── fault_classifier_model.pkl
│   ├── feature_scaler.pkl
│   ├── feature_columns.pkl
│   ├── fault_label_map.pkl
│   ├── condition_classifier.joblib
│   ├── condition_scaler.joblib
│   ├── condition_scaler_v2_with_current.joblib
│   ├── rul_iso.joblib
│   ├── health_iso.joblib
│   ├── registry.json                   # Deployed model version pointer (auto-created on first run)
│   └── vN/                             # Versioned candidate/deployed models from retraining
├── expert_validation_data/             # Review queue + expert-approved rows (auto-created on first run)
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── forgot_password.html
└── static/
    ├── css/style.css
    └── js/main.js
```

`models/registry.json` and `expert_validation_data/` do not need to exist beforehand — they are created automatically the first time the app runs (via `_ensure_registry()` in `retrain_pipeline.py` and `_ensure_storage()` in `expert_validation.py`).

---

### How to Run

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Make sure `datasets/client_training_dataset.csv` and `datasets/raw_sensor_new_data.csv` are present — `app.py` loads `raw_sensor_new_data.csv` at startup and will fail to launch without it.

3. Make sure the model artifacts described in Model Setup below are in `models/`.

4. Start the server:
```
python app.py
```

5. Open `http://127.0.0.1:5000/`.

6. (Optional) Start Ollama to enable the chatbot widget:
```
ollama serve
```

---

### Model Setup

The trained model artifacts are not included in this repository due to file size limitations.

To set up the models:

1. Download all model files from this Google Drive folder:
https://drive.google.com/drive/folders/10mtiQiR-3vHIzjPZhIe3paoIe2K2AvcI?usp=sharing

2. Place all downloaded files into the `models/` folder at the root of the project:
```
predictive-maintenance/
└── models/
    ├── fault_classifier_model.pkl
    ├── feature_scaler.pkl
    ├── feature_columns.pkl
    ├── fault_label_map.pkl
    ├── condition_classifier.joblib
    ├── condition_scaler.joblib
    ├── condition_scaler_v2_with_current.joblib
    ├── rul_iso.joblib
    └── health_iso.joblib
```

3. The application will automatically load these files when you run `app.py`.

---

### About the Models

This application uses four independently trained models, each with its own dedicated scaler and feature set — they are never mixed during inference:

| Model | Purpose | Output | Scaler used |
|-------|---------|--------|-------------|
| Fault Classifier | Predicts fault type | Fault category label | feature_scaler.pkl |
| Condition Classifier | Assesses severity | Normal / Warning / Critical / Failure | condition_scaler.joblib |
| Health Score Model | Condition-grounded health | Health score (0-100) | condition_scaler_v2_with_current.joblib |
| RUL Model | Estimates remaining useful life | RUL in hours | condition_scaler_v2_with_current.joblib |

Condition-Based Monitoring Alerts (threshold rules in `astra_config.py` / `AlarmRule` table) run independently of all ML models, so newly emerging sensor patterns not yet in training data can still be flagged, reviewed by an expert via the Data Training page, and fed back into retraining via `retrain_pipeline.py`. Retraining is gated by `DEPLOYMENT_GATE` in `retrain_pipeline.py` — a candidate model only replaces the live one if it meets the minimum F1/accuracy thresholds and does not regress against the currently deployed version.
```