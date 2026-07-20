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
- **Settings (admin only)**: Customization of alert thresholds and AI model configuration.
- **Light/Dark mode** and mobile-responsive layout.

---

### Prerequisites

Python 3.10+ and:
```bash
pip install -r requirements.txt
```

For the chatbot widget, also install [Ollama](https://ollama.com/download) and pull a model:
```bash
ollama pull llama3.2:1b
```

---

### Project Structure

```text
predictive-maintenance/
│
├── app.py                              # Flask backend server & API routes
├── auth.py                             # Login/register/forgot-password/logout routes
├── models.py                           # Database models (User, Threshold)
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
│   └── registry.json                   # Deployed model version pointer (retrain_pipeline.py)
├── expert_validation_data/             # Review queue + expert-approved labeled rows
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   └── forgot_password.html
└── static/
    ├── css/style.css
    └── js/main.js
```

---

### How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model artifacts described in **Model Setup** below are in `models/`.

3. Start the server:
```bash
python app.py
```

4. Open `http://127.0.0.1:5000/`.

5. (Optional) Start Ollama to enable the chatbot widget:
```bash
ollama serve
```

---

### Model Setup

The trained model artifacts are **not included** in this repository due to file size limitations.

**To set up the models:**

1. Download all model files from this Google Drive folder:  
   [https://drive.google.com/drive/folders/10mtiQiR-3vHIzjPZhIe3paoIe2K2AvcI?usp=sharing](https://drive.google.com/drive/folders/10mtiQiR-3vHIzjPZhIe3paoIe2K2AvcI?usp=sharing)

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

This application uses **four independently trained models**, each with its own scaler and feature list:

| Model | Purpose | Output |
|-------|---------|--------|
| **Fault Classifier** | Predicts fault type | Fault category label |
| **Condition Classifier** | Assesses severity | Normal / Warning / Critical / Failure |
| **Health Score Model** | Condition-grounded health | Health score (0-100) |
| **RUL Model** | Estimates remaining useful life | RUL in hours |

Each model is trained on different feature sets and uses its own dedicated scaler — they are never mixed during inference. Condition-Based Monitoring Alerts (threshold rules) run independently of all ML models, so newly emerging sensor patterns not yet in training data can still be flagged, reviewed by an expert, and fed back into retraining via `retrain_pipeline.py`.