### Project Overview

Winteq Predictive Maintenance is a web-based dashboard application built with Flask and Chart.js. It monitors industrial induction motors by tracking sensor telemetry, predicting failures using multiple trained machine learning models, and generating automated maintenance alerts.

---

### Key Features

- **Dashboard**: Displays real-time health scores, remaining useful life (RUL), and multi-parameter sensor trend analysis.
- **Motor Assets**: Provides a fleet-wide overview with filtering options and dynamic health progress bars.
- **Sensor Data**: Tracks live telemetry for temperature, vibration, voltage, and rotational speed per motor.
- **AI Alerts**: Logs automated diagnostic insights categorized by severity level, generated from real model predictions.
- **Reports**: Generates downloadable PDF maintenance reports with customizable filters (motor, fields).
- **Authentication**: Login/register system with admin and standard user roles.
- **Settings**: Allows customization of alert thresholds (admin only).

---

### Prerequisites

Python 3.10+ and:
```bash
pip install -r requirements.txt
```

---

### Project Structure

```text
predictive-maintenance/
│
├── app.py                              # Flask backend server
├── auth.py                             # Login/register/logout routes
├── models.py                           # Database models (User, Threshold)
├── predictive_maintenance_starter.ipynb  # Fault type + RUL model training notebook
├── astra_anomaly_detection.py          # Condition severity model training script
├── requirements.txt
├── datasets/
│   ├── client_training_dataset.csv     # Labeled dataset used for training
│   └── raw_sensor_new_data.csv         # Unlabeled dataset simulating live sensor input
├── models/                             # All trained model artifacts live here
│   ├── fault_classifier_model.pkl
│   ├── rul_regressor_model.pkl
│   ├── feature_scaler.pkl
│   ├── feature_columns.pkl
│   ├── fault_label_map.pkl
│   ├── condition_classifier.joblib
│   └── condition_scaler.joblib
├── templates/
│   ├── index.html
│   ├── login.html
│   └── register.html
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

2. Generate the model artifacts (only needed once, or after retraining):
   - Run all cells in `predictive_maintenance_starter.ipynb` to produce: `fault_classifier_model.pkl`, `rul_regressor_model.pkl`, `feature_scaler.pkl`, `feature_columns.pkl`, `fault_label_map.pkl`
   - Run `astra_anomaly_detection.py` to produce: `condition_classifier.joblib`, `condition_scaler.joblib` (saved to an `outputs/` folder — copy these two files into `models/` afterward)

3. Start the server:
```bash
python app.py
```

4. Open `http://127.0.0.1:5000/`. 

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
    ├── rul_regressor_model.pkl
    ├── feature_scaler.pkl
    ├── feature_columns.pkl
    ├── fault_label_map.pkl
    ├── condition_classifier.joblib
    └── condition_scaler.joblib
```

3. The application will automatically load these files when you run `app.py`.

---

### About the Models

This application uses **three independently trained models**, each with its own scaler and feature list:

| Model | Purpose | Output |
|-------|---------|--------|
| **Fault Classifier** | Predicts fault type | Fault category label |
| **RUL Regressor** | Estimates remaining useful life | RUL in hours |
| **Condition Severity** | Assesses overall health | Health score & severity level |

Each model is trained on different feature sets and uses its own dedicated scaler — they are never mixed during inference.