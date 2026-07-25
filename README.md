3. The application will automatically load these files when you run `app.py`.

---

### About the Models

This application uses **four independently trained models**, each with its own dedicated scaler and feature set — they are never mixed during inference:

| Model | Purpose | Output | Scaler used |
|-------|---------|--------|-------------|
| **Fault Classifier** | Predicts fault type | Fault category label | `feature_scaler.pkl` |
| **Condition Classifier** | Assesses severity | Normal / Warning / Critical / Failure | `condition_scaler.joblib` |
| **Health Score Model** | Condition-grounded health | Health score (0-100) | `condition_scaler_v2_with_current.joblib` |
| **RUL Model** | Estimates remaining useful life | RUL in hours | `condition_scaler_v2_with_current.joblib` |

Condition-Based Monitoring Alerts (threshold rules in `astra_config.py` / `AlarmRule` table) run independently of all ML models, so newly emerging sensor patterns not yet in training data can still be flagged, reviewed by an expert via the Data Training page, and fed back into retraining via `retrain_pipeline.py`. Retraining is gated by `DEPLOYMENT_GATE` in `retrain_pipeline.py` — a candidate model only replaces the live one if it meets the minimum F1/accuracy thresholds and does not regress against the currently deployed version.