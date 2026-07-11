from flask import Flask, render_template, jsonify, request
from flask_login import LoginManager, login_required, current_user
import pandas as pd
import numpy as np
import joblib
import math
import os

from models import db, User
from auth import auth_bp

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-to-a-random-secret-string"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

app.register_blueprint(auth_bp)

MODEL_DIR = "models"

# The 10 features based on the new dataset
FEATURE_COLS = [
    "Voltage_L1", "Voltage_L2", "Voltage_L3", "Frequency",
    "Power_Factor", "Temperature", "Vibration_X", "Vibration_Y",
    "Vibration_Z", "Rotational_Speed"
]

RUL_DISPLAY_CAP = 5000

# Load raw incoming data from the datasets folder
df = pd.read_csv("datasets/raw_sensor_new_data.csv")

# Rename columns to match the existing logic format
df.rename(columns={"Timestamp": "timestamp", "Motor_ID": "motor_id"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Load all machine learning models
# NOTE: only ONE scaler is used across classifier, RUL regressor, and health regressor,
# because all three models were trained on the same FEATURE_COLS. Using separate
# rul_feature_scaler.pkl / health_feature_scaler.pkl from older training runs caused
# mismatched feature scaling, which is why health scores were collapsing near 0.
clf = joblib.load(os.path.join(MODEL_DIR, "condition_classifier.pkl"))
reg = joblib.load(os.path.join(MODEL_DIR, "rul_regressor.pkl"))
health_reg = joblib.load(os.path.join(MODEL_DIR, "health_regressor.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

LATEST_ROWS = (
    df.groupby("motor_id", group_keys=False)
    .apply(lambda g: g.sample(n=1, random_state=42))
    .reset_index(drop=True)
    .sort_values("motor_id")
    .reset_index(drop=True)
)

ALL_MOTOR_IDS = sorted(df["motor_id"].unique().tolist())

STATUS_MAP = {
    "Normal": "Active",
    "Warning": "Idle",
    "Critical": "Maintenance",
    "Failure": "Stopped"
}

SENSOR_STATUS_MAP = {
    "Normal": "Normal",
    "Warning": "Watch",
    "Critical": "Critical",
    "Failure": "Critical"
}

# --- Mathematical Helpers ---
def get_avg_voltage(row):
    return round((row["Voltage_L1"] + row["Voltage_L2"] + row["Voltage_L3"]) / 3.0, 1)

def get_vibration_rms(row):
    return round(math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2), 2)


def predict_row(row):
    features = row[FEATURE_COLS].values.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Classify Condition
    pred_encoded = clf.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Predict Health Score
    health_score = float(health_reg.predict(features_scaled)[0])
    health_score = max(0.0, min(100.0, health_score))
    failure_probability = round(100 - health_score, 1)

    # Predict Remaining Useful Life
    predicted_rul = float(reg.predict(features_scaled)[0])
    predicted_rul = min(predicted_rul, RUL_DISPLAY_CAP)

    return {
        "condition_label": pred_label,
        "health_score": round(health_score, 1),
        "failure_probability": failure_probability,
        "rul_hours": round(predicted_rul, 1)
    }

def get_row_for_device(device_id):
    match = LATEST_ROWS[LATEST_ROWS["motor_id"] == device_id]
    if match.empty:
        return LATEST_ROWS.iloc[0]
    return match.iloc[0]


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        is_admin=current_user.is_admin,
        username=current_user.username
    )

@app.route("/api/devices")
@login_required
def get_devices():
    return jsonify({"devices": ALL_MOTOR_IDS})


@app.route("/api/status")
@login_required
def get_status():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    row = get_row_for_device(device_id)
    prediction = predict_row(row)

    return jsonify({
        "health_score": prediction["health_score"],
        "rul_hours": prediction["rul_hours"],
        "failure_probability": prediction["failure_probability"],
        "false_alarm_rate": 0.02,
        "temperature": float(row["Temperature"]),
        "vibration": get_vibration_rms(row),
        "current": get_avg_voltage(row),
        "pressure": float(row["Rotational_Speed"]),
        "device": str(row["motor_id"])
    })


@app.route("/api/alerts")
@login_required
def get_alerts():
    device_id = request.args.get("device")
    alerts = []
    rows = LATEST_ROWS if device_id is None else LATEST_ROWS[LATEST_ROWS["motor_id"] == device_id]

    for _, row in rows.iterrows():
        prediction = predict_row(row)
        label = prediction["condition_label"]

        if label == "Critical":
            alerts.append({
                "type": "warning",
                "icon": "fa-temperature-high",
                "title": f"Critical Pattern Detected on {row['motor_id']}",
                "description": "Sensor pattern matches critical degradation signature identified by the AI model.",
                "time": "Just now",
                "action": "Investigate"
            })
        elif label == "Failure":
            alerts.append({
                "type": "warning",
                "icon": "fa-circle-exclamation",
                "title": f"Failure Risk on {row['motor_id']}",
                "description": "Model predicts imminent failure risk based on current sensor readings.",
                "time": "Just now",
                "action": "Investigate"
            })
        elif label == "Warning":
            alerts.append({
                "type": "info",
                "icon": "fa-chart-line",
                "title": f"Early Degradation Signs on {row['motor_id']}",
                "description": "Maintenance recommended within the next scheduled window.",
                "time": "Just now",
                "action": "View Plan"
            })

    if not alerts:
        alerts.append({
            "type": "safe",
            "icon": "fa-shield",
            "title": "All Motors Normal",
            "description": "No degradation patterns detected across monitored motors.",
            "time": "Just now",
            "action": None
        })

    return jsonify(alerts[:20])


@app.route("/api/motors")
@login_required
def get_motors():
    motors = []

    for _, row in LATEST_ROWS.iterrows():
        prediction = predict_row(row)
        status = STATUS_MAP.get(prediction["condition_label"], "Active")

        motors.append({
            "id": str(row["motor_id"]),
            "name": f"Induction Motor {row['motor_id']}",
            "location": "Production Line",
            "status": status,
            "health_score": int(round(prediction["health_score"])),
            "rul_hours": int(prediction["rul_hours"]),
            "last_update": row["timestamp"].strftime("%Y-%m-%d %H:%M")
        })

    summary = {
        "total": len(motors),
        "active": sum(1 for m in motors if m["status"] == "Active"),
        "idle": sum(1 for m in motors if m["status"] == "Idle"),
        "maintenance": sum(1 for m in motors if m["status"] == "Maintenance"),
        "stopped": sum(1 for m in motors if m["status"] == "Stopped"),
    }

    return jsonify({"motors": motors, "summary": summary})


@app.route("/api/sensors")
@login_required
def get_sensors():
    readings = []

    for _, row in LATEST_ROWS.iterrows():
        prediction = predict_row(row)
        status = SENSOR_STATUS_MAP.get(prediction["condition_label"], "Normal")

        readings.append({
            "motor_id": str(row["motor_id"]),
            "motor_name": f"Induction Motor {row['motor_id']}",
            "temperature": float(row["Temperature"]),
            "vibration": get_vibration_rms(row),
            "current": get_avg_voltage(row),
            "pressure": float(row["Rotational_Speed"]),
            "status": status
        })

    return jsonify(readings)


@app.route("/api/logs")
@login_required
def get_logs():
    logs = []

    for i, row in LATEST_ROWS.iterrows():
        prediction = predict_row(row)
        label = prediction["condition_label"]

        icon_map = {
            "Normal": "fa-shield",
            "Warning": "fa-temperature-high",
            "Critical": "fa-circle-exclamation",
            "Failure": "fa-circle-exclamation"
        }
        type_map = {
            "Normal": "safe",
            "Warning": "warning",
            "Critical": "critical",
            "Failure": "critical"
        }
        title_map = {
            "Normal": "Normal Operating Pattern",
            "Warning": "Early Degradation Pattern Detected",
            "Critical": "Critical Degradation Pattern Detected",
            "Failure": "Failure Risk Pattern Detected"
        }

        logs.append({
            "id": i,
            "type": type_map.get(label, "safe"),
            "icon": icon_map.get(label, "fa-shield"),
            "title": title_map.get(label, "Status Update"),
            "description": f"Predicted condition: {label}. Estimated RUL: {int(prediction['rul_hours'])} hours.",
            "device": f"Induction Motor {row['motor_id']}",
            "time": row["timestamp"].strftime("%Y-%m-%d %H:%M")
        })

    return jsonify(logs)


@app.route("/api/settings", methods=["POST"])
@login_required
def update_settings():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    return jsonify({"message": "Settings saved."})


with app.app_context():
    db.create_all()
    if not User.query.filter_by(role="admin").first():
        admin = User(username="admin", email="admin@local", role="admin")
        admin.set_password("admin123")  # CHANGE THIS AFTER FIRST LOGIN
        db.session.add(admin)
        db.session.commit()
        print("Default admin created -> username: admin | password: admin123")


if __name__ == "__main__":
    app.run(debug=True, port=5000)