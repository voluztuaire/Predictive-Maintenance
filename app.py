from flask import Flask, render_template, jsonify, request, send_file
from flask_login import LoginManager, login_required, current_user
import pandas as pd
import numpy as np
import joblib
import math
import os
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

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

FEATURE_COLS = [
    "Voltage_L1", "Voltage_L2", "Voltage_L3", "Frequency",
    "Power_Factor", "Temperature", "Vibration_X", "Vibration_Y",
    "Vibration_Z", "Rotational_Speed"
]

RUL_DISPLAY_CAP = 5000

df = pd.read_csv("datasets/raw_sensor_new_data.csv")
df.rename(columns={"Timestamp": "timestamp", "Motor_ID": "motor_id"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])

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

RECOMMENDATION_MAP = {
    "Normal": "System operating normally. Continue routine monitoring.",
    "Warning": "Early degradation detected. Schedule inspection within the next 200 operating hours.",
    "Critical": "Significant degradation detected. Schedule maintenance within the next 48 hours.",
    "Failure": "Immediate shutdown and inspection recommended to avoid unplanned downtime."
}


def get_avg_voltage(row):
    return round((row["Voltage_L1"] + row["Voltage_L2"] + row["Voltage_L3"]) / 3.0, 1)


def get_vibration_rms(row):
    return round(math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2), 2)


def predict_row(row):
    features = row[FEATURE_COLS].values.reshape(1, -1)
    features_scaled = scaler.transform(features)

    pred_encoded = clf.predict(features_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    health_score = float(health_reg.predict(features_scaled)[0])
    health_score = max(0.0, min(100.0, health_score))
    failure_probability = round(100 - health_score, 1)

    predicted_rul = float(reg.predict(features_scaled)[0])
    predicted_rul = min(predicted_rul, RUL_DISPLAY_CAP)

    rul_days = predicted_rul / 24
    if rul_days <= 30:
        risk_window_label = f"Estimated within {round(rul_days)} days"
    else:
        risk_window_label = "Beyond 30-day horizon"

    return {
        "condition_label": pred_label,
        "health_score": round(health_score, 1),
        "failure_probability": failure_probability,
        "rul_hours": round(predicted_rul, 1),
        "risk_window_label": risk_window_label,
        "recommendation": RECOMMENDATION_MAP.get(pred_label, RECOMMENDATION_MAP["Normal"])
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
        "risk_window_label": prediction["risk_window_label"],
        "recommendation": prediction["recommendation"],
        "false_alarm_rate": 0.02,
        "temperature": float(row["Temperature"]),
        "vibration": get_vibration_rms(row),
        "current": get_avg_voltage(row),
        "pressure": float(row["Rotational_Speed"]),
        "device": str(row["motor_id"])
    })


@app.route("/api/history")
@login_required
def get_history():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    points = int(request.args.get("points", 20))

    motor_df = df[df["motor_id"] == device_id].sort_values("timestamp").tail(points)

    return jsonify({
        "labels": motor_df["timestamp"].dt.strftime("%H:%M").tolist(),
        "temperature": motor_df["Temperature"].round(1).tolist(),
        "vibration": [get_vibration_rms(row) for _, row in motor_df.iterrows()],
        "voltage": [get_avg_voltage(row) for _, row in motor_df.iterrows()],
        "rpm": motor_df["Rotational_Speed"].round(1).tolist()
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
        device_id_str = str(row["motor_id"])

        if label == "Critical":
            alerts.append({
                "type": "warning",
                "icon": "fa-temperature-high",
                "title": f"Critical Pattern Detected on {device_id_str}",
                "description": "Sensor pattern matches critical degradation signature identified by the AI model.",
                "time": "Just now",
                "action": "Investigate",
                "device_id": device_id_str
            })
        elif label == "Failure":
            alerts.append({
                "type": "warning",
                "icon": "fa-circle-exclamation",
                "title": f"Failure Risk on {device_id_str}",
                "description": "Model predicts imminent failure risk based on current sensor readings.",
                "time": "Just now",
                "action": "Investigate",
                "device_id": device_id_str
            })
        elif label == "Warning":
            alerts.append({
                "type": "info",
                "icon": "fa-chart-line",
                "title": f"Early Degradation Signs on {device_id_str}",
                "description": "Maintenance recommended within the next scheduled window.",
                "time": "Just now",
                "action": "View Plan",
                "device_id": device_id_str
            })

    if not alerts:
        alerts.append({
            "type": "safe",
            "icon": "fa-shield",
            "title": "All Motors Normal",
            "description": "No degradation patterns detected across monitored motors.",
            "time": "Just now",
            "action": None,
            "device_id": None
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
            "device_id": str(row["motor_id"]),
            "time": row["timestamp"].strftime("%Y-%m-%d %H:%M")
        })

    return jsonify(logs)


@app.route("/api/settings", methods=["POST"])
@login_required
def update_settings():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    return jsonify({"message": "Settings saved."})


@app.route("/api/report", methods=["POST"])
@login_required
def generate_report():
    payload = request.get_json()
    selected_motors = payload.get("motors", ALL_MOTOR_IDS)
    selected_fields = payload.get("fields", ["temperature", "vibration", "voltage", "rpm"])
    include_predictions = payload.get("include_predictions", True)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], textColor=colors.HexColor("#f97316"))

    elements = []
    elements.append(Paragraph("Winteq Predictive Maintenance Report", title_style))
    elements.append(Paragraph(f"Generated by: {current_user.username}", styles["Normal"]))
    elements.append(Paragraph(f"Motors included: {', '.join(selected_motors)}", styles["Normal"]))
    elements.append(Spacer(1, 0.5*cm))

    for device_id in selected_motors:
        row = get_row_for_device(device_id)
        prediction = predict_row(row)

        elements.append(Paragraph(f"Motor: {device_id}", styles["Heading2"]))

        table_data = [["Field", "Value"]]
        field_map = {
            "temperature": ("Temperature (C)", float(row["Temperature"])),
            "vibration": ("Vibration RMS (mm/s)", get_vibration_rms(row)),
            "voltage": ("Voltage Avg (V)", get_avg_voltage(row)),
            "rpm": ("Rotational Speed (RPM)", float(row["Rotational_Speed"])),
        }
        for f in selected_fields:
            if f in field_map:
                label, value = field_map[f]
                table_data.append([label, str(value)])

        if include_predictions:
            table_data.append(["Predicted Condition", prediction["condition_label"]])
            table_data.append(["Health Score", f"{prediction['health_score']}%"])
            table_data.append(["Estimated RUL", f"{prediction['rul_hours']} hours"])
            table_data.append(["Failure Probability", f"{prediction['failure_probability']}%"])
            table_data.append(["Risk Window", prediction["risk_window_label"]])
            table_data.append(["Recommendation", prediction["recommendation"]])

        t = Table(table_data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)

        motor_hist = df[df["motor_id"] == device_id].sort_values("timestamp").tail(30)
        if not motor_hist.empty and "temperature" in selected_fields:
            fig, ax = plt.subplots(figsize=(6, 2.2))
            ax.plot(range(len(motor_hist)), motor_hist["Temperature"], color="#f97316")
            ax.set_title(f"{device_id} - Temperature Trend", fontsize=9)
            ax.set_xticks([])
            img_buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(img_buf, format="png", dpi=120)
            plt.close(fig)
            img_buf.seek(0)
            elements.append(Spacer(1, 0.3*cm))
            elements.append(Image(img_buf, width=14*cm, height=4*cm))

        elements.append(Spacer(1, 0.8*cm))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="winteq_maintenance_report.pdf"
    )


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