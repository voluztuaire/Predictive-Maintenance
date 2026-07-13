from xgboost import XGBClassifier
from flask import Flask, render_template, jsonify, request, send_file
from flask_login import LoginManager, login_required, current_user
import pandas as pd
import numpy as np
import joblib
import math
import os
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from models import db, User, Threshold, DEFAULT_THRESHOLDS
from auth import auth_bp
from forecast_engine import forecast_health_and_rul

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

# ============================================================
# NA'S MODEL (Fault Type classifier)
# ------------------------------------------------------------
# NA's RUL regressor is no longer used for health_score/rul_hours.
# Those two fields are now sourced from SN's condition-grounded
# health_v2 / rul_v2 models (see below) since they are trained
# against labels that actually reflect physical condition rather
# than pure elapsed time. NA's fault classifier is still the only
# source for fault_type / probable_cause / recommendation.
# ============================================================
fault_clf = joblib.load(os.path.join(MODEL_DIR, "fault_classifier_model.pkl"))
na_scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))

NA_FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
FAULT_LABEL_MAP = joblib.load(os.path.join(MODEL_DIR, "fault_label_map.pkl"))

# ============================================================
# SH'S MODEL (Severity condition: Normal/Warning/Critical/Failure)
# ============================================================
condition_clf = joblib.load(os.path.join(MODEL_DIR, "condition_classifier.joblib"))
sh_scaler = joblib.load(os.path.join(MODEL_DIR, "condition_scaler.joblib"))

SH_FEATURE_COLS = [
    "Voltage_L1", "Voltage_L2", "Voltage_L3", "Frequency", "Power_Factor",
    "Temperature", "Vibration_X", "Vibration_Y", "Vibration_Z", "Rotational_Speed",
    "Voltage_Imbalance", "Vibration_Total", "Voltage_Mean", "RPM_Deviation"
]

# ============================================================
# SN'S MODELS (condition-grounded Health Score v2 + RUL v2)
# ------------------------------------------------------------
# SN's feature spec reuses the exact same 14 base features and the
# exact same shared scaler as SH's model (condition_scaler.joblib),
# then adds rolling (1h) / trend (24h) features on top, computed
# per-motor. These extra columns are computed once at startup below
# and cached on `df`, so no per-request feature engineering is needed.
# ============================================================
sn_rul_model = joblib.load(os.path.join(MODEL_DIR, "rul_v2.joblib"))
sn_health_model = joblib.load(os.path.join(MODEL_DIR, "health_v2.joblib"))

SN_SENSOR_COLS = [
    "Voltage_L1", "Voltage_L2", "Voltage_L3",
    "Frequency", "Power_Factor", "Temperature",
    "Vibration_X", "Vibration_Y", "Vibration_Z",
    "Rotational_Speed",
]
SN_ENGINEERED_BASE = ["Voltage_Imbalance", "Vibration_Total", "Voltage_Mean", "RPM_Deviation"]
SN_FEATURE_COLS = SN_SENSOR_COLS + SN_ENGINEERED_BASE

SN_SHORT_WINDOW = 4
SN_LONG_WINDOW = 96


def sn_add_scaled_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    scaled = sh_scaler.transform(data[SN_FEATURE_COLS])
    for i, col in enumerate(SN_FEATURE_COLS):
        data[f"{col}_scaled"] = scaled[:, i]
    return data


def sn_add_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sort_values(["motor_id", "timestamp"]).reset_index(drop=True)
    cols_to_track = [f"{c}_scaled" for c in SN_FEATURE_COLS]
    n = SN_LONG_WINDOW
    out_frames = []

    for motor_id, g in data.groupby("motor_id"):
        g = g.copy()
        t = np.arange(len(g))
        for col in cols_to_track:
            g[f"{col}_roll_mean_1h"] = g[col].rolling(SN_SHORT_WINDOW, min_periods=1).mean()
            g[f"{col}_roll_std_1h"] = g[col].rolling(SN_SHORT_WINDOW, min_periods=1).std().fillna(0)

            y = g[col].values
            roll_sum_y = pd.Series(y).rolling(n, min_periods=4).sum().values
            roll_sum_t = pd.Series(t).rolling(n, min_periods=4).sum().values
            roll_sum_ty = pd.Series(t * y).rolling(n, min_periods=4).sum().values
            roll_sum_tt = pd.Series(t * t).rolling(n, min_periods=4).sum().values
            roll_count = pd.Series(y).rolling(n, min_periods=4).count().values

            denom = roll_count * roll_sum_tt - roll_sum_t ** 2
            slope = np.where(denom != 0, (roll_count * roll_sum_ty - roll_sum_t * roll_sum_y) / denom, 0.0)
            g[f"{col}_trend_24h"] = np.nan_to_num(slope)
        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


def sn_get_feature_columns():
    cols = []
    for c in SN_FEATURE_COLS:
        sc = f"{c}_scaled"
        cols += [sc, f"{sc}_roll_mean_1h", f"{sc}_roll_std_1h", f"{sc}_trend_24h"]
    return cols


# ============================================================
# RECOMMENDATION TABLE
# ============================================================
RECOMMENDATION_TABLE = {
    "Normal": {
        "probable_cause": "-",
        "recommended_action": "Normal condition, continue routine monitoring"
    },
    "Rotor Bar": {
        "probable_cause": "Broken/cracked rotor bar due to repeated thermal stress or excessive starting frequency",
        "recommended_action": "Perform motor current signature analysis (MCSA) and schedule rotor inspection/teardown"
    },
    "Bearing Wear": {
        "probable_cause": "Inadequate lubrication, contamination, or bearing wear from excessive load",
        "recommended_action": "Re-lubricate or replace the bearing, check seal condition and shaft alignment"
    },
    "Misalignment": {
        "probable_cause": "Motor shaft misaligned with the driven load/coupling",
        "recommended_action": "Perform laser alignment check and correct the motor mounting position"
    },
    "Stator Winding": {
        "probable_cause": "Winding insulation degradation from repeated overheating or moisture contamination",
        "recommended_action": "Perform insulation resistance test (megger) and check the cooling system"
    }
}


# ============================================================
# NA'S FEATURE ENGINEERING (used for fault type classification only)
# ============================================================
def engineer_features(data: pd.DataFrame, short_window=4, long_window=24) -> pd.DataFrame:
    data = data.copy()

    data["Vibration_RMS_Combined"] = np.sqrt(
        data["Vibration_X"]**2 + data["Vibration_Y"]**2 + data["Vibration_Z"]**2
    )

    v_mean = data[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].mean(axis=1)
    v_max_dev = data[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].sub(v_mean, axis=0).abs().max(axis=1)
    data["Voltage_Imbalance_Pct"] = (v_max_dev / v_mean) * 100

    feature_cols = ["Temperature", "Vibration_RMS_Combined", "Rotational_Speed", "Voltage_Imbalance_Pct"]

    grouped = data.groupby("motor_id")
    for col in feature_cols:
        data[f"{col}_roll_mean_short"] = grouped[col].transform(lambda s: s.rolling(short_window, min_periods=1).mean())
        data[f"{col}_roll_std_short"] = grouped[col].transform(lambda s: s.rolling(short_window, min_periods=1).std().fillna(0))
        data[f"{col}_roll_mean_long"] = grouped[col].transform(lambda s: s.rolling(long_window, min_periods=1).mean())
        data[f"{col}_delta"] = grouped[col].diff().fillna(0)

    return data


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
df = pd.read_csv("datasets/raw_sensor_new_data.csv")
df.rename(columns={"Timestamp": "timestamp", "Motor_ID": "motor_id"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["motor_id", "timestamp"]).reset_index(drop=True)

df = engineer_features(df)

df["Voltage_Imbalance"] = df[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].std(axis=1)
df["Vibration_Total"] = np.sqrt(df["Vibration_X"]**2 + df["Vibration_Y"]**2 + df["Vibration_Z"]**2)
df["Voltage_Mean"] = df[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].mean(axis=1)
df["RPM_Deviation"] = abs(df["Rotational_Speed"] - 1500)

df = sn_add_scaled_features(df)
df = sn_add_temporal_features(df)

SN_ALL_FEATURE_COLS = sn_get_feature_columns()
_sn_X_all = df[SN_ALL_FEATURE_COLS].fillna(0).values
df["sn_rul_hours"] = sn_rul_model.predict(_sn_X_all)
df["Temperature_smooth"] = df.groupby("motor_id")["Temperature"].transform(lambda s: s.rolling(20, min_periods=1).mean())
df["sn_health_score"] = np.clip(sn_health_model.predict(_sn_X_all), 0, 100)

ALL_MOTOR_IDS = sorted(df["motor_id"].unique().tolist())

motor_row_counts = df.groupby("motor_id").size().to_dict()
motor_row_index = {mid: int(motor_row_counts[mid] * 0.7) for mid in ALL_MOTOR_IDS}

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_avg_voltage(row):
    return round((row["Voltage_L1"] + row["Voltage_L2"] + row["Voltage_L3"]) / 3.0, 1)


def get_vibration_rms(row):
    return round(math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2), 2)


def get_row_for_device(device_id):
    motor_df = df[df["motor_id"] == device_id].reset_index(drop=True)
    idx = motor_row_index.get(device_id, 0) % len(motor_df)
    return motor_df.iloc[idx]


def advance_all_motors():
    for mid in ALL_MOTOR_IDS:
        motor_row_index[mid] = (motor_row_index.get(mid, 0) + 1) % motor_row_counts[mid]


def predict_row(row):
    na_features = row[NA_FEATURE_COLS].values.reshape(1, -1)
    na_scaled = na_scaler.transform(na_features)

    fault_pred_raw = fault_clf.predict(na_scaled)[0]
    if isinstance(fault_clf, XGBClassifier):
        fault_pred = FAULT_LABEL_MAP[int(fault_pred_raw)]
    else:
        fault_pred = str(fault_pred_raw)

    sh_features = row[SH_FEATURE_COLS].values.reshape(1, -1)
    sh_scaled = sh_scaler.transform(sh_features)
    condition_label = condition_clf.predict(sh_scaled)[0]

    rul_pred = float(row["sn_rul_hours"])
    ml_health = float(np.clip(row["sn_health_score"], 0, 100))

    temp = float(row["Temperature"])
    vib = math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2)
    rpm_dev = abs(float(row["Rotational_Speed"]) - 1475)

    def score_from(value, healthy_max, fail_at):
        if value <= healthy_max:
            return 100.0
        if value >= fail_at:
            return 0.0
        return 100.0 * (fail_at - value) / (fail_at - healthy_max)

    temp_score = score_from(temp, 50, 85)
    vib_score = score_from(vib, 2.5, 9.0)
    rpm_score = score_from(rpm_dev, 30, 90)
    rule_health = min(temp_score, vib_score, rpm_score)

    health_score = min(ml_health, rule_health)
    failure_probability = round(100 - health_score, 1)

    fault_info = RECOMMENDATION_TABLE.get(fault_pred, RECOMMENDATION_TABLE["Normal"])

    rul_display = min(rul_pred, 5000)
    rul_days = rul_pred / 24
    if rul_days <= 30:
        risk_window_label = f"Estimated within {round(rul_days)} days"
    else:
        risk_window_label = "Beyond 30-day horizon"

    return {
        "condition_label": condition_label,
        "fault_type": fault_pred,
        "probable_cause": fault_info["probable_cause"],
        "recommendation": fault_info["recommended_action"],
        "health_score": round(health_score, 1),
        "failure_probability": failure_probability,
        "rul_hours": round(rul_display, 1),
        "risk_window_label": risk_window_label
    }


# ============================================================
# ROUTES
# ============================================================
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
        "probable_cause": prediction["probable_cause"],
        "fault_type": prediction["fault_type"],
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

    if device_id:
        rows = [get_row_for_device(device_id)]
    else:
        rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for row in rows:
        prediction = predict_row(row)
        label = prediction["condition_label"]
        fault = prediction["fault_type"]
        health = prediction["health_score"]
        device_id_str = str(row["motor_id"])

        if label == "Critical" or health < 30:
            alerts.append({
                "type": "warning",
                "icon": "fa-temperature-high",
                "title": f"Critical Pattern Detected on {device_id_str}",
                "description": f"{fault}: {prediction['probable_cause']}",
                "time": "Just now",
                "action": "Investigate",
                "device_id": device_id_str
            })
        elif label == "Failure" or health < 15:
            alerts.append({
                "type": "warning",
                "icon": "fa-circle-exclamation",
                "title": f"Failure Risk on {device_id_str}",
                "description": f"{fault}: {prediction['probable_cause']}",
                "time": "Just now",
                "action": "Investigate",
                "device_id": device_id_str
            })
        elif label == "Warning" or health < 60:
            alerts.append({
                "type": "info",
                "icon": "fa-chart-line",
                "title": f"Early Degradation Signs on {device_id_str}",
                "description": f"{fault}: {prediction['probable_cause']}",
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
    rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for row in rows:
        prediction = predict_row(row)
        status = STATUS_MAP.get(prediction["condition_label"], "Active")

        motors.append({
            "id": str(row["motor_id"]),
            "name": f"Induction Motor {row['motor_id']}",
            "location": "Production Line",
            "status": status,
            "health_score": int(round(prediction["health_score"])),
            "rul_hours": int(prediction["rul_hours"]),
            "fault_type": prediction["fault_type"],
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
    rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for row in rows:
        prediction = predict_row(row)
        status = SENSOR_STATUS_MAP.get(prediction["condition_label"], "Normal")

        readings.append({
            "motor_id": str(row["motor_id"]),
            "motor_name": f"Induction Motor {row['motor_id']}",
            "temperature": float(row["Temperature"]),
            "vibration": get_vibration_rms(row),
            "current": get_avg_voltage(row),
            "pressure": float(row["Rotational_Speed"]),
            "status": status,
            "fault_type": prediction["fault_type"]
        })

    return jsonify(readings)


@app.route("/api/logs")
@login_required
def get_logs():
    logs = []
    rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for i, row in enumerate(rows):
        prediction = predict_row(row)
        label = prediction["condition_label"]
        fault = prediction["fault_type"]

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
            "description": f"Fault: {fault}. RUL: {int(prediction['rul_hours'])} hours. {prediction['probable_cause']}",
            "device": f"Induction Motor {row['motor_id']}",
            "device_id": str(row["motor_id"]),
            "time": row["timestamp"].strftime("%Y-%m-%d %H:%M")
        })

    return jsonify(logs)

@app.route("/api/forecast")
@login_required
def get_forecast():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    horizon = float(request.args.get("horizon", 48))

    motor_df = df[df["motor_id"] == device_id].sort_values("timestamp")
    history_df = motor_df.rename(columns={"timestamp": "Timestamp"}).tail(200)

    result = forecast_health_and_rul(
        history_df, horizon,
        scaler=sh_scaler, rul_model=sn_rul_model, health_model=sn_health_model
    )

    smooth_window = 8
    result["predicted_RUL_hours_smoothed"] = result["predicted_RUL_hours"].rolling(smooth_window, min_periods=1, center=True).mean()
    result["predicted_Health_Score_smoothed"] = result["predicted_Health_Score"].rolling(smooth_window, min_periods=1, center=True).mean()

    return jsonify({
        "labels": result["Timestamp"].dt.strftime("%H:%M").tolist(),
        "predicted_health": result["predicted_Health_Score_smoothed"].round(1).tolist(),
        "predicted_rul": result["predicted_RUL_hours_smoothed"].round(1).tolist(),
        "sensors": {
            "Temperature": result["Temperature"].round(1).tolist(),
            "Vibration_X": result["Vibration_X"].round(2).tolist(),
            "Vibration_Y": result["Vibration_Y"].round(2).tolist(),
            "Vibration_Z": result["Vibration_Z"].round(2).tolist(),
            "Voltage_L1": result["Voltage_L1"].round(1).tolist(),
            "Voltage_L2": result["Voltage_L2"].round(1).tolist(),
            "Voltage_L3": result["Voltage_L3"].round(1).tolist(),
            "Frequency": result["Frequency"].round(2).tolist(),
            "Power_Factor": result["Power_Factor"].round(2).tolist(),
            "Rotational_Speed": result["Rotational_Speed"].round(1).tolist(),
        }
    })

@app.route("/api/tick", methods=["POST"])
@login_required
def tick():
    advance_all_motors()
    return jsonify({"status": "ok"})


@app.route("/api/settings", methods=["GET", "POST"])
@login_required
def update_settings():
    threshold = Threshold.query.first()

    if request.method == "GET":
        return jsonify({
            "temperature": threshold.temperature,
            "vibration": threshold.vibration,
            "current_deviation": threshold.current_deviation,
            "pressure": threshold.pressure
        })

    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403

    payload = request.get_json()
    threshold.temperature = payload.get("temperature", threshold.temperature)
    threshold.vibration = payload.get("vibration", threshold.vibration)
    threshold.current_deviation = payload.get("current_deviation", threshold.current_deviation)
    threshold.pressure = payload.get("pressure", threshold.pressure)
    db.session.commit()

    return jsonify({"message": "Settings saved."})


@app.route("/api/settings/reset", methods=["POST"])
@login_required
def reset_settings():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403

    threshold = Threshold.query.first()
    threshold.temperature = DEFAULT_THRESHOLDS["temperature"]
    threshold.vibration = DEFAULT_THRESHOLDS["vibration"]
    threshold.current_deviation = DEFAULT_THRESHOLDS["current_deviation"]
    threshold.pressure = DEFAULT_THRESHOLDS["pressure"]
    db.session.commit()

    return jsonify(DEFAULT_THRESHOLDS)


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

    wrap_style = ParagraphStyle(
        "WrapStyle",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
        wordWrap='CJK'
    )

    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], textColor=colors.HexColor("#f97316"))

    elements = []
    elements.append(Paragraph("Winteq Predictive Maintenance Report", title_style))
    elements.append(Paragraph(f"Generated by: {current_user.username}", styles["Normal"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Paragraph(f"Motors included: {', '.join(selected_motors)}", styles["Normal"]))
    elements.append(Spacer(1, 0.5*cm))

    for device_id in selected_motors:
        row = get_row_for_device(device_id)
        prediction = predict_row(row)

        elements.append(Paragraph(f"Motor: {device_id}", styles["Heading2"]))

        table_data = [["Field", "Value"]]
        field_map = {
            "temperature": ("Temperature (C)", str(round(float(row["Temperature"]), 1))),
            "vibration": ("Vibration RMS (mm/s)", str(get_vibration_rms(row))),
            "voltage": ("Voltage Avg (V)", str(get_avg_voltage(row))),
            "rpm": ("Rotational Speed (RPM)", str(round(float(row["Rotational_Speed"]), 1))),
        }
        for f in selected_fields:
            if f in field_map:
                label, value = field_map[f]
                table_data.append([label, value])

        if include_predictions:
            table_data.append(["Predicted Condition", prediction["condition_label"]])
            table_data.append(["Fault Type", prediction["fault_type"]])
            table_data.append(["Health Score", f"{prediction['health_score']}%"])
            table_data.append(["Estimated RUL", f"{prediction['rul_hours']} hours"])
            table_data.append(["Failure Probability", f"{prediction['failure_probability']}%"])
            table_data.append(["Risk Window", prediction["risk_window_label"]])
            probable_cause_para = Paragraph(prediction["probable_cause"], wrap_style)
            recommendation_para = Paragraph(prediction["recommendation"], wrap_style)
            table_data.append(["Probable Cause", probable_cause_para])
            table_data.append(["Recommendation", recommendation_para])

        t = Table(table_data, colWidths=[6*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(t)

        motor_hist = df[df["motor_id"] == device_id].sort_values("timestamp").tail(30)

        chart_configs = [
            ("temperature", "Temperature", motor_hist["Temperature"], "#f97316"),
            ("vibration", "Vibration RMS", [get_vibration_rms(r) for _, r in motor_hist.iterrows()], "#a855f7"),
            ("voltage", "Voltage Avg", [get_avg_voltage(r) for _, r in motor_hist.iterrows()], "#38bdf8"),
            ("rpm", "Rotational Speed", motor_hist["Rotational_Speed"], "#22c55e"),
        ]

        for field_key, field_label, series, color in chart_configs:
            if not motor_hist.empty and field_key in selected_fields:
                fig, ax = plt.subplots(figsize=(6, 2.2))
                ax.plot(range(len(motor_hist)), series, color=color)
                ax.set_title(f"{device_id} - {field_label} Trend", fontsize=9)
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

    if not Threshold.query.first():
        db.session.add(Threshold())
        db.session.commit()
        print("Default thresholds created.")

    if not User.query.filter_by(role="admin").first():
        admin = User(username="admin", email="admin@local", role="admin")
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("Default admin created -> username: admin | password: admin123")


if __name__ == "__main__":
    app.run(debug=True, port=5000)