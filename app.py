from xgboost import XGBClassifier
from expert_validation import submit_for_review, list_reviews, approve_review, reject_review, review_stats
from retrain_pipeline import run_retraining, get_deployed_model_dir
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

from models import db, User, AlarmRule 
from auth import auth_bp
from forecast_engine import forecast_health_and_rul

from chatbot_llm import (
    chatbot_response_llm,
    LLMConversationContext,
    build_system_prompt,
    save_history,
)

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
    "Voltage_L1", "Voltage_L2", "Voltage_L3",
    "Current_L1", "Current_L2", "Current_L3",
    "Frequency", "Power_Factor", "Temperature",
    "Vibration_X", "Vibration_Y", "Vibration_Z", "Rotational_Speed",
    "Voltage_Imbalance_Pct", "Current_Imbalance_Pct",
    "Vibration_Total", "Voltage_Mean", "Current_Mean", "RPM_Deviation"
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
sn_scaler = joblib.load(os.path.join(MODEL_DIR, "condition_scaler_v2_with_current.joblib"))
sn_rul_model = joblib.load(os.path.join(MODEL_DIR, "rul_iso.joblib"))
sn_health_model = joblib.load(os.path.join(MODEL_DIR, "health_iso.joblib"))

SN_SENSOR_COLS = [
    "Voltage_L1", "Voltage_L2", "Voltage_L3",
    "Current_L1", "Current_L2", "Current_L3",
    "Frequency", "Power_Factor", "Temperature",
    "Vibration_X", "Vibration_Y", "Vibration_Z",
    "Rotational_Speed",
]
SN_ENGINEERED_BASE = ["Voltage_Imbalance", "Current_Imbalance", "Vibration_Total", "Voltage_Mean", "RPM_Deviation"]
SN_FEATURE_COLS = SN_SENSOR_COLS + SN_ENGINEERED_BASE
SN_SHORT_WINDOW = 4
SN_LONG_WINDOW = 96


def sn_add_scaled_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    scaled = sn_scaler.transform(data[SN_FEATURE_COLS])
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
# CONDITION-BASED MONITORING (Salsa's threshold rules, no ML)
# ============================================================
THRESHOLDS = {
    'Temperature': {'warning': 51.0, 'critical': 57.0, 'failure': 66.0},
    'Vibration_X': {'warning': 1.80, 'critical': 3.60, 'failure': 5.50},
    'Vibration_Y': {'warning': 1.80, 'critical': 3.60, 'failure': 5.50},
    'Vibration_Z': {'warning': 1.90, 'critical': 3.90, 'failure': 5.80},
    'Voltage_L1_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L2_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L3_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L1_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L2_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L3_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Current_L1_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    'Current_L2_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    'Current_L3_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    'Frequency_high': {'warning': 50.10, 'critical': 50.15, 'failure': 50.20},
    'Frequency_low':  {'warning': 49.90, 'critical': 49.85, 'failure': 49.80},
    'RPM_low': {'warning': 1455.0, 'critical': 1450.0, 'failure': 1445.0},
    'Voltage_Imbalance_Pct': {'warning': 0.5, 'critical': 1.0, 'failure': 2.5},
    'Current_Imbalance_Pct': {'warning': 0.6, 'critical': 1.8, 'failure': 3.5},
}

# ============================================================
# DYNAMIC THRESHOLDS — dibaca dari tabel AlarmRule, bukan hardcode
# ============================================================
_rules_cache = None  # None = perlu di-refresh dari DB

def invalidate_rules_cache():
    global _rules_cache
    _rules_cache = None

def get_active_rules(device_id: str = None):
    """
    Ambil semua AlarmRule yang enabled, cocok untuk device tertentu
    (rule dengan device='All' berlaku untuk semua motor).
    Di-cache di memory supaya gak query DB tiap kali check_violations()
    dipanggil (dipanggil sangat sering: tiap tick, tiap api/status, dst).
    Cache di-invalidate otomatis tiap kali rule ditambah/diubah/dihapus.
    """
    global _rules_cache
    if _rules_cache is None:
        _rules_cache = [r.to_dict() for r in AlarmRule.query.filter_by(enabled=True).all()]

    if device_id is None:
        return _rules_cache
    return [r for r in _rules_cache if r["device"] == "All" or r["device"] == device_id]

def check_violations(row: dict, device_id: str = None) -> dict:
    """
    SEKARANG baca dari AlarmRule (database) via get_active_rules(),
    bukan dict THRESHOLDS statis. Ganti value di Settings -> langsung
    ngaruh di sini pada request berikutnya.
    """
    v = {'warning': [], 'critical': [], 'failure': []}
    rules = get_active_rules(device_id)

    for r in rules:
        val = row.get(r["parameter"])
        if val is None:
            continue
        if r["condition"] == "more_than":
            breached = val > r["value"]
            label = r["parameter"]
        else:  # less_than
            breached = val < r["value"]
            label = f'{r["parameter"]}(low)'
        if breached:
            tier = r["tier"] if r["tier"] in v else "warning"
            v[tier].append(label)

    return v

def assign_threshold_label(row: dict, device_id: str = None) -> str:
    viol = check_violations(row, device_id=device_id)
    n_f, n_c, n_w = len(viol['failure']), len(viol['critical']), len(viol['warning'])
    n_t = n_f + n_c + n_w
    if n_f >= 1 or n_t >= 4: return 'Failure'
    elif n_c >= 1: return 'Critical'
    elif n_w >= 1: return 'Warning'
    else: return 'Normal'

def get_threshold_alerts(row) -> dict:
    """row: a pandas Series from df (must have Voltage_Imbalance_Pct / Current_Imbalance_Pct already computed)."""
    row_dict = row.to_dict()
    motor_id = str(row.get("motor_id", ""))
    viol = check_violations(row_dict, device_id=motor_id)
    condition = assign_threshold_label(row_dict, device_id=motor_id)
    
    detail = []
    for tier in ('warning', 'critical', 'failure'):
        for param in viol[tier]:
            base = param.replace('(low)', '')
            actual_val = row_dict.get(base)
            key = base
            if base in ('Voltage_L1', 'Voltage_L2', 'Voltage_L3'):
                key = f"{base}_{'low' if '(low)' in param else 'high'}"
            elif base == 'Frequency':
                key = f"Frequency_{'low' if '(low)' in param else 'high'}"
            elif base == 'Rotational_Speed':
                key = 'RPM_low'
            elif base in ('Current_L1', 'Current_L2', 'Current_L3'):
                key = f"{base}_high"
            matching_rule = next(
                (r for r in get_active_rules(motor_id)
                 if r["parameter"] == base and r["tier"] == tier), None
            )
            threshold_val = matching_rule["value"] if matching_rule else None
            detail.append({
                'parameter': param,
                'tier': tier,
                'actual_value': round(float(actual_val), 3) if actual_val is not None else None,
                'threshold': threshold_val,
            })

    color_map = {'Normal': 'green', 'Warning': 'yellow', 'Critical': 'orange', 'Failure': 'red'}
    n_t = len(viol['warning']) + len(viol['critical']) + len(viol['failure'])

    return {
        'motor_id': str(row['motor_id']),
        'timestamp': str(row['timestamp']),
        'condition_label': condition,
        'status_color': color_map[condition],
        'violations': detail,
        'total_violations': n_t,
        'is_labeling_candidate': condition != 'Normal',
        'source': 'threshold_rule',
    }

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

    # NEW: current signature analysis, strongest signal for Rotor Bar fault
    data["Current_RMS_Combined"] = np.sqrt(
        data["Current_L1"]**2 + data["Current_L2"]**2 + data["Current_L3"]**2
    )
    i_mean = data[["Current_L1", "Current_L2", "Current_L3"]].mean(axis=1)
    i_max_dev = data[["Current_L1", "Current_L2", "Current_L3"]].sub(i_mean, axis=0).abs().max(axis=1)
    data["Current_Imbalance_Pct"] = (i_max_dev / i_mean) * 100

    feature_cols = [
        "Temperature", "Vibration_RMS_Combined", "Rotational_Speed", "Voltage_Imbalance_Pct",
        "Current_RMS_Combined", "Current_Imbalance_Pct",
    ]

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
df["Current_Imbalance"] = df[["Current_L1", "Current_L2", "Current_L3"]].std(axis=1)
df["Vibration_Total"] = np.sqrt(df["Vibration_X"]**2 + df["Vibration_Y"]**2 + df["Vibration_Z"]**2)
df["Voltage_Mean"] = df[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].mean(axis=1)
df["Current_Mean"] = df[["Current_L1", "Current_L2", "Current_L3"]].mean(axis=1)
df["RPM_Deviation"] = abs(df["Rotational_Speed"] - 1500)

# NEMA MG-1 style %imbalance for Salsa's condition classifier
_v_mean = df[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].mean(axis=1)
_v_dev = df[["Voltage_L1", "Voltage_L2", "Voltage_L3"]].sub(_v_mean, axis=0).abs().max(axis=1)
df["Voltage_Imbalance_Pct"] = (_v_dev / _v_mean) * 100

_i_mean = df[["Current_L1", "Current_L2", "Current_L3"]].mean(axis=1)
_i_dev = df[["Current_L1", "Current_L2", "Current_L3"]].sub(_i_mean, axis=0).abs().max(axis=1)
df["Current_Imbalance_Pct"] = (_i_dev / _i_mean) * 100

df = sn_add_scaled_features(df)
df = sn_add_temporal_features(df)

SN_ALL_FEATURE_COLS = sn_get_feature_columns()
_sn_X_all = df[SN_ALL_FEATURE_COLS].fillna(0).values
df["sn_rul_hours"] = sn_rul_model.predict(_sn_X_all)
df["Temperature_smooth"] = df.groupby("motor_id")["Temperature"].transform(lambda s: s.rolling(20, min_periods=1).mean())
df["sn_health_score"] = np.clip(sn_health_model.predict(_sn_X_all), 0, 100)
df["sn_health_score_smooth"] = df.groupby("motor_id")["sn_health_score"].transform(lambda s: s.rolling(5, min_periods=1, center=True).mean())

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
# NOTIFICATION STATE (sustained-state + cooldown to avoid spam)
# ============================================================
from collections import deque

SEVERITY_RANK = {"Normal": 0, "Warning": 1, "Critical": 2, "Failure": 3}
SUSTAIN_THRESHOLD = 3          # must hold this severity for 3 consecutive ticks
COOLDOWN_SECONDS = 120         # don't repeat same-level notif within 2 min

notif_state = {
    mid: {"streak_label": "Normal", "streak_count": 0, "last_notified_rank": 0, "last_notified_at": None}
    for mid in ALL_MOTOR_IDS
}
notifications_log = deque(maxlen=50)  # newest first
_notif_id_counter = 0


def check_motor_notification(motor_id, condition_label, fault_type, probable_cause):
    global _notif_id_counter
    state = notif_state[motor_id]
    now = datetime.now()

    # Track how many consecutive ticks this motor has held the current label
    if condition_label == state["streak_label"]:
        state["streak_count"] += 1
    else:
        state["streak_label"] = condition_label
        state["streak_count"] = 1

    rank = SEVERITY_RANK.get(condition_label, 0)
    sustained = state["streak_count"] >= SUSTAIN_THRESHOLD

    # Only care about Warning/Critical/Failure, sustained, and worse than last notified
    if not sustained or rank == 0:
        return

    is_escalation = rank > state["last_notified_rank"]
    cooldown_passed = (
        state["last_notified_at"] is None
        or (now - state["last_notified_at"]).total_seconds() >= COOLDOWN_SECONDS
    )

    if is_escalation or cooldown_passed:
        _notif_id_counter += 1
        notifications_log.appendleft({
            "id": _notif_id_counter,
            "motor_id": motor_id,
            "severity": condition_label,
            "title": f"{condition_label} — {motor_id}",
            "description": f"{fault_type}: {probable_cause}",
            "time": now.strftime("%H:%M:%S"),
            "read": False,
        })
        state["last_notified_rank"] = rank
        state["last_notified_at"] = now

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_avg_voltage(row):
    return round((row["Voltage_L1"] + row["Voltage_L2"] + row["Voltage_L3"]) / 3.0, 1)


def get_vibration_rms(row):
    return round(math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2), 2)


def get_avg_current(row):
    return round((row["Current_L1"] + row["Current_L2"] + row["Current_L3"]) / 3.0, 2)


def get_row_for_device(device_id):
    motor_df = df[df["motor_id"] == device_id].reset_index(drop=True)
    idx = motor_row_index.get(device_id, 0) % len(motor_df)
    return motor_df.iloc[idx]


def advance_all_motors():
    for mid in ALL_MOTOR_IDS:
        motor_row_index[mid] = (motor_row_index.get(mid, 0) + 1) % motor_row_counts[mid]

        row = get_row_for_device(mid)
        prediction = predict_row_cached(mid, row)
        check_motor_notification(
            mid,
            prediction["condition_label"],
            prediction["fault_type"],
            prediction["probable_cause"]
        )


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
    ml_health = float(np.clip(row["sn_health_score_smooth"], 0, 100))

    temp = float(row["Temperature"])
    vib = math.sqrt(row["Vibration_X"]**2 + row["Vibration_Y"]**2 + row["Vibration_Z"]**2)
    rpm_dev = abs(float(row["Rotational_Speed"]) - 1500)
    voltage_imbalance_pct = float(row["Voltage_Imbalance"]) / float(row["Voltage_Mean"]) * 100

    def score_from(value, healthy_max, fail_at):
        if value <= healthy_max:
            return 100.0
        if value >= fail_at:
            return 0.0
        return 100.0 * (fail_at - value) / (fail_at - healthy_max)

    temp_score = score_from(temp, 75.0, 75.0 * 1.13)
    vib_score = score_from(vib, 3.5, 3.5 * 2.6)
    rpm_score = score_from(rpm_dev, 5.5 * 5.5, 5.5 * 16.4)
    volt_score = score_from(voltage_imbalance_pct, 15.0, 15.0 * 2)

    rule_health = min(temp_score, vib_score, rpm_score, volt_score)
    health_score = min(ml_health, rule_health)
    failure_probability = round(100 - health_score, 1)

    fault_info = RECOMMENDATION_TABLE.get(fault_pred, RECOMMENDATION_TABLE["Normal"])

    rul_display = round(max(0, min(rul_pred, 5000)))
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
        "rul_hours": rul_display,
        "risk_window_label": risk_window_label
    }

_prediction_cache = {}

def predict_row_cached(motor_id, row):
    idx = motor_row_index.get(motor_id, 0)
    cache_key = (motor_id, idx)
    
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]
        
    result = predict_row(row)
    _prediction_cache[cache_key] = result
    
    if len(_prediction_cache) > 40:
        oldest_key = next(iter(_prediction_cache))
        del _prediction_cache[oldest_key]
        
    return result

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

@app.route("/api/alarm-rules", methods=["GET"])
@login_required
def list_alarm_rules():
    rules = AlarmRule.query.all()
    return jsonify([r.to_dict() for r in rules])

@app.route("/api/alarm-rules", methods=["POST"])
@login_required
def add_alarm_rule():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    payload = request.get_json()
    rule = AlarmRule(
        name=payload.get("name", ""),
        parameter=payload["parameter"],
        tier=payload.get("tier", "warning"),
        device=payload.get("device", "All"),
        message=payload.get("message", ""),
        value=float(payload.get("value", 0)),
        condition=payload.get("condition", "more_than"),
        enabled=payload.get("enabled", True),
    )
    db.session.add(rule)
    db.session.commit()
    invalidate_rules_cache()
    return jsonify(rule.to_dict())

@app.route("/api/alarm-rules/<int:rule_id>", methods=["PUT"])
@login_required
def update_alarm_rule(rule_id):
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    rule = AlarmRule.query.get_or_404(rule_id)
    payload = request.get_json()
    for field in ["name", "parameter", "tier", "device", "message", "condition", "enabled"]:
        if field in payload:
            setattr(rule, field, payload[field])
    if "value" in payload:
        rule.value = float(payload["value"])
    db.session.commit()
    invalidate_rules_cache()          
    return jsonify(rule.to_dict())

@app.route("/api/alarm-rules/<int:rule_id>", methods=["DELETE"])
@login_required
def delete_alarm_rule(rule_id):
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden"}), 403
    rule = AlarmRule.query.get_or_404(rule_id)
    db.session.delete(rule)
    db.session.commit()
    invalidate_rules_cache()          
    return jsonify({"status": "deleted"})

@app.route("/api/status")
@login_required
def get_status():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    row = get_row_for_device(device_id)
    prediction = predict_row_cached(device_id, row)

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
        "voltage": get_avg_voltage(row),
        "current": get_avg_current(row),
        "pressure": round(float(row["Rotational_Speed"])),
        "vibration_x": round(float(row["Vibration_X"]), 2),
        "vibration_y": round(float(row["Vibration_Y"]), 2),
        "vibration_z": round(float(row["Vibration_Z"]), 2),
        "voltage_l1": round(float(row["Voltage_L1"]), 1),
        "voltage_l2": round(float(row["Voltage_L2"]), 1),
        "voltage_l3": round(float(row["Voltage_L3"]), 1),
        "current_l1": round(float(row["Current_L1"]), 2),
        "current_l2": round(float(row["Current_L2"]), 2),
        "current_l3": round(float(row["Current_L3"]), 2),
        "frequency": round(float(row["Frequency"]), 2),
        "power_factor": round(float(row["Power_Factor"]), 2),
        "device": str(row["motor_id"])
    })

@app.route("/api/history")
@login_required
def get_history():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    points = int(request.args.get("points", 20))

    motor_df = df[df["motor_id"] == device_id].sort_values("timestamp").reset_index(drop=True)
    idx = motor_row_index.get(device_id, 0) % len(motor_df)
    start = max(0, idx - points + 1)
    window_df = motor_df.iloc[start:idx + 1]

    return jsonify({
        "labels": window_df["timestamp"].dt.strftime("%H:%M").tolist(),
        "temperature": window_df["Temperature"].round(1).tolist(),
        "vibration": [get_vibration_rms(row) for _, row in window_df.iterrows()],
        "voltage": [get_avg_voltage(row) for _, row in window_df.iterrows()],
        "current": [get_avg_current(row) for _, row in window_df.iterrows()],
        "rpm": window_df["Rotational_Speed"].round(0).astype(int).tolist(),
        "vibration_x": window_df["Vibration_X"].round(2).tolist(),
        "vibration_y": window_df["Vibration_Y"].round(2).tolist(),
        "vibration_z": window_df["Vibration_Z"].round(2).tolist(),
        "voltage_l1": window_df["Voltage_L1"].round(1).tolist(),
        "voltage_l2": window_df["Voltage_L2"].round(1).tolist(),
        "voltage_l3": window_df["Voltage_L3"].round(1).tolist(),
        "current_l1": window_df["Current_L1"].round(2).tolist(),
        "current_l2": window_df["Current_L2"].round(2).tolist(),
        "current_l3": window_df["Current_L3"].round(2).tolist(),
    })

@app.route("/api/threshold-alerts")
@login_required
def api_threshold_alerts():
    all_flag = request.args.get("all")
    if all_flag:
        alerts = []
        for mid in ALL_MOTOR_IDS:
            row = get_row_for_device(mid)
            result = get_threshold_alerts(row)
            if result["is_labeling_candidate"]:
                alerts.append(result)
        return jsonify({"count": len(alerts), "alerts": alerts})

    device_id = request.args.get("device", ALL_MOTOR_IDS[0])
    row = get_row_for_device(device_id)
    result = get_threshold_alerts(row)
    return jsonify(result)

@app.route("/api/expert-review/submit", methods=["POST"])
@login_required
def api_submit_review():
    device_id = request.args.get("device", ALL_MOTOR_IDS[0])

    # Cegah duplicate: kalau motor ini udah ada di antrian pending, jangan submit lagi
    existing = list_reviews(status="pending", motor_id=device_id)
    if existing:
        return jsonify({"error": f"{device_id} is already in the pending review queue."}), 400

    row = get_row_for_device(device_id)
    alert = get_threshold_alerts(row)
    if not alert["is_labeling_candidate"]:
        return jsonify({"error": "Motor is currently Normal, nothing to review."}), 400
    sensor_data = row[SN_SENSOR_COLS].to_dict()
    review = submit_for_review(alert, sensor_data)
    return jsonify(review.to_dict())


@app.route("/api/expert-review/list")
@login_required
def api_list_reviews():
    status = request.args.get("status", "pending")
    reviews = list_reviews(status=status if status != "all" else None)
    return jsonify([r.to_dict() for r in reviews])


@app.route("/api/expert-review/approve", methods=["POST"])
@login_required
def api_approve_review():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    payload = request.get_json()
    review = approve_review(
        review_id=payload["review_id"],
        expert_id=current_user.username,
        expert_label=payload["expert_label"],
        expert_fault_type=payload["expert_fault_type"],
        notes=payload.get("notes"),
    )
    return jsonify(review.to_dict())


@app.route("/api/expert-review/reject", methods=["POST"])
@login_required
def api_reject_review():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    payload = request.get_json()
    review = reject_review(
        review_id=payload["review_id"],
        expert_id=current_user.username,
        notes=payload.get("notes"),
    )
    return jsonify(review.to_dict())


@app.route("/api/expert-review/stats")
@login_required
def api_review_stats():
    return jsonify(review_stats())


@app.route("/api/admin/retrain", methods=["POST"])
@login_required
def api_retrain():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    
    result = run_retraining(
        base_csv="datasets/client_training_dataset.csv", 
        triggered_by=current_user.username
    )
    return jsonify(result)

@app.route("/api/admin/models/history")
@login_required
def api_models_history():
    if not current_user.is_admin:
        return jsonify({"error": "Forbidden: admin access required"}), 403
    from retrain_pipeline import _load_registry
    try:
        reg = _load_registry()
        # Sort history newest first
        if "history" in reg:
            reg["history"] = sorted(reg["history"], key=lambda x: x.get("version", 0), reverse=True)
        return jsonify(reg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        prediction = predict_row_cached(str(row["motor_id"]), row)
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

@app.route("/api/notifications")
@login_required
def get_notifications():
    unread_count = sum(1 for n in notifications_log if not n["read"])
    return jsonify({
        "notifications": list(notifications_log),
        "unread_count": unread_count
    })


@app.route("/api/notifications/read", methods=["POST"])
@login_required
def mark_notifications_read():
    for n in notifications_log:
        n["read"] = True
    return jsonify({"status": "ok"})

@app.route("/api/motors")
@login_required
def get_motors():
    motors = []
    rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for row in rows:
        prediction = predict_row_cached(str(row["motor_id"]), row)
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
        prediction = predict_row_cached(str(row["motor_id"]), row)
        status = SENSOR_STATUS_MAP.get(prediction["condition_label"], "Normal")

        readings.append({
            "motor_id": str(row["motor_id"]),
            "motor_name": f"Induction Motor {row['motor_id']}",
            "temperature": float(row["Temperature"]),
            "vibration": get_vibration_rms(row),
            "voltage": get_avg_voltage(row),
            "current": get_avg_current(row),
            "pressure": float(row["Rotational_Speed"]),
            "status": status,
            "fault_type": prediction["fault_type"],
            "vibration_x": round(float(row["Vibration_X"]), 2),
            "vibration_y": round(float(row["Vibration_Y"]), 2),
            "vibration_z": round(float(row["Vibration_Z"]), 2),
            "voltage_l1": round(float(row["Voltage_L1"]), 1),
            "voltage_l2": round(float(row["Voltage_L2"]), 1),
            "voltage_l3": round(float(row["Voltage_L3"]), 1),
            "current_l1": round(float(row["Current_L1"]), 2),
            "current_l2": round(float(row["Current_L2"]), 2),
            "current_l3": round(float(row["Current_L3"]), 2),
        })

    return jsonify(readings)


@app.route("/api/logs")
@login_required
def get_logs():
    logs = []
    rows = [get_row_for_device(mid) for mid in ALL_MOTOR_IDS]

    for i, row in enumerate(rows):
        prediction = predict_row_cached(str(row["motor_id"]), row)
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

    motor_df = df[df["motor_id"] == device_id].sort_values("timestamp").reset_index(drop=True)
    idx = motor_row_index.get(device_id, 0) % len(motor_df)
    start = max(0, idx - 199)
    history_df = motor_df.iloc[start:idx + 1].rename(columns={"timestamp": "Timestamp"})

    result = forecast_health_and_rul(
        history_df, horizon,
        scaler=sn_scaler, rul_model=sn_rul_model, health_model=sn_health_model
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
            "Current_L1": result["Current_L1"].round(2).tolist(),
            "Current_L2": result["Current_L2"].round(2).tolist(),
            "Current_L3": result["Current_L3"].round(2).tolist(),
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

@app.route("/api/report", methods=["POST"])
@login_required
def generate_report():
    payload = request.get_json()
    selected_motors = payload.get("motors", ALL_MOTOR_IDS)
    selected_fields = payload.get("fields", ["temperature", "vibration", "voltage", "current", "rpm"])
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
        prediction = predict_row_cached(device_id, row)

        elements.append(Paragraph(f"Motor: {device_id}", styles["Heading2"]))

        table_data = [["Field", "Value"]]
        field_map = {
            "temperature": ("Temperature (C)", str(round(float(row["Temperature"]), 1))),
            "vibration": ("Vibration RMS (mm/s)", str(get_vibration_rms(row))),
            "voltage": ("Voltage Avg (V)", str(get_avg_voltage(row))),
            "current": ("Current Avg (A)", str(get_avg_current(row))),
            "rpm": ("Rotational Speed (RPM)", str(round(float(row["Rotational_Speed"])))),
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
            ("current", "Current Avg", [get_avg_current(r) for _, r in motor_hist.iterrows()], "#eab308"),
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

    if not User.query.filter_by(role="admin").first():
        admin = User(username="admin", email="admin@local", role="admin")
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("Default admin created -> username: admin | password: admin123")

    if not AlarmRule.query.first():
        default_rules = []
        for param_key, tiers in THRESHOLDS.items():
            is_low = param_key.endswith('_low')
            base_param = param_key.replace('_high', '').replace('_low', '')
            if base_param == 'RPM':
                base_param = 'Rotational_Speed'
            clean_name = base_param.replace('_', ' ') 
            for tier_name, tier_value in tiers.items():   # warning / critical / failure
                default_rules.append(AlarmRule(
                    name=f"{clean_name.title()} {'(Low)' if is_low else ''} - {tier_name.title()}".strip(),
                    parameter=base_param,
                    tier=tier_name,
                    device="All",
                    message=f"{clean_name} {'below' if is_low else 'above'} {tier_name} range",
                    value=tier_value,
                    condition="less_than" if is_low else "more_than",
                    enabled=True,
                ))
        db.session.add_all(default_rules)
        db.session.commit()
        print(f"Seeded {len(default_rules)} default alarm rules ({len(THRESHOLDS)} params x 3 tiers) from THRESHOLDS.")

# ============================================================
# CHATBOT INTEGRATION (Ollama LLM)
# ============================================================
llm_sessions = {}

@app.route("/api/chat/llm", methods=["POST"])
@login_required
def chat_llm():
    sid = str(current_user.id)
    user_input = request.json.get("message", "")

    if sid not in llm_sessions:
        llm_sessions[sid] = LLMConversationContext()

    # Rebuild tiap request supaya chatbot selalu tahu kondisi motor TERKINI,
    # bukan snapshot saat server pertama start.
    live_system_prompt = build_system_prompt()

    result = chatbot_response_llm(
        user_input,
        llm_sessions[sid],
        live_system_prompt,
        print_streaming=False
    )
    return jsonify(result)


@app.route("/api/chat/llm/save", methods=["POST"])
@login_required
def save_llm_chat():
    sid = str(current_user.id)
    if sid in llm_sessions:
        save_history(llm_sessions[sid])
    return jsonify({"status": "saved"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)