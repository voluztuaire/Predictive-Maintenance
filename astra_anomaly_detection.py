# ============================================================
# ANOMALY DETECTION MODULE
# Role    : ML Engineer — Anomaly Detection
# Name    : Salsabila Hidayat
# Project : Predictive Maintenance — Astra Otoparts WINTEQ
# Team    : Group 3 AI Bootcamp
# Tasks   : ASTRA-10, ASTRA-11, ASTRA-12
# ============================================================
# UPDATE LOG (post meeting 15 Juli 2026 — feedback Pak Wahyu & Pak Andri)
# ------------------------------------------------------------
# 1. Dataset Nadine sekarang punya Current_L1/L2/L3 (Ampere asli).
#    Sebelumnya kolom ini tidak ada — itu penyebab bug UI "396.8
#    ampere" yang sebenarnya nilai Voltage. Current sekarang jadi
#    SENSOR_COLS resmi + dipakai fault-type classifier (sinyal
#    kuat: Stator Winding & Rotor Bar sama-sama menaikkan current).
# 2. Ditambahkan get_threshold_alerts() — TERPISAH dari
#    get_anomaly_output(). Ini yang jadi "Condition-Based
#    Monitoring Alert" (murni dari check_violations(), tanpa model
#    ML) sesuai poin #1 feedback Pak Wahyu: harus ada 2 jenis alert
#    yang independen, supaya kejadian baru yang belum pernah ada
#    di training data tetap ketangkep walau model ML belum "tahu".
#    -> Siap di-expose ke endpoint baru: /api/threshold-alerts
#    (lihat flask_endpoints_example.py).
# 3. probable_cause / recommended_action sekarang diambil dari
#    CAUSE_ACTION_MAP (lookup langsung dari kolom Probable_Cause /
#    Recommended_Action di client_training_dataset.csv — sudah
#    di-approve client), dipetakan lewat Fault_Type classifier baru
#    — bukan heuristik manual (FAULT_RULES) lagi. Lebih akurat &
#    konsisten dengan teks yang disetujui client.
# 4. astra_config.py (BARU) berisi salinan tersinkron dari SENSOR_COLS,
#    THRESHOLDS, FEATURE_COLS, dan pure functions (phase_imbalance_pct,
#    check_violations, assign_threshold_label, add_engineered_features)
#    di file ini — dipakai oleh expert_validation.py & retrain_pipeline.py
#    supaya tidak perlu re-run seluruh script (~1-2 menit) hanya untuk
#    import fungsi murni. File INI (astra_anomaly_detection.py) tetap
#    jadi single source of truth; kalau threshold/fitur diubah di sini,
#    sinkronkan juga ke astra_config.py.
# ============================================================

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ['OMP_NUM_THREADS'] = '1'
import time
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    accuracy_score
)

# ============================================================
# 0. OUTPUT FOLDER SETUP
# ============================================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> outputs/{filename}")

print("=" * 60)
print("ANOMALY DETECTION MODULE — SALSABILA HIDAYAT")
print("Tasks: ASTRA-10, ASTRA-11, ASTRA-12")
print("=" * 60)
print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")

# ============================================================
# 1. LOAD DATASETS
# ============================================================
print("\n[STEP 1] Loading datasets...")

train = pd.read_csv("datasets/client_training_dataset.csv", parse_dates=["Timestamp"])
raw   = pd.read_csv("datasets/raw_sensor_new_data.csv",     parse_dates=["Timestamp"])

print(f"  Training dataset : {train.shape[0]:,} rows x {train.shape[1]} columns")
print(f"  Raw sensor data  : {raw.shape[0]:,} rows x {raw.shape[1]} columns")
print(f"  Motors           : {train['Motor_ID'].nunique()} motors")
print(f"  Missing values   : {train.isnull().sum().sum()} (training) | "
      f"{raw.isnull().sum().sum()} (raw)")
print(f"  NEW columns from Nadine : Current_L1, Current_L2, Current_L3 (Ampere)")

# Sensor features (approved by Pak Wahyu & Pak Andri)
# Current_L1/L2/L3 ditambahkan minggu ini.
SENSOR_COLS = [
    'Voltage_L1', 'Voltage_L2', 'Voltage_L3',
    'Current_L1', 'Current_L2', 'Current_L3',
    'Frequency',  'Power_Factor',
    'Temperature',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Rotational_Speed',
]
LABEL_COL     = 'Motor_State'
LABEL_ORDER   = ['Normal', 'Warning', 'Critical', 'Failure']
FAULT_COL     = 'Fault_Type_True'
FAULT_ORDER   = ['Normal', 'Rotor Bar', 'Bearing Wear', 'Misalignment', 'Stator Winding']

print(f"\n  Label distribution:")
for lbl, cnt in train[LABEL_COL].value_counts().items():
    pct = cnt / len(train) * 100
    bar = '#' * int(pct / 3)
    print(f"    {lbl:<10}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")

# Cause/Action lookup extracted directly from client-approved data
CAUSE_ACTION_MAP = (
    train[[FAULT_COL, 'Probable_Cause', 'Recommended_Action']]
    .drop_duplicates()
    .set_index(FAULT_COL)
    .to_dict(orient='index')
)
print("\n  Cause/Action map (client-approved, extracted from data):")
for k, v in CAUSE_ACTION_MAP.items():
    print(f"    [{k}] -> {v['Recommended_Action']}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[STEP 2] Exploratory Data Analysis...")

fault_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']

print("  Mean sensor values per Fault Type:")
print(train.groupby(FAULT_COL)[SENSOR_COLS].mean().round(3).T.to_string())

key_sensors = ['Temperature', 'Vibration_X', 'Vibration_Z',
               'Current_L1', 'Voltage_L1', 'Rotational_Speed']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, sensor in zip(axes.flatten(), key_sensors):
    data = [train[train[FAULT_COL] == f][sensor].dropna().values
            for f in FAULT_ORDER]
    bp   = ax.boxplot(data, patch_artist=True, labels=FAULT_ORDER)
    for patch, color in zip(bp['boxes'], fault_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(sensor, fontweight='bold')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=20)

plt.suptitle('EDA — Sensor Distribution per Fault Type (incl. Current)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig("01_eda_sensor_per_fault.png")
print("  EDA plot saved.")

# ============================================================
# ASTRA-11: SET THRESHOLD RULES FOR ANOMALY ALERTS
# ============================================================
# Threshold rules dipakai untuk 2 hal (per feedback Pak Wahyu):
#   1. Menghasilkan Threshold_Label untuk VALIDASI terhadap
#      Motor_State (di bawah).
#   2. get_threshold_alerts() -- "Condition-Based Monitoring Alert"
#      yang independen dari model ML, untuk endpoint
#      /api/threshold-alerts (dipakai expert untuk temuan anomali
#      baru yang belum ada polanya di training data -> data
#      labeling berikutnya).
# ============================================================
print("\n[ASTRA-11] Setting Threshold Rules for Anomaly Alerts...")

THRESHOLDS = {
    # TEMPERATURE (degC)
    'Temperature': {'warning': 51.0, 'critical': 57.0, 'failure': 66.0},
    # VIBRATION (mm/s)
    'Vibration_X': {'warning': 1.80, 'critical': 3.60, 'failure': 5.50},
    'Vibration_Y': {'warning': 1.80, 'critical': 3.60, 'failure': 5.50},
    'Vibration_Z': {'warning': 1.90, 'critical': 3.90, 'failure': 5.80},
    # VOLTAGE per phase (V) — nominal 400V
    'Voltage_L1_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L2_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L3_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L1_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L2_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L3_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    # ------------------------------------------------------------
    # CURRENT per phase (A) — justification (per review feedback)
    # ------------------------------------------------------------
    # There is no single universal "% over nominal" standard for current
    # the way there is for Voltage (+-5% nameplate) or Frequency (+-1%),
    # because safe operating current is defined relative to a motor's
    # nameplate Full Load Amps (FLA), not a fixed absolute value. We do
    # not have an explicit FLA/nameplate field in this dataset, so we
    # ESTIMATE FLA as the empirical Normal-condition mean current
    # (6.484 A across L1/L2/L3 — see EDA "Mean sensor values per Fault
    # Type") and cross-check the resulting thresholds as %FLA against
    # typical thermal-overload-relay trip bands referenced in
    # IEC 60947-4-1 / NEMA ICS 2 (Class 10/20 relays commonly trip in
    # the ~105-125% FLA range for sustained overcurrent, well below
    # locked-rotor levels of 500-700% FLA):
    #
    #   Tier      Value   Dataset basis                   Approx %FLA
    #   warning   7.3 A   just above Normal P95 (7.23)      ~112.6%
    #   critical  7.9 A   ~ Critical-state P95 (8.22)       ~121.8%
    #   failure   9.0 A   below Failure-state P95 (9.38)    ~138.8%
    #
    # ~112-122% FLA sits inside the typical overload-relay caution/trip
    # band, and 139% FLA is a conservative sustained-overcurrent failure
    # cutoff — consistent with, though not a substitute for, the actual
    # nameplate FLA + relay curve once available. Values live in
    # thresholds.json so admin can recalibrate once real nameplate FLA
    # is confirmed by the client (see README, halaman admin).
    'Current_L1_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    'Current_L2_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    'Current_L3_high': {'warning': 7.3, 'critical': 7.9, 'failure': 9.0},
    # FREQUENCY (Hz) — nominal 50Hz
    'Frequency_high': {'warning': 50.10, 'critical': 50.15, 'failure': 50.20},
    'Frequency_low':  {'warning': 49.90, 'critical': 49.85, 'failure': 49.80},
    # ROTATIONAL SPEED (RPM) — lower bound only
    'RPM_low': {'warning': 1455.0, 'critical': 1450.0, 'failure': 1445.0},
    # ------------------------------------------------------------
    # PHASE IMBALANCE (%) — NEW, NEMA MG-1 style formula (see
    # add_engineered_features() / _engineer_single()):
    #   %Unbalance = 100 * max(|phase_i - avg|) / avg
    # NEMA MG-1 (14.35) recommends continuous-duty VOLTAGE unbalance
    # should not exceed 1%, with a derating curve applying up to 5%.
    # There is no equally codified number for CURRENT unbalance, but
    # common industry practice (the "current unbalance runs roughly
    # 6-10x the voltage unbalance %" rule of thumb) is used alongside
    # actual dataset separation (Current_Imbalance_Pct mean 0.24%
    # Normal vs 1.13% Failure; Voltage_Imbalance_Pct stays under ~1.6%
    # for all states here, so its tiers sit close to the NEMA 1%/5%
    # reference points rather than dataset percentiles).
    'Voltage_Imbalance_Pct': {'warning': 0.5, 'critical': 1.0, 'failure': 2.5},
    'Current_Imbalance_Pct': {'warning': 0.6, 'critical': 1.8, 'failure': 3.5},
    'Power_Factor_low': {'warning': 0.85, 'critical': 0.75, 'failure': 0.65},
}

print("\n  Threshold rules:")
print(f"  {'Parameter':<22} {'Warning':>10} {'Critical':>10} {'Failure':>10}  Direction")
print("  " + "-" * 65)
for param, tiers in THRESHOLDS.items():
    direction = 'LOWER BOUND' if param.endswith('_low') else 'UPPER BOUND'
    print(f"  {param:<22} {tiers['warning']:>10} {tiers['critical']:>10} "
          f"{tiers['failure']:>10}  {direction}")


def phase_imbalance_pct(a: float, b: float, c: float) -> float:
    """
    NEMA MG-1 style percentage phase imbalance:
        %Unbalance = 100 * max(|phase_i - avg|) / avg
    Works for both Voltage (V) and Current (A) triplets.
    Replaces the earlier std()-based imbalance metric, which does not
    correspond to any published industrial definition.
    """
    avg = (a + b + c) / 3.0
    if avg == 0:
        return 0.0
    max_dev = max(abs(a - avg), abs(b - avg), abs(c - avg))
    return 100.0 * max_dev / avg


def phase_imbalance_pct_vec(df: pd.DataFrame, cols: list) -> pd.Series:
    """Vectorized NEMA MG-1 style % imbalance for a DataFrame (3 columns)."""
    avg = df[cols].mean(axis=1)
    max_dev = df[cols].sub(avg, axis=0).abs().max(axis=1)
    return (100.0 * max_dev / avg.replace(0, np.nan)).fillna(0.0)


def check_violations(row: dict) -> dict:
    """
    Check which sensor parameters violate which threshold tier.

    Pure function, no I/O, no model dependency -> safe to call
    directly from Flask (El Shaddai) for the /api/threshold-alerts
    endpoint. Returns lists of violated parameter names per tier.

    `row` may contain either just the raw SENSOR_COLS, or raw sensors
    plus engineered Voltage_Imbalance_Pct / Current_Imbalance_Pct. If
    the imbalance keys are missing, they are computed on the fly from
    the raw phase values so this function stays correct either way.
    """
    v = {'warning': [], 'critical': [], 'failure': []}

    if 'Voltage_Imbalance_Pct' not in row and all(k in row for k in
            ('Voltage_L1', 'Voltage_L2', 'Voltage_L3')):
        row = dict(row)
        row['Voltage_Imbalance_Pct'] = phase_imbalance_pct(
            row['Voltage_L1'], row['Voltage_L2'], row['Voltage_L3'])
    if 'Current_Imbalance_Pct' not in row and all(k in row for k in
            ('Current_L1', 'Current_L2', 'Current_L3')):
        row = dict(row)
        row['Current_Imbalance_Pct'] = phase_imbalance_pct(
            row['Current_L1'], row['Current_L2'], row['Current_L3'])

    def upper(feat, key):
        val = row.get(feat, 0)
        t   = THRESHOLDS[key]
        if   val > t['failure']:   v['failure'].append(feat)
        elif val > t['critical']:  v['critical'].append(feat)
        elif val > t['warning']:   v['warning'].append(feat)

    def lower(feat, key):
        val = row.get(feat, 9999)
        t   = THRESHOLDS[key]
        if   val < t['failure']:   v['failure'].append(f"{feat}(low)")
        elif val < t['critical']:  v['critical'].append(f"{feat}(low)")
        elif val < t['warning']:   v['warning'].append(f"{feat}(low)")

    upper('Temperature',    'Temperature')
    upper('Vibration_X',    'Vibration_X')
    upper('Vibration_Y',    'Vibration_Y')
    upper('Vibration_Z',    'Vibration_Z')
    upper('Voltage_L1',     'Voltage_L1_high')
    upper('Voltage_L2',     'Voltage_L2_high')
    upper('Voltage_L3',     'Voltage_L3_high')
    upper('Current_L1',     'Current_L1_high')
    upper('Current_L2',     'Current_L2_high')
    upper('Current_L3',     'Current_L3_high')
    upper('Frequency',      'Frequency_high')
    upper('Voltage_Imbalance_Pct', 'Voltage_Imbalance_Pct')
    upper('Current_Imbalance_Pct', 'Current_Imbalance_Pct')
    lower('Voltage_L1',       'Voltage_L1_low')
    lower('Voltage_L2',       'Voltage_L2_low')
    lower('Voltage_L3',       'Voltage_L3_low')
    lower('Frequency',        'Frequency_low')
    lower('Rotational_Speed', 'RPM_low')
    return v


def assign_threshold_label(row: dict) -> str:
    """
    Assign 4-class condition label from threshold violations.

    Decision logic (conservative — minimize false alarms):
      Failure  : ANY parameter in failure zone OR 4+ total violations
      Critical : 1+ parameter in critical zone
      Warning  : 1+ parameter in warning zone
      Normal   : no violations at all
    """
    viol = check_violations(row)
    n_f  = len(viol['failure'])
    n_c  = len(viol['critical'])
    n_w  = len(viol['warning'])
    n_t  = n_f + n_c + n_w

    if n_f >= 1 or n_t >= 4: return 'Failure'
    elif n_c >= 1:            return 'Critical'
    elif n_w >= 1:            return 'Warning'
    else:                     return 'Normal'


# Generate threshold-based labels (for validation only)
print("\n  Generating threshold-based labels...")
train['Threshold_Label'] = train[SENSOR_COLS].apply(
    lambda r: assign_threshold_label(r.to_dict()), axis=1
)

print("\n  Validation — Threshold Labels vs Motor_State:")
cross = pd.crosstab(train[LABEL_COL], train['Threshold_Label'], margins=True)
print(cross.to_string())

agree = (train[LABEL_COL] == train['Threshold_Label']).sum()
print(f"\n  Agreement rate: {agree:,} / {len(train):,} "
      f"({agree/len(train)*100:.1f}%)")
sev_map = {'Normal': 0, 'Warning': 1, 'Critical': 2, 'Failure': 3}
sev_diff = (train['Threshold_Label'].map(sev_map) - train[LABEL_COL].map(sev_map)).abs()
print(f"  Within +-1 severity tier: {(sev_diff <= 1).mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts_nadine = [train[LABEL_COL].value_counts().get(l, 0) for l in LABEL_ORDER]
colors_cond   = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[0].bar(LABEL_ORDER, counts_nadine, color=colors_cond)
axes[0].set_title("Motor_State Distribution (Client Ground Truth)", fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts_nadine):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

counts_thresh = [train['Threshold_Label'].value_counts().get(l, 0) for l in LABEL_ORDER]
axes[1].bar(LABEL_ORDER, counts_thresh, color=colors_cond)
axes[1].set_title("Threshold_Label Distribution\n(Our Rules — ASTRA-11)", fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(counts_thresh):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

plt.suptitle('ASTRA-11: Label Comparison — Client Dataset vs Our Threshold Rules',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("02_astra11_threshold_label_comparison.png")

sensor_means = train.groupby(LABEL_COL)[SENSOR_COLS].mean()
sensor_norm  = (sensor_means - sensor_means.min()) / \
               (sensor_means.max() - sensor_means.min())
sensor_norm  = sensor_norm.reindex(LABEL_ORDER)

plt.figure(figsize=(15, 5))
sns.heatmap(sensor_norm.T, annot=True, fmt='.2f',
            cmap='RdYlGn_r', linewidths=0.5,
            cbar_kws={'label': 'Normalized Mean (0=low, 1=high)'})
plt.title('Sensor Mean Value per Condition (Normalized)\n'
          'Darker red = higher relative value', fontweight='bold')
plt.tight_layout()
save_fig("03_heatmap_sensor_per_condition.png")
print("  ASTRA-11 complete.")

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
print("\n[STEP 3] Feature Engineering...")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # NEMA MG-1 style %imbalance (replaces old std()-based metric, which
    # has no standard industrial interpretation — see THRESHOLDS comment
    # above for rationale).
    df['Voltage_Imbalance_Pct'] = phase_imbalance_pct_vec(
        df, ['Voltage_L1', 'Voltage_L2', 'Voltage_L3'])
    df['Current_Imbalance_Pct'] = phase_imbalance_pct_vec(
        df, ['Current_L1', 'Current_L2', 'Current_L3'])
    df['Vibration_Total']   = np.sqrt(df['Vibration_X']**2 +
                                       df['Vibration_Y']**2 +
                                       df['Vibration_Z']**2)
    df['Voltage_Mean']      = df[['Voltage_L1','Voltage_L2','Voltage_L3']].mean(axis=1)
    df['Current_Mean']      = df[['Current_L1','Current_L2','Current_L3']].mean(axis=1)
    df['RPM_Deviation']     = abs(df['Rotational_Speed'] - 1500)
    return df

train = add_engineered_features(train)
raw   = add_engineered_features(raw)

FEATURE_COLS = SENSOR_COLS + [
    'Voltage_Imbalance_Pct',
    'Current_Imbalance_Pct',
    'Vibration_Total',
    'Voltage_Mean',
    'Current_Mean',
    'RPM_Deviation',
]

print(f"  Original sensor features   : {len(SENSOR_COLS)} (incl. 3 new Current sensors)")
print(f"  Engineered features added  : 6")
print(f"    Voltage_Imbalance_Pct = NEMA MG-1 %unbalance = 100*max|Vi-avg|/avg  [UPDATED]")
print(f"    Current_Imbalance_Pct = NEMA MG-1 %unbalance = 100*max|Ii-avg|/avg  [UPDATED]")
print(f"    Vibration_Total       = sqrt(X^2 + Y^2 + Z^2)")
print(f"    Voltage_Mean          = mean(L1, L2, L3)")
print(f"    Current_Mean          = mean(L1, L2, L3)")
print(f"    RPM_Deviation         = |RPM - 1500|")
print(f"  Total features             : {len(FEATURE_COLS)}")

# ============================================================
# ASTRA-10: TRAIN ANOMALY DETECTION MODEL
# ============================================================
print("\n[ASTRA-10] Training Anomaly Detection Models...")

X = train[FEATURE_COLS].copy()
y = train[LABEL_COL].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows (80/20 stratified)")

y_test_bin  = (y_test != 'Normal').astype(int)

comparison  = {}

# --- Model 1: IsolationForest ---
print("\n  [1/3] IsolationForest (unsupervised)...")
contam  = min(0.499, round((y_train != 'Normal').sum() / len(y_train), 3))
X_norm  = X_train_sc[y_train == 'Normal']
t0      = time.time()
iso     = IsolationForest(contamination=contam, n_estimators=100,
                          random_state=42, n_jobs=-1)
iso.fit(X_norm)
iso_t   = round(time.time() - t0, 2)

iso_pred    = iso.predict(X_test_sc)
iso_bin     = (iso_pred == -1).astype(int)
prec = precision_score(y_test_bin, iso_bin, zero_division=0)
rec  = recall_score(y_test_bin, iso_bin, zero_division=0)
f1   = f1_score(y_test_bin, iso_bin, zero_division=0)
fa   = int(((iso_bin == 1) & (y_test_bin == 0)).sum())

comparison['IsolationForest'] = {
    'type': 'Unsupervised (binary)',
    'precision': round(prec, 4), 'recall': round(rec, 4),
    'f1': round(f1, 4), 'f1_macro': round(f1, 4),
    'false_alarms': fa, 'train_time': iso_t,
}
print(f"    Done in {iso_t}s | F1={f1:.4f} | False Alarms={fa:,}")

# --- Model 2: RandomForestClassifier ---
print("  [2/3] RandomForestClassifier (supervised 4-class)...")
t0  = time.time()
rf  = RandomForestClassifier(n_estimators=100, max_depth=15,
                              min_samples_leaf=5, class_weight='balanced',
                              random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
rf_t   = round(time.time() - t0, 2)
rf_pred = rf.predict(X_test_sc)
rf_bin  = (rf_pred != 'Normal').astype(int)

comparison['RandomForest'] = {
    'type': 'Supervised (4-class)',
    'accuracy': round(accuracy_score(y_test, rf_pred), 4),
    'precision': round(precision_score(y_test_bin, rf_bin, zero_division=0), 4),
    'recall': round(recall_score(y_test_bin, rf_bin, zero_division=0), 4),
    'f1': round(f1_score(y_test_bin, rf_bin, zero_division=0), 4),
    'f1_macro': round(f1_score(y_test, rf_pred, average='macro', zero_division=0), 4),
    'false_alarms': int(((rf_bin==1)&(y_test_bin==0)).sum()),
    'train_time': rf_t,
}
print(f"    Done in {rf_t}s | Accuracy={comparison['RandomForest']['accuracy']:.4f} "
      f"| F1 Macro={comparison['RandomForest']['f1_macro']:.4f}")

# --- Model 3: Logistic Regression ---
print("  [3/3] LogisticRegression (linear baseline)...")
t0  = time.time()
lr  = LogisticRegression(max_iter=1000, class_weight='balanced',
                          random_state=42, n_jobs=-1)
lr.fit(X_train_sc, y_train)
lr_t    = round(time.time() - t0, 2)
lr_pred = lr.predict(X_test_sc)
lr_bin  = (lr_pred != 'Normal').astype(int)

comparison['LogisticRegression'] = {
    'type': 'Supervised (4-class)',
    'accuracy': round(accuracy_score(y_test, lr_pred), 4),
    'precision': round(precision_score(y_test_bin, lr_bin, zero_division=0), 4),
    'recall': round(recall_score(y_test_bin, lr_bin, zero_division=0), 4),
    'f1': round(f1_score(y_test_bin, lr_bin, zero_division=0), 4),
    'f1_macro': round(f1_score(y_test, lr_pred, average='macro', zero_division=0), 4),
    'false_alarms': int(((lr_bin==1)&(y_test_bin==0)).sum()),
    'train_time': lr_t,
}
print(f"    Done in {lr_t}s | Accuracy={comparison['LogisticRegression']['accuracy']:.4f} "
      f"| F1 Macro={comparison['LogisticRegression']['f1_macro']:.4f}")

print()
print("  " + "=" * 75)
print("  MODEL COMPARISON TABLE (ASTRA-10)")
print("  " + "=" * 75)
print(f"  {'Model':<22} {'Type':<25} {'Accuracy':>9} {'F1 Mac':>7} "
      f"{'F1 Bin':>7} {'FAlarms':>9} {'Time':>6}")
print("  " + "-" * 75)
for m, r in comparison.items():
    acc = r.get('accuracy', '-')
    acc_str = f"{acc:.4f}" if isinstance(acc, float) else f"{'N/A':>9}"
    print(f"  {m:<22} {r['type']:<25} {acc_str:>9} "
          f"{r['f1_macro']:>7.4f} {r['f1']:>7.4f} "
          f"{r['false_alarms']:>9,} {r['train_time']:>5.1f}s")

best_name = max(['RandomForest', 'LogisticRegression'],
                key=lambda m: comparison[m]['f1_macro'])
best_clf  = rf if best_name == 'RandomForest' else lr
print(f"\n  Best model (highest F1 Macro): {best_name}")

# --- Bonus: Fault_Type classifier (for accurate cause/action lookup) ---
print("\n  [Bonus] Training Fault_Type classifier (for cause/action lookup)...")
# LEAKAGE FIX (per review feedback): the previous version called a
# *separate* train_test_split() for y_fault (different stratification
# column -> different row partition) but reused `scaler`, which was
# fit()-ed on X_train from the CONDITION split. That meant some rows
# ending up in Xf_test could have already contributed to the scaler's
# fitted mean/std via the condition split's X_train -- a genuine (if
# mild, since it's only global feature statistics, not label info)
# train/test leakage.
#
# Fix: reuse the EXACT SAME row partition (X_train/X_test, same
# indices) used for the condition classifier, instead of a fresh
# split. This guarantees the scaler (fit only on X_train) never sees
# any row that ends up in the fault-type test set either.
yf_train = train.loc[X_train.index, FAULT_COL]
yf_test  = train.loc[X_test.index,  FAULT_COL]
Xf_train_sc = X_train_sc   # same scaled arrays as condition classifier
Xf_test_sc  = X_test_sc    # (scaler fit only on X_train -> no leakage)

fault_clf = RandomForestClassifier(
    n_estimators=150, max_depth=18, min_samples_leaf=3,
    class_weight='balanced', random_state=42, n_jobs=-1
)
fault_clf.fit(Xf_train_sc, yf_train)
fault_pred = fault_clf.predict(Xf_test_sc)
fault_acc  = accuracy_score(yf_test, fault_pred)
fault_prec = precision_score(yf_test, fault_pred, average='macro', zero_division=0)
fault_rec  = recall_score(yf_test, fault_pred, average='macro', zero_division=0)
fault_f1m  = f1_score(yf_test, fault_pred, average='macro', zero_division=0)
print(f"    Fault_Type accuracy  : {fault_acc:.4f}")
print(f"    Precision (macro)    : {fault_prec:.4f}")
print(f"    Recall (macro)       : {fault_rec:.4f}")
print(f"    F1 (macro)           : {fault_f1m:.4f}")
print(f"    Train/test split     : SAME partition as condition classifier "
      f"(no independent re-split -> no scaler leakage)")

# Save all artifacts
joblib.dump(best_clf,          os.path.join(OUTPUT_DIR, 'condition_classifier.joblib'))
joblib.dump(fault_clf,         os.path.join(OUTPUT_DIR, 'fault_type_classifier.joblib'))
joblib.dump(iso,               os.path.join(OUTPUT_DIR, 'isolation_forest.joblib'))
joblib.dump(scaler,            os.path.join(OUTPUT_DIR, 'condition_scaler.joblib'))
joblib.dump(CAUSE_ACTION_MAP,  os.path.join(OUTPUT_DIR, 'cause_action_map.joblib'))
with open(os.path.join(OUTPUT_DIR, 'cause_action_map.json'), 'w') as f:
    json.dump(CAUSE_ACTION_MAP, f, indent=2)
with open(os.path.join(OUTPUT_DIR, 'thresholds.json'), 'w') as f:
    json.dump(THRESHOLDS, f, indent=2)

print(f"\n  Artifacts saved to outputs/:")
print(f"    condition_classifier.joblib   ({best_name})")
print(f"    fault_type_classifier.joblib  (RandomForest, 5-class)")
print(f"    isolation_forest.joblib")
print(f"    condition_scaler.joblib")
print(f"    cause_action_map.joblib / .json")
print(f"    thresholds.json               <- for El's admin threshold-adjust UI")

# ============================================================
# ASTRA-12: TEST AND VALIDATE ANOMALY DETECTION RESULTS
# ============================================================
print("\n[ASTRA-12] Validating Anomaly Detection Results...")

best_pred = rf_pred if best_name == 'RandomForest' else lr_pred

print(f"\n  Classification Report — {best_name} (Condition Label):")
print(classification_report(y_test, best_pred,
                             target_names=LABEL_ORDER, zero_division=0))

print(f"\n  Classification Report — Fault_Type classifier:")
print(classification_report(yf_test, fault_pred,
                             target_names=FAULT_ORDER, zero_division=0))

fault_report = classification_report(yf_test, fault_pred, target_names=FAULT_ORDER,
                                     zero_division=0, output_dict=True)
print("  Per-class Precision / Recall / F1 — Fault_Type classifier:")
print(f"  {'Class':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
for cls in FAULT_ORDER:
    r = fault_report[cls]
    print(f"  {cls:<16} {r['precision']:>10.4f} {r['recall']:>8.4f} "
          f"{r['f1-score']:>8.4f} {int(r['support']):>8}")

fig, axes = plt.subplots(1, 3, figsize=(22, 6))
cm_best = confusion_matrix(y_test, best_pred, labels=LABEL_ORDER)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\n{best_name} (4-Class Condition)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_iso = confusion_matrix(y_test_bin, iso_bin)
sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal','Anomaly'],
            yticklabels=['Normal','Anomaly'], ax=axes[1])
axes[1].set_title('Confusion Matrix\nIsolationForest (Binary)', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

cm_fault = confusion_matrix(yf_test, fault_pred, labels=FAULT_ORDER)
sns.heatmap(cm_fault, annot=True, fmt='d', cmap='Purples',
            xticklabels=FAULT_ORDER, yticklabels=FAULT_ORDER, ax=axes[2])
axes[2].set_title('Confusion Matrix\nFault_Type Classifier (5-Class)', fontweight='bold')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].tick_params(axis='x', rotation=25)

plt.suptitle('ASTRA-12: Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("04_astra12_confusion_matrices.png")

models  = list(comparison.keys())
f1_vals = [comparison[m]['f1_macro'] for m in models]
fa_vals = [comparison[m]['false_alarms'] for m in models]
x       = np.arange(len(models))
w       = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(x - w/2, f1_vals, w, label='F1 Macro Score', color='steelblue')
ax1.set_ylabel('F1 Macro Score', color='steelblue', fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
for bar, val in zip(bars, f1_vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01, f'{val:.4f}',
             ha='center', fontsize=9, color='steelblue')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + w/2, fa_vals, w, label='False Alarms', color='salmon', alpha=0.8)
ax2.set_ylabel('False Alarms (count)', color='salmon', fontsize=11)
for bar, val in zip(bars2, fa_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(fa_vals)*0.01, f'{val:,}',
             ha='center', fontsize=9, color='salmon')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title('ASTRA-12: Model Comparison — F1 Score vs False Alarms',
          fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("05_astra12_model_comparison.png")

if best_name == 'RandomForest':
    imp = pd.Series(rf.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    imp.plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Feature Importance — RandomForest\n(Condition Classification)',
                      fontweight='bold')
    axes[0].set_xlabel('Importance Score')
    for i, (val, name) in enumerate(zip(imp.values, imp.index)):
        axes[0].text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)

    imp_df = imp.sort_values(ascending=False).to_frame('Importance')
    sns.heatmap(imp_df, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=axes[1], cbar_kws={'label': 'Score'}, linewidths=0.5)
    axes[1].set_title('Feature Importance Heatmap', fontweight='bold')

    plt.suptitle('ASTRA-12: Feature Importance Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("06_astra12_feature_importance.png")

print("\n  Generating learning curve (this takes ~1 min)...")
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=30, max_depth=8,
                           random_state=42, n_jobs=-1),
    X_train_sc, y_train,
    cv=2, scoring='f1_macro',
    train_sizes=np.linspace(0.1, 1.0, 5),
    n_jobs=-1
)
train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)

plt.figure(figsize=(9, 5))
plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training Score')
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std,
                 alpha=0.2, color='steelblue')
plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation Score (CV)')
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std,
                 alpha=0.2, color='darkorange')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Macro Score')
plt.title('ASTRA-12: Learning Curve\n(Training vs Cross-Validation Score)',
          fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("07_astra12_learning_curve.png")

report_dict = classification_report(y_test, best_pred, target_names=LABEL_ORDER,
                                    zero_division=0, output_dict=True)
per_class_f1 = {lbl: report_dict[lbl]['f1-score'] for lbl in LABEL_ORDER}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(LABEL_ORDER, list(per_class_f1.values()), color=colors_cond)
ax.set_title(f'ASTRA-12: Per-Class F1 Score — {best_name}', fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, per_class_f1.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontsize=10)
plt.tight_layout()
save_fig("08_astra12_per_class_f1.png")

print("  ASTRA-12 validation complete.")

# ============================================================
# 4a. AI PREDICTIVE ALERT — get_anomaly_output()
# ============================================================
# Dipanggil El Shaddai dari /api/status (dashboard utama + AI Alert
# tab). Sumber: output model ML (RandomForest condition_classifier +
# fault_type_classifier). Ini prediksi berbasis PATTERN yang sudah
# dipelajari dari training data.
# ============================================================
print("\n[STEP 4a] Building AI Predictive Alert function...")

clf_loaded    = joblib.load(os.path.join(OUTPUT_DIR, 'condition_classifier.joblib'))
fault_loaded  = joblib.load(os.path.join(OUTPUT_DIR, 'fault_type_classifier.joblib'))
iso_loaded    = joblib.load(os.path.join(OUTPUT_DIR, 'isolation_forest.joblib'))
scaler_loaded = joblib.load(os.path.join(OUTPUT_DIR, 'condition_scaler.joblib'))
cause_map     = joblib.load(os.path.join(OUTPUT_DIR, 'cause_action_map.joblib'))


def _engineer_single(sensor_data: dict) -> dict:
    """Shared feature engineering for a single sensor reading."""
    e = dict(sensor_data)
    e['Voltage_Imbalance_Pct'] = phase_imbalance_pct(
        e.get('Voltage_L1', 400), e.get('Voltage_L2', 400), e.get('Voltage_L3', 400),
    )
    e['Current_Imbalance_Pct'] = phase_imbalance_pct(
        e.get('Current_L1', 6.5), e.get('Current_L2', 6.5), e.get('Current_L3', 6.5),
    )
    e['Vibration_Total'] = float(np.sqrt(
        e.get('Vibration_X', 0)**2 + e.get('Vibration_Y', 0)**2 + e.get('Vibration_Z', 0)**2
    ))
    e['Voltage_Mean'] = float(np.mean([
        e.get('Voltage_L1', 400), e.get('Voltage_L2', 400), e.get('Voltage_L3', 400),
    ]))
    e['Current_Mean'] = float(np.mean([
        e.get('Current_L1', 6.5), e.get('Current_L2', 6.5), e.get('Current_L3', 6.5),
    ]))
    e['RPM_Deviation'] = float(abs(e.get('Rotational_Speed', 1500) - 1500))
    return e


def get_anomaly_output(sensor_data: dict) -> dict:
    """
    AI PREDICTIVE ALERT — full ML inference for one sensor reading.

    Called by:
      - El Shaddai: /api/status endpoint (main dashboard + AI Alert tab,
        including probable_cause/recommended_action per feedback poin #5)
      - Shania: to get condition_label as input for RUL/Health Score model

    Args:
        sensor_data (dict): one row of sensor readings.
            Required keys: Voltage_L1/L2/L3, Current_L1/L2/L3, Frequency,
                           Power_Factor, Temperature, Vibration_X/Y/Z,
                           Rotational_Speed

    Returns:
        dict:
            condition_label     : Normal / Warning / Critical / Failure
            status_color        : green / yellow / orange / red
            if_anomaly_score    : IsolationForest raw score (float)
            if_is_anomaly       : bool
            fault_type          : predicted fault type (Normal/Rotor Bar/...)
            violated_warning/critical/failure : list of parameter names
            total_violations    : int
            probable_cause      : string (client-approved text)
            recommended_action  : string (client-approved text)
            alert_message       : string (English)
            source              : 'ai_model'  <- for UI to distinguish
                                    from get_threshold_alerts()
    """
    e = _engineer_single(sensor_data)

    arr    = np.array([[e.get(f, 0) for f in FEATURE_COLS]])
    arr_sc = scaler_loaded.transform(arr)

    condition   = clf_loaded.predict(arr_sc)[0]
    if_score    = float(iso_loaded.score_samples(arr_sc)[0])
    if_is_anom  = bool(iso_loaded.predict(arr_sc)[0] == -1)

    viol = check_violations(e)
    n_t  = len(viol['warning']) + len(viol['critical']) + len(viol['failure'])

    fault_type = fault_loaded.predict(arr_sc)[0]
    if condition == 'Normal':
        cause, action = '-', cause_map['Normal']['Recommended_Action']
    else:
        if fault_type == 'Normal':
            proba   = fault_loaded.predict_proba(arr_sc)[0]
            classes = fault_loaded.classes_
            ranked  = sorted(zip(classes, proba), key=lambda x: -x[1])
            fault_type = next((c for c, p in ranked if c != 'Normal'), ranked[0][0])
        cause  = cause_map[fault_type]['Probable_Cause']
        action = cause_map[fault_type]['Recommended_Action']

    color_map = {'Normal': 'green', 'Warning': 'yellow', 'Critical': 'orange', 'Failure': 'red'}
    msg_map = {
        'Normal'  : 'Motor operating within normal parameters.',
        'Warning' : f'Warning: {n_t} parameter(s) above normal range. Monitor closely.',
        'Critical': f'Critical: {len(viol["critical"])+len(viol["failure"])} parameter(s) '
                    f'in danger zone. Prepare maintenance action.',
        'Failure' : 'FAILURE ALERT: Motor in critical failure zone. '
                    'Stop operation and perform immediate inspection!',
    }

    return {
        'condition_label'    : condition,
        'status_color'       : color_map[condition],
        'if_anomaly_score'   : round(if_score, 4),
        'if_is_anomaly'      : if_is_anom,
        'fault_type'         : fault_type,
        'violated_warning'   : viol['warning'],
        'violated_critical'  : viol['critical'],
        'violated_failure'   : viol['failure'],
        'total_violations'   : n_t,
        'probable_cause'     : cause,
        'recommended_action' : action,
        'alert_message'      : msg_map[condition],
        'source'             : 'ai_model',
    }


# ============================================================
# 4b. CONDITION-BASED MONITORING ALERT — get_threshold_alerts()
# ============================================================
# INI FUNGSI BARU sesuai feedback Pak Wahyu poin #1.
#
# Beda dengan get_anomaly_output() (AI Predictive Alert):
#   - TIDAK pakai model ML sama sekali -> murni threshold sensor.
#   - Tujuannya BUKAN kasih rekomendasi maintenance, tapi DATA
#     COLLECTION: nangkep kejadian sensor keluar batas yang mungkin
#     belum pernah dilihat model (pattern baru). Expert lalu cek
#     fisik motor & kasih label manual -> jadi training data baru.
#   - Independen total dari get_anomaly_output(), supaya kalau
#     model ML "buta" terhadap suatu pola, threshold alert tetap
#     jalan sebagai jaring pengaman.
#
# Dipanggil El Shaddai dari endpoint BARU: /api/threshold-alerts
# (lihat flask_endpoints_example.py untuk contoh route-nya).
# ============================================================
print("[STEP 4b] Building Condition-Based Monitoring Alert function...")


def get_threshold_alerts(sensor_data: dict, motor_id: str = None,
                          timestamp: str = None) -> dict:
    """
    CONDITION-BASED MONITORING ALERT — raw threshold check, no ML.

    Args:
        sensor_data (dict): one row of sensor readings (same schema as
            get_anomaly_output).
        motor_id (str, optional): motor identifier, passthrough for logging.
        timestamp (str, optional): reading timestamp, passthrough for logging.

    Returns:
        dict:
            motor_id           : passthrough
            timestamp          : passthrough
            condition_label    : Normal / Warning / Critical / Failure
                                  (derived purely from threshold rules,
                                  may DIFFER from get_anomaly_output's
                                  condition_label -- that's expected and
                                  by design, see feedback poin #1)
            status_color       : green / yellow / orange / red
            violations         : list of dicts, one per breached parameter:
                                  {parameter, tier, actual_value, threshold}
            total_violations   : int
            is_labeling_candidate : bool -- True kalau condition != Normal,
                                  dipakai El buat highlight baris yang
                                  layak di-review & dikasih label manual
                                  expert (data collection loop poin #1)
            source              : 'threshold_rule'  <- untuk UI bedain
                                   dari get_anomaly_output()
    """
    e = _engineer_single(sensor_data)
    viol = check_violations(e)
    condition = assign_threshold_label(e)

    # Build detailed per-parameter violation list with actual values
    # and the threshold that was crossed (buat expert review / labeling UI)
    detail = []
    for tier in ('warning', 'critical', 'failure'):
        for param in viol[tier]:
            base_param = param.replace('(low)', '')
            actual_val = e.get(base_param)
            key = base_param
            if base_param in ('Voltage_L1', 'Voltage_L2', 'Voltage_L3'):
                key = f"{base_param}_{'low' if '(low)' in param else 'high'}"
            elif base_param == 'Frequency':
                key = f"Frequency_{'low' if '(low)' in param else 'high'}"
            elif base_param == 'Rotational_Speed':
                key = 'RPM_low'
            elif base_param in ('Current_L1', 'Current_L2', 'Current_L3'):
                key = f"{base_param}_high"
            threshold_val = THRESHOLDS.get(key, {}).get(tier)
            detail.append({
                'parameter'    : param,
                'tier'         : tier,
                'actual_value' : round(float(actual_val), 3) if actual_val is not None else None,
                'threshold'    : threshold_val,
            })

    color_map = {'Normal': 'green', 'Warning': 'yellow', 'Critical': 'orange', 'Failure': 'red'}
    n_t = len(viol['warning']) + len(viol['critical']) + len(viol['failure'])

    return {
        'motor_id'             : motor_id,
        'timestamp'            : timestamp,
        'condition_label'      : condition,
        'status_color'         : color_map[condition],
        'violations'           : detail,
        'total_violations'     : n_t,
        'is_labeling_candidate': condition != 'Normal',
        'source'               : 'threshold_rule',
    }


# Quick sanity test — compare both alert sources on 4 sample rows
print("\n  Sanity check: AI alert vs Threshold alert (may legitimately differ):")
print("  " + "-" * 66)
for label in LABEL_ORDER:
    sample_rows = train[train[LABEL_COL] == label]
    if len(sample_rows) == 0:
        continue
    row = sample_rows.iloc[0]
    sd  = row[SENSOR_COLS].to_dict()
    ai  = get_anomaly_output(sd)
    th  = get_threshold_alerts(sd, motor_id=row['Motor_ID'], timestamp=str(row['Timestamp']))
    print(f"  [{label:<8}] AI={ai['condition_label']:<8} (fault={ai['fault_type']:<14}) "
          f"| Threshold={th['condition_label']:<8} (violations={th['total_violations']})")

# ============================================================
# 5. RUN INFERENCE ON RAW SENSOR DATA (Vectorized — Fast)
# ============================================================
print("\n[STEP 5] Running inference on raw sensor data (vectorized)...")

raw_feat = raw.copy()

X_raw_sc = scaler_loaded.transform(raw_feat[FEATURE_COLS])
raw_feat['Condition_Label']  = clf_loaded.predict(X_raw_sc)
raw_feat['Fault_Type']       = fault_loaded.predict(X_raw_sc)
raw_feat['IF_Anomaly_Score'] = iso_loaded.score_samples(X_raw_sc)
raw_feat['IF_Is_Anomaly']    = iso_loaded.predict(X_raw_sc) == -1

# Vectorized threshold violation counts (for Threshold_Label / condition alert)
raw_feat['viol_temp']    = (raw_feat['Temperature']   > THRESHOLDS['Temperature']['warning']).astype(int)
raw_feat['viol_vx']      = (raw_feat['Vibration_X']   > THRESHOLDS['Vibration_X']['warning']).astype(int)
raw_feat['viol_vy']      = (raw_feat['Vibration_Y']   > THRESHOLDS['Vibration_Y']['warning']).astype(int)
raw_feat['viol_vz']      = (raw_feat['Vibration_Z']   > THRESHOLDS['Vibration_Z']['warning']).astype(int)
raw_feat['viol_v1h']     = (raw_feat['Voltage_L1']    > THRESHOLDS['Voltage_L1_high']['warning']).astype(int)
raw_feat['viol_v2h']     = (raw_feat['Voltage_L2']    > THRESHOLDS['Voltage_L2_high']['warning']).astype(int)
raw_feat['viol_v3h']     = (raw_feat['Voltage_L3']    > THRESHOLDS['Voltage_L3_high']['warning']).astype(int)
raw_feat['viol_c1h']     = (raw_feat['Current_L1']    > THRESHOLDS['Current_L1_high']['warning']).astype(int)
raw_feat['viol_c2h']     = (raw_feat['Current_L2']    > THRESHOLDS['Current_L2_high']['warning']).astype(int)
raw_feat['viol_c3h']     = (raw_feat['Current_L3']    > THRESHOLDS['Current_L3_high']['warning']).astype(int)
raw_feat['viol_v1l']     = (raw_feat['Voltage_L1']    < THRESHOLDS['Voltage_L1_low']['warning']).astype(int)
raw_feat['viol_v2l']     = (raw_feat['Voltage_L2']    < THRESHOLDS['Voltage_L2_low']['warning']).astype(int)
raw_feat['viol_v3l']     = (raw_feat['Voltage_L3']    < THRESHOLDS['Voltage_L3_low']['warning']).astype(int)
raw_feat['viol_freq_h']  = (raw_feat['Frequency']     > THRESHOLDS['Frequency_high']['warning']).astype(int)
raw_feat['viol_freq_l']  = (raw_feat['Frequency']     < THRESHOLDS['Frequency_low']['warning']).astype(int)
raw_feat['viol_rpm']     = (raw_feat['Rotational_Speed'] < THRESHOLDS['RPM_low']['warning']).astype(int)
raw_feat['viol_vimb']    = (raw_feat['Voltage_Imbalance_Pct'] > THRESHOLDS['Voltage_Imbalance_Pct']['warning']).astype(int)
raw_feat['viol_cimb']    = (raw_feat['Current_Imbalance_Pct'] > THRESHOLDS['Current_Imbalance_Pct']['warning']).astype(int)

viol_cols = [c for c in raw_feat.columns if c.startswith('viol_')]
raw_feat['Total_Violations'] = raw_feat[viol_cols].sum(axis=1)
raw_feat.drop(columns=viol_cols, inplace=True)

# Probable cause & action from client-approved lookup (vectorized via map)
raw_feat['Probable_Cause']     = raw_feat['Fault_Type'].map(
    lambda f: cause_map[f]['Probable_Cause'] if f in cause_map else '-')
raw_feat['Recommended_Action'] = raw_feat['Fault_Type'].map(
    lambda f: cause_map[f]['Recommended_Action'] if f in cause_map else '-')
# Normal rows shouldn't carry a fault cause even if fault_clf mispredicts
raw_feat.loc[raw_feat['Condition_Label'] == 'Normal', 'Probable_Cause'] = '-'
raw_feat.loc[raw_feat['Condition_Label'] == 'Normal', 'Recommended_Action'] = \
    cause_map['Normal']['Recommended_Action']


def alert_msg(row):
    m = {
        'Normal'  : 'Motor operating within normal parameters.',
        'Warning' : f"Warning: {int(row['Total_Violations'])} parameter(s) above normal range. Monitor closely.",
        'Critical': 'Critical: parameters in danger zone. Prepare maintenance action.',
        'Failure' : 'FAILURE ALERT: Motor in critical failure zone. Stop operation and inspect immediately!',
    }
    return m.get(row['Condition_Label'], '')

raw_feat['Alert_Message'] = raw_feat.apply(alert_msg, axis=1)

output_cols = (['Timestamp','Motor_ID'] + SENSOR_COLS +
               ['Condition_Label','Fault_Type','IF_Anomaly_Score','IF_Is_Anomaly',
                'Total_Violations','Probable_Cause','Recommended_Action','Alert_Message'])
results_df = raw_feat[output_cols].copy()
results_path = os.path.join(OUTPUT_DIR, 'raw_data_inference_results.csv')
results_df.to_csv(results_path, index=False)

print(f"  Processed {len(results_df):,} rows from raw sensor data")
print(f"  Predicted condition distribution:")
for lbl, cnt in results_df['Condition_Label'].value_counts().items():
    pct = cnt / len(results_df) * 100
    bar = '#' * int(pct / 3)
    print(f"    {lbl:<10}: {cnt:>7,} ({pct:.1f}%)  {bar}")
print(f"  Saved -> outputs/raw_data_inference_results.csv")

# ============================================================
# 6. INFERENCE LATENCY BENCHMARK (both alert paths)
# ============================================================
print("\n[STEP 6] Inference Latency Benchmark...")

sample = raw.iloc[0][SENSOR_COLS].to_dict()

lat_ai, lat_th = [], []
for _ in range(1000):
    t0 = time.perf_counter(); get_anomaly_output(sample);    lat_ai.append((time.perf_counter()-t0)*1000)
    t0 = time.perf_counter(); get_threshold_alerts(sample);  lat_th.append((time.perf_counter()-t0)*1000)

bench = {'mean': round(np.mean(lat_ai), 2), 'p95': round(np.percentile(lat_ai, 95), 2),
         'p99': round(np.percentile(lat_ai, 99), 2)}
bench_th = {'mean': round(np.mean(lat_th), 2), 'p95': round(np.percentile(lat_th, 95), 2),
            'p99': round(np.percentile(lat_th, 99), 2)}
result_str    = "PASS" if bench['p99'] < 100 else "FAIL"
result_str_th = "PASS" if bench_th['p99'] < 100 else "FAIL"

print(f"  get_anomaly_output()   (AI alert)       : Mean={bench['mean']}ms P95={bench['p95']}ms "
      f"P99={bench['p99']}ms -> {result_str}")
print(f"  get_threshold_alerts() (Condition alert): Mean={bench_th['mean']}ms P95={bench_th['p95']}ms "
      f"P99={bench_th['p99']}ms -> {result_str_th}")

plt.figure(figsize=(9, 4.5))
plt.hist(lat_ai, bins=50, alpha=0.6, label='AI Predictive Alert', color='steelblue')
plt.hist(lat_th, bins=50, alpha=0.6, label='Condition-Based Alert', color='salmon')
plt.axvline(bench['p99'], color='steelblue', linestyle='--', label=f"AI P99={bench['p99']}ms")
plt.axvline(bench_th['p99'], color='salmon', linestyle='--', label=f"Threshold P99={bench_th['p99']}ms")
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.title(f'Inference Latency Distribution (1000 runs each)', fontweight='bold')
plt.legend(fontsize=8)
plt.tight_layout()
save_fig("09_inference_latency.png")

# ============================================================
# 7. FINAL SUMMARY
# ============================================================
print()
print("=" * 60)
print("FINAL SUMMARY — SALSABILA HIDAYAT")
print("=" * 60)

print(f"\nDataset         : client_training_dataset.csv (Nadine, updated w/ Current)")
print(f"Records         : {len(train):,} rows, {len(FEATURE_COLS)} features")

print(f"\nASTRA-11 — Threshold Rules:")
print(f"  4-class threshold rules set and validated")
print(f"  Agreement with Motor_State  : {agree/len(train)*100:.1f}%")
print(f"  Within +-1 severity tier    : {(sev_diff <= 1).mean()*100:.1f}%")

print(f"\nASTRA-10 — Model Comparison:")
print(f"  {'Model':<25} {'Accuracy':>9} {'F1 Macro':>9} {'FAlarms':>9}")
print(f"  {'-'*56}")
for m, r in comparison.items():
    acc = r.get('accuracy', None)
    acc_str = f"{acc:.4f}" if acc else "   N/A  "
    print(f"  {m:<25} {acc_str:>9} {r['f1_macro']:>9.4f} {r['false_alarms']:>9,}")
print(f"  Best model selected         : {best_name}")
print(f"  Fault_Type classifier       : Acc={fault_acc:.4f} Prec={fault_prec:.4f} "
      f"Rec={fault_rec:.4f} F1={fault_f1m:.4f} (macro, leakage-safe split)")

print(f"\nASTRA-12 — Validation:")
print(f"  Confusion matrix, per-class F1, feature importance,")
print(f"  learning curve — all saved to outputs/")

print(f"\nInference Latency:")
print(f"  AI Predictive Alert      : P99 = {bench['p99']} ms -> {result_str}")
print(f"  Condition-Based Alert    : P99 = {bench_th['p99']} ms -> {result_str_th}")

print(f"\nTwo independent alert sources now available (feedback poin #1):")
print(f"  get_anomaly_output()    -> source='ai_model'      -> /api/status")
print(f"  get_threshold_alerts()  -> source='threshold_rule' -> /api/threshold-alerts (NEW)")

print(f"\nAll outputs saved to: outputs/")
print(f"  Model artifacts:")
print(f"    condition_classifier.joblib   ({best_name})")
print(f"    fault_type_classifier.joblib")
print(f"    isolation_forest.joblib")
print(f"    condition_scaler.joblib")
print(f"    cause_action_map.joblib / .json")
print(f"    thresholds.json")
print(f"  Visualizations: 01..09 (see outputs/)")
print(f"  Dataset: raw_data_inference_results.csv  <- labeled raw data for Shania")

print()
print("ASTRA-10 : Train Anomaly Detection Model          — DONE")
print("ASTRA-11 : Set Threshold Rules for Anomaly Alerts — DONE")
print("ASTRA-12 : Test and Validate Anomaly Detection    — DONE")
print("NEW      : get_threshold_alerts() for /api/threshold-alerts — DONE")
