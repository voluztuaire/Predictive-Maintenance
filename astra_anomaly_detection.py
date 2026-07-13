# ============================================================
# ANOMALY DETECTION MODULE
# Role    : ML Engineer — Anomaly Detection
# Name    : Salsabila Hidayat
# Project : Predictive Maintenance — Astra Otoparts WINTEQ
# Team    : Group 3 AI Bootcamp
# Tasks   : ASTRA-10, ASTRA-11, ASTRA-12
# ============================================================

import sklearn
assert sklearn.__version__ == "1.4.0", (
    f"scikit-learn version mismatch: found {sklearn.__version__}, "
    f"need 1.4.0 to match the rest of the team. Run: pip install scikit-learn==1.4.0"
)

import os
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    accuracy_score, ConfusionMatrixDisplay
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
    print(f"  Saved → outputs/{filename}")

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

# Sensor features (approved by Pak Wahyu & Pak Andri)
SENSOR_COLS = [
    'Voltage_L1', 'Voltage_L2', 'Voltage_L3',
    'Frequency',  'Power_Factor',
    'Temperature',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Rotational_Speed',
]
LABEL_COL     = 'Motor_State'
LABEL_ORDER   = ['Normal', 'Warning', 'Critical', 'Failure']
FAULT_COL     = 'Fault_Type_True'

print(f"\n  Label distribution:")
for lbl, cnt in train[LABEL_COL].value_counts().items():
    pct = cnt / len(train) * 100
    bar = '█' * int(pct / 3)
    print(f"    {lbl:<10}: {cnt:>7,}  ({pct:5.1f}%)  {bar}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
print("\n[STEP 2] Exploratory Data Analysis...")

fault_order  = ['Normal', 'Rotor Bar', 'Bearing Wear', 'Misalignment', 'Stator Winding']
fault_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']

print("  Mean sensor values per Fault Type:")
print(train.groupby(FAULT_COL)[SENSOR_COLS].mean().round(3).T.to_string())

# Plot: boxplot 6 key sensors per fault type
key_sensors = ['Temperature', 'Vibration_X', 'Vibration_Y',
               'Vibration_Z', 'Voltage_L1', 'Rotational_Speed']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, sensor in zip(axes.flatten(), key_sensors):
    data = [train[train[FAULT_COL] == f][sensor].dropna().values
            for f in fault_order]
    bp   = ax.boxplot(data, patch_artist=True, labels=fault_order)
    for patch, color in zip(bp['boxes'], fault_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(sensor, fontweight='bold')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=20)

plt.suptitle('EDA — Sensor Distribution per Fault Type',
             fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig("01_eda_sensor_per_fault.png")
print("  EDA plot saved.")

# ============================================================
# ASTRA-11: SET THRESHOLD RULES FOR ANOMALY ALERTS
# ============================================================
# WHY THRESHOLD-BASED LABELING?
# Dataset from Nadine has Motor_State (4 class labels).
# This step VALIDATES those labels by applying our own
# physics-based threshold rules derived from:
#   - P95 of Normal data  → Warning boundary
#   - P95 of Warning data → Critical boundary
#   - P95 of Critical data→ Failure boundary
#   - Industry standards for 3-phase 400V / 50Hz motors
#
# WHY NOT JUST USE Motor_State DIRECTLY?
# We need to ensure our threshold logic is sound and
# independently reproducible for real-time inference on
# raw sensor data (raw_sensor_new_data.csv) that has NO labels.
# ============================================================
print("\n[ASTRA-11] Setting Threshold Rules for Anomaly Alerts...")

# Threshold table (derived from data analysis above)
# Format: {'warning': T1, 'critical': T2, 'failure': T3}
# T1 = boundary between Normal and Warning
# T2 = boundary between Warning and Critical
# T3 = boundary between Critical and Failure
THRESHOLDS = {
    # TEMPERATURE (°C)
    # Normal P95=50.46 | Warning mean=48.4 | Critical mean=51.4 | Failure mean=57.1
    'Temperature': {
        'warning' : 51.0,
        'critical': 57.0,
        'failure' : 66.0,
    },
    # VIBRATION X (mm/s) — radial axis
    # Normal P95=1.69 | Warning P95=3.51 | Critical P95=5.15 | Failure P95=8.36
    'Vibration_X': {
        'warning' : 1.80,
        'critical': 3.60,
        'failure' : 5.50,
    },
    # VIBRATION Y (mm/s) — radial axis
    'Vibration_Y': {
        'warning' : 1.80,
        'critical': 3.60,
        'failure' : 5.50,
    },
    # VIBRATION Z (mm/s) — axial axis (typically higher than X/Y)
    'Vibration_Z': {
        'warning' : 1.90,
        'critical': 3.90,
        'failure' : 5.80,
    },
    # VOLTAGE per phase — upper bound (nominal 400V, ±2% = 392-408V)
    'Voltage_L1_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L2_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    'Voltage_L3_high': {'warning': 403.0, 'critical': 405.0, 'failure': 407.0},
    # VOLTAGE per phase — lower bound
    'Voltage_L1_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L2_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    'Voltage_L3_low':  {'warning': 397.0, 'critical': 395.0, 'failure': 393.0},
    # FREQUENCY (Hz) — nominal 50Hz, ±0.1Hz
    'Frequency_high': {'warning': 50.10, 'critical': 50.15, 'failure': 50.20},
    'Frequency_low':  {'warning': 49.90, 'critical': 49.85, 'failure': 49.80},
    # ROTATIONAL SPEED (RPM) — lower bound only (too slow = high slip)
    # Normal mean=1463, Warning/Critical/Failure means ~1461-1463
    'RPM_low': {'warning': 1455.0, 'critical': 1450.0, 'failure': 1445.0},
}

# Print threshold table
print("\n  Threshold rules:")
print(f"  {'Parameter':<22} {'Warning':>10} {'Critical':>10} {'Failure':>10}  Direction")
print("  " + "-" * 65)
for param, tiers in THRESHOLDS.items():
    direction = 'LOWER BOUND' if param.endswith('_low') else 'UPPER BOUND'
    print(f"  {param:<22} {tiers['warning']:>10} {tiers['critical']:>10} "
          f"{tiers['failure']:>10}  {direction}")


def check_violations(row: dict) -> dict:
    """
    Check which sensor parameters violate which threshold tier.

    WHY: Instead of flagging an alert on a single parameter breach,
    we count violations across ALL parameters. This reduces false
    alarms — a single vibration spike does not immediately mean
    failure if all other sensors are normal.

    Returns dict with lists of violated parameter names per tier.
    """
    v = {'warning': [], 'critical': [], 'failure': []}

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
    upper('Frequency',      'Frequency_high')
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
      Critical : 2+ parameters in critical zone
      Warning  : 1+ parameter in warning/critical zone
      Normal   : no violations at all
    """
    viol = check_violations(row)
    n_f  = len(viol['failure'])
    n_c  = len(viol['critical'])
    n_w  = len(viol['warning'])
    n_t  = n_f + n_c + n_w

    if n_f >= 1 or n_t >= 4: return 'Failure'
    elif n_c >= 2:            return 'Critical'
    elif n_c >= 1 or n_w >= 1: return 'Warning'
    else:                     return 'Normal'


# Generate our threshold-based labels
print("\n  Generating threshold-based labels...")
train['Threshold_Label'] = train[SENSOR_COLS].apply(
    lambda r: assign_threshold_label(r.to_dict()), axis=1
)

# Compare our labels vs Nadine's Motor_State
print("\n  Validation — Threshold Labels vs Motor_State (Nadine):")
cross = pd.crosstab(train[LABEL_COL], train['Threshold_Label'],
                    margins=True)
print(cross.to_string())

# Agreement rate
agree = (train[LABEL_COL] == train['Threshold_Label']).sum()
print(f"\n  Agreement rate: {agree:,} / {len(train):,} "
      f"({agree/len(train)*100:.1f}%)")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Nadine's labels
counts_nadine = [train[LABEL_COL].value_counts().get(l, 0) for l in LABEL_ORDER]
colors_cond   = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
axes[0].bar(LABEL_ORDER, counts_nadine, color=colors_cond)
axes[0].set_title("Motor_State Distribution\n(Nadine's Labels)", fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts_nadine):
    axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

# Our threshold labels
counts_thresh = [train['Threshold_Label'].value_counts().get(l, 0) for l in LABEL_ORDER]
axes[1].bar(LABEL_ORDER, counts_thresh, color=colors_cond)
axes[1].set_title("Threshold_Label Distribution\n(Our Rules — ASTRA-11)", fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(counts_thresh):
    axes[1].text(i, v + 500, f'{v:,}', ha='center', fontsize=9)

plt.suptitle('ASTRA-11: Label Comparison — Nadine vs Our Threshold Rules',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("02_astra11_threshold_label_comparison.png")

# Sensor distribution heatmap per condition
sensor_means = train.groupby(LABEL_COL)[SENSOR_COLS].mean()
sensor_norm  = (sensor_means - sensor_means.min()) / \
               (sensor_means.max() - sensor_means.min())
sensor_norm  = sensor_norm.reindex(LABEL_ORDER)

plt.figure(figsize=(14, 5))
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
# WHY FEATURE ENGINEERING?
# Raw sensor values capture point-in-time readings.
# Derived features capture RELATIONSHIPS between sensors
# that are more informative for fault detection:
#
#   Voltage_Imbalance → detects phase imbalance (stator winding fault signature)
#   Vibration_Total   → overall vibration severity regardless of axis
#   Voltage_Mean      → average supply level across 3 phases
#   RPM_Deviation     → how far from nominal speed (indicates motor slip/load issues)
# ============================================================
print("\n[STEP 3] Feature Engineering...")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Voltage_Imbalance'] = df[['Voltage_L1','Voltage_L2','Voltage_L3']].std(axis=1)
    df['Vibration_Total']   = np.sqrt(df['Vibration_X']**2 +
                                       df['Vibration_Y']**2 +
                                       df['Vibration_Z']**2)
    df['Voltage_Mean']      = df[['Voltage_L1','Voltage_L2','Voltage_L3']].mean(axis=1)
    df['RPM_Deviation']     = abs(df['Rotational_Speed'] - 1500)
    return df

train = add_engineered_features(train)
raw   = add_engineered_features(raw)

FEATURE_COLS = SENSOR_COLS + [
    'Voltage_Imbalance',
    'Vibration_Total',
    'Voltage_Mean',
    'RPM_Deviation',
]

print(f"  Original sensor features   : {len(SENSOR_COLS)}")
print(f"  Engineered features added  : 4")
print(f"    Voltage_Imbalance = std(L1, L2, L3)")
print(f"    Vibration_Total   = sqrt(X² + Y² + Z²)")
print(f"    Voltage_Mean      = mean(L1, L2, L3)")
print(f"    RPM_Deviation     = |RPM - 1500|")
print(f"  Total features             : {len(FEATURE_COLS)}")

# ============================================================
# ASTRA-10: TRAIN ANOMALY DETECTION MODEL
# ============================================================
# WHY THESE 3 MODELS?
#
# IsolationForest (unsupervised):
#   Chosen because it does not need labels to detect anomalies.
#   It learns what "normal" looks like and flags anything unusual.
#   Useful as a first-pass detector and for cases where labels
#   might be noisy or unavailable.
#
# RandomForestClassifier (supervised):
#   Chosen because we DO have 4-class labels, making supervised
#   learning possible. Random Forest handles non-linear boundaries,
#   works well with mixed-scale features (no need for careful
#   tuning), is robust to outliers, and provides feature importance.
#
# LogisticRegression (supervised — baseline):
#   Chosen as a linear baseline. If logistic regression performs
#   close to Random Forest, the problem is linearly separable
#   and we don't need the complexity of Random Forest.
#   If RF is much better, non-linear patterns matter.
#
# WHY StandardScaler?
#   Features have different units (°C, mm/s, V, Hz, RPM).
#   StandardScaler normalizes all to z-scores: z = (x-mean)/std
#   so no single feature dominates because of its unit magnitude.
#   Fit on TRAINING SET ONLY to prevent data leakage.
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

# Print comparison table
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

# Select best supervised model
best_name = max(['RandomForest', 'LogisticRegression'],
                key=lambda m: comparison[m]['f1_macro'])
best_clf  = rf if best_name == 'RandomForest' else lr
print(f"\n  Best model (highest F1 Macro): {best_name}")

# Save all artifacts
joblib.dump(best_clf, os.path.join(OUTPUT_DIR, 'condition_classifier.joblib'))
joblib.dump(iso,      os.path.join(OUTPUT_DIR, 'isolation_forest.joblib'))
joblib.dump(scaler,   os.path.join(OUTPUT_DIR, 'condition_scaler.joblib'))
print(f"\n  Artifacts saved to outputs/:")
print(f"    condition_classifier.joblib  ({best_name})")
print(f"    isolation_forest.joblib")
print(f"    condition_scaler.joblib")

# ============================================================
# ASTRA-12: TEST AND VALIDATE ANOMALY DETECTION RESULTS
# ============================================================
print("\n[ASTRA-12] Validating Anomaly Detection Results...")

best_pred = rf_pred if best_name == 'RandomForest' else lr_pred

# Classification report
print(f"\n  Classification Report — {best_name}:")
print(classification_report(y_test, best_pred,
                             target_names=LABEL_ORDER, zero_division=0))

# --- Plot 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_best = confusion_matrix(y_test, best_pred, labels=LABEL_ORDER)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues',
            xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER, ax=axes[0])
axes[0].set_title(f'Confusion Matrix\n{best_name} (4-Class)',
                  fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_iso = confusion_matrix(y_test_bin, iso_bin)
sns.heatmap(cm_iso, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal','Anomaly'],
            yticklabels=['Normal','Anomaly'], ax=axes[1])
axes[1].set_title('Confusion Matrix\nIsolationForest (Binary)',
                  fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('ASTRA-12: Confusion Matrices', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig("04_astra12_confusion_matrices.png")

# --- Plot 2: Model Comparison Bar Chart ---
models  = list(comparison.keys())
f1_vals = [comparison[m]['f1_macro'] for m in models]
fa_vals = [comparison[m]['false_alarms'] for m in models]
x       = np.arange(len(models))
w       = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(x - w/2, f1_vals, w, label='F1 Macro Score',
               color='steelblue')
ax1.set_ylabel('F1 Macro Score', color='steelblue', fontsize=11)
ax1.set_ylim(0, 1.1)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
for bar, val in zip(bars, f1_vals):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01, f'{val:.4f}',
             ha='center', fontsize=9, color='steelblue')

ax2 = ax1.twinx()
bars2 = ax2.bar(x + w/2, fa_vals, w, label='False Alarms',
                color='salmon', alpha=0.8)
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

# --- Plot 3: Feature Importance ---
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

    plt.suptitle('ASTRA-12: Feature Importance Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig("06_astra12_feature_importance.png")

# --- Plot 4: Learning Curve ---
# WHY LEARNING CURVE?
# Shows whether our model is overfitting (high train, low val)
# or underfitting (both low), and whether more data would help.
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
plt.fill_between(train_sizes, train_mean-train_std,
                 train_mean+train_std, alpha=0.2, color='steelblue')
plt.plot(train_sizes, val_mean, 'o-', color='darkorange', label='Validation Score (CV)')
plt.fill_between(train_sizes, val_mean-val_std,
                 val_mean+val_std, alpha=0.2, color='darkorange')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Macro Score')
plt.title('ASTRA-12: Learning Curve\n(Training vs Cross-Validation Score)',
          fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
save_fig("07_astra12_learning_curve.png")

# --- Plot 5: Per-Class F1 Score ---
report_dict = classification_report(y_test, best_pred,
                                    target_names=LABEL_ORDER,
                                    zero_division=0, output_dict=True)
per_class_f1 = {lbl: report_dict[lbl]['f1-score'] for lbl in LABEL_ORDER}

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(LABEL_ORDER, list(per_class_f1.values()), color=colors_cond)
ax.set_title(f'ASTRA-12: Per-Class F1 Score — {best_name}', fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, per_class_f1.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01, f'{val:.4f}',
            ha='center', fontsize=10)
plt.tight_layout()
save_fig("08_astra12_per_class_f1.png")

print("  ASTRA-12 validation complete.")

# ============================================================
# 4. INFERENCE FUNCTION — FOR EL SHADDAI'S DASHBOARD
# ============================================================
# This function is called once per sensor reading in real-time.
# El Shaddai calls this from Flask each time /api/status is hit.
# Shania calls this to get condition_label for her RUL/Health model.
# ============================================================

# Reload saved artifacts (simulates production usage)
clf_loaded    = joblib.load(os.path.join(OUTPUT_DIR, 'condition_classifier.joblib'))
iso_loaded    = joblib.load(os.path.join(OUTPUT_DIR, 'isolation_forest.joblib'))
scaler_loaded = joblib.load(os.path.join(OUTPUT_DIR, 'condition_scaler.joblib'))

# Fault-based probable cause & action rules
FAULT_RULES = [
    {
        'condition': lambda e: e['Voltage_Imbalance'] > 3.0 or
                               (e['Temperature'] > 57 and e['Vibration_Total'] < 4),
        'cause' : 'Winding insulation degradation due to repeated overheating or voltage phase imbalance.',
        'action': 'Perform insulation resistance test (megger) and check the motor cooling system.',
    },
    {
        'condition': lambda e: e.get('Vibration_Z', 0) > 5.0 and
                               e['Vibration_Total'] > 7.0,
        'cause' : 'Motor shaft misalignment against load or coupling.',
        'action': 'Perform laser alignment check and correct the motor mounting position.',
    },
    {
        'condition': lambda e: e['Vibration_Total'] > 6.0 and
                               e['RPM_Deviation'] > 40,
        'cause' : 'Insufficient lubrication or bearing wear due to excessive load.',
        'action': 'Re-lubricate or replace the bearing, check seal condition and shaft alignment.',
    },
    {
        'condition': lambda e: e['Vibration_Total'] > 3.5 and
                               e['RPM_Deviation'] > 30,
        'cause' : 'Broken or cracked rotor bar due to repeated thermal stress or high starting frequency.',
        'action': 'Perform motor current signature analysis (MCSA) and schedule rotor inspection/teardown.',
    },
]


def get_anomaly_output(sensor_data: dict) -> dict:
    """
    Full inference pipeline for one sensor reading.

    Called by:
      - El Shaddai: from Flask /api/status endpoint (real-time dashboard)
      - Shania: to get condition_label for RUL/Health Score model input

    Args:
        sensor_data (dict): one row of sensor readings.
            Required keys: Voltage_L1, Voltage_L2, Voltage_L3,
                           Frequency, Power_Factor, Temperature,
                           Vibration_X, Vibration_Y, Vibration_Z,
                           Rotational_Speed

    Returns:
        dict with all fields needed by the dashboard:
            condition_label     : Normal / Warning / Critical / Failure
            status_color        : green / yellow / orange / red
            if_anomaly_score    : IsolationForest raw score (float)
            if_is_anomaly       : bool
            violated_warning    : list of parameter names
            violated_critical   : list of parameter names
            violated_failure    : list of parameter names
            total_violations    : int
            probable_cause      : string (English)
            recommended_action  : string (English)
            alert_message       : string (English)
    """
    # 1. Add engineered features
    e = dict(sensor_data)
    e['Voltage_Imbalance'] = float(np.std([
        e.get('Voltage_L1', 400),
        e.get('Voltage_L2', 400),
        e.get('Voltage_L3', 400),
    ]))
    e['Vibration_Total']  = float(np.sqrt(
        e.get('Vibration_X', 0)**2 +
        e.get('Vibration_Y', 0)**2 +
        e.get('Vibration_Z', 0)**2
    ))
    e['Voltage_Mean']     = float(np.mean([
        e.get('Voltage_L1', 400),
        e.get('Voltage_L2', 400),
        e.get('Voltage_L3', 400),
    ]))
    e['RPM_Deviation']    = float(abs(e.get('Rotational_Speed', 1500) - 1500))

    # 2. Scale and predict
    arr    = np.array([[e.get(f, 0) for f in FEATURE_COLS]])
    arr_sc = scaler_loaded.transform(arr)

    condition   = clf_loaded.predict(arr_sc)[0]
    if_score    = float(iso_loaded.score_samples(arr_sc)[0])
    if_is_anom  = bool(iso_loaded.predict(arr_sc)[0] == -1)

    # 3. Threshold violations
    viol = check_violations(e)
    n_t  = len(viol['warning']) + len(viol['critical']) + len(viol['failure'])

    # 4. Determine probable cause & recommended action
    cause  = 'Normal condition, continue routine monitoring.'
    action = 'Normal condition, continue routine monitoring.'
    if condition != 'Normal':
        for rule in FAULT_RULES:
            try:
                if rule['condition'](e):
                    cause  = rule['cause']
                    action = rule['action']
                    break
            except Exception:
                continue
        if cause == 'Normal condition, continue routine monitoring.':
            cause  = 'Sensor parameters exceeded normal range — further investigation required.'
            action = 'Perform visual inspection and direct measurement on the motor.'

    # 5. Alert message and color
    color_map = {
        'Normal': 'green', 'Warning': 'yellow',
        'Critical': 'orange', 'Failure': 'red',
    }
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
        'violated_warning'   : viol['warning'],
        'violated_critical'  : viol['critical'],
        'violated_failure'   : viol['failure'],
        'total_violations'   : n_t,
        'probable_cause'     : cause,
        'recommended_action' : action,
        'alert_message'      : msg_map[condition],
    }


# ============================================================
# 5. RUN INFERENCE ON RAW SENSOR DATA (Vectorized — Fast)
# ============================================================
# WHY VECTORIZED?
# Row-by-row inference on 189k rows is very slow in Python.
# We batch all preprocessing and model prediction at once
# using NumPy/sklearn's native vectorized operations,
# then apply the fault rule logic only to non-Normal rows
# (much smaller subset), keeping total runtime under 30s.
# ============================================================
print("\n[STEP 5] Running inference on raw sensor data (vectorized)...")

# --- Step 5a: Engineered features (already done above) ---
raw_feat = raw.copy()
raw_feat['Voltage_Imbalance'] = raw_feat[['Voltage_L1','Voltage_L2','Voltage_L3']].std(axis=1)
raw_feat['Vibration_Total']   = np.sqrt(raw_feat['Vibration_X']**2 +
                                         raw_feat['Vibration_Y']**2 +
                                         raw_feat['Vibration_Z']**2)
raw_feat['Voltage_Mean']      = raw_feat[['Voltage_L1','Voltage_L2','Voltage_L3']].mean(axis=1)
raw_feat['RPM_Deviation']     = abs(raw_feat['Rotational_Speed'] - 1500)

# --- Step 5b: Scale and predict all at once ---
X_raw_sc         = scaler_loaded.transform(raw_feat[FEATURE_COLS])
raw_feat['Condition_Label'] = clf_loaded.predict(X_raw_sc)
raw_feat['IF_Anomaly_Score'] = iso_loaded.score_samples(X_raw_sc)
raw_feat['IF_Is_Anomaly']    = iso_loaded.predict(X_raw_sc) == -1

# --- Step 5c: Vectorized threshold violation counts ---
# Upper bounds
raw_feat['viol_temp']    = (raw_feat['Temperature']   > THRESHOLDS['Temperature']['warning']).astype(int)
raw_feat['viol_vx']      = (raw_feat['Vibration_X']   > THRESHOLDS['Vibration_X']['warning']).astype(int)
raw_feat['viol_vy']      = (raw_feat['Vibration_Y']   > THRESHOLDS['Vibration_Y']['warning']).astype(int)
raw_feat['viol_vz']      = (raw_feat['Vibration_Z']   > THRESHOLDS['Vibration_Z']['warning']).astype(int)
raw_feat['viol_v1h']     = (raw_feat['Voltage_L1']    > THRESHOLDS['Voltage_L1_high']['warning']).astype(int)
raw_feat['viol_v2h']     = (raw_feat['Voltage_L2']    > THRESHOLDS['Voltage_L2_high']['warning']).astype(int)
raw_feat['viol_v3h']     = (raw_feat['Voltage_L3']    > THRESHOLDS['Voltage_L3_high']['warning']).astype(int)
# Lower bounds
raw_feat['viol_v1l']     = (raw_feat['Voltage_L1']    < THRESHOLDS['Voltage_L1_low']['warning']).astype(int)
raw_feat['viol_v2l']     = (raw_feat['Voltage_L2']    < THRESHOLDS['Voltage_L2_low']['warning']).astype(int)
raw_feat['viol_v3l']     = (raw_feat['Voltage_L3']    < THRESHOLDS['Voltage_L3_low']['warning']).astype(int)
raw_feat['viol_freq_h']  = (raw_feat['Frequency']     > THRESHOLDS['Frequency_high']['warning']).astype(int)
raw_feat['viol_freq_l']  = (raw_feat['Frequency']     < THRESHOLDS['Frequency_low']['warning']).astype(int)
raw_feat['viol_rpm']     = (raw_feat['Rotational_Speed'] < THRESHOLDS['RPM_low']['warning']).astype(int)

viol_cols = [c for c in raw_feat.columns if c.startswith('viol_')]
raw_feat['Total_Violations'] = raw_feat[viol_cols].sum(axis=1)
raw_feat.drop(columns=viol_cols, inplace=True)

# --- Step 5d: Vectorized probable cause (rule-based on engineered features) ---
def assign_cause_action(row):
    if row['Condition_Label'] == 'Normal':
        return ('Normal condition, continue routine monitoring.',
                'Normal condition, continue routine monitoring.')
    elif row['Voltage_Imbalance'] > 3.0 or (row['Temperature'] > 57 and row['Vibration_Total'] < 4):
        return ('Winding insulation degradation due to repeated overheating or voltage phase imbalance.',
                'Perform insulation resistance test (megger) and check the motor cooling system.')
    elif row['Vibration_Z'] > 5.0 and row['Vibration_Total'] > 7.0:
        return ('Motor shaft misalignment against load or coupling.',
                'Perform laser alignment check and correct the motor mounting position.')
    elif row['Vibration_Total'] > 6.0 and row['RPM_Deviation'] > 40:
        return ('Insufficient lubrication or bearing wear due to excessive load.',
                'Re-lubricate or replace the bearing, check seal condition and shaft alignment.')
    elif row['Vibration_Total'] > 3.5 and row['RPM_Deviation'] > 30:
        return ('Broken or cracked rotor bar due to repeated thermal stress or high starting frequency.',
                'Perform motor current signature analysis (MCSA) and schedule rotor inspection.')
    else:
        return ('Sensor parameters exceeded normal range — further investigation required.',
                'Perform visual inspection and direct measurement on the motor.')

print("  Assigning probable cause and recommended action...")
cause_action = raw_feat.apply(assign_cause_action, axis=1, result_type='expand')
raw_feat['Probable_Cause']      = cause_action[0]
raw_feat['Recommended_Action']  = cause_action[1]

# --- Step 5e: Alert message ---
def alert_msg(row):
    m = {
        'Normal'  : 'Motor operating within normal parameters.',
        'Warning' : f"Warning: {int(row['Total_Violations'])} parameter(s) above normal range. Monitor closely.",
        'Critical': 'Critical: parameters in danger zone. Prepare maintenance action.',
        'Failure' : 'FAILURE ALERT: Motor in critical failure zone. Stop operation and inspect immediately!',
    }
    return m.get(row['Condition_Label'], '')

raw_feat['Alert_Message'] = raw_feat.apply(alert_msg, axis=1)

# --- Save results ---
output_cols = (['Timestamp','Motor_ID'] + SENSOR_COLS +
               ['Condition_Label','IF_Anomaly_Score','IF_Is_Anomaly',
                'Total_Violations','Probable_Cause','Recommended_Action','Alert_Message'])
results_df = raw_feat[output_cols].copy()
results_path = os.path.join(OUTPUT_DIR, 'raw_data_inference_results.csv')
results_df.to_csv(results_path, index=False)

print(f"  Processed {len(results_df):,} rows from raw sensor data")
print(f"  Predicted condition distribution:")
for lbl, cnt in results_df['Condition_Label'].value_counts().items():
    pct = cnt / len(results_df) * 100
    bar = '█' * int(pct / 3)
    print(f"    {lbl:<10}: {cnt:>7,} ({pct:.1f}%)  {bar}")
print(f"  Saved → outputs/raw_data_inference_results.csv")

# ============================================================
# 6. INFERENCE LATENCY BENCHMARK
# ============================================================
print("\n[STEP 6] Inference Latency Benchmark...")

sample   = raw.iloc[0][SENSOR_COLS].to_dict()
latencies = []
for _ in range(1000):
    t0 = time.perf_counter()
    get_anomaly_output(sample)
    latencies.append((time.perf_counter() - t0) * 1000)

bench = {
    'mean': round(np.mean(latencies), 2),
    'p95' : round(np.percentile(latencies, 95), 2),
    'p99' : round(np.percentile(latencies, 99), 2),
}
result_str = "PASS" if bench['p99'] < 100 else "FAIL"

print(f"  Mean: {bench['mean']} ms | P95: {bench['p95']} ms | "
      f"P99: {bench['p99']} ms")
print(f"  Target P99 < 100ms → {result_str}")

plt.figure(figsize=(8, 4))
plt.hist(latencies, bins=50, color='steelblue', edgecolor='white')
plt.axvline(bench['p99'], color='red', linestyle='--',
            label=f"P99 = {bench['p99']} ms")
plt.xlabel('Latency (ms)')
plt.ylabel('Count')
plt.title(f'Inference Latency Distribution (1000 runs) — {result_str}',
          fontweight='bold')
plt.legend()
plt.tight_layout()
save_fig("09_inference_latency.png")

# ============================================================
# 7. FINAL SUMMARY
# ============================================================
print()
print("=" * 60)
print("FINAL SUMMARY — SALSABILA HIDAYAT")
print("=" * 60)

print(f"\nDataset         : client_training_dataset.csv")
print(f"Records         : {len(train):,} rows, {len(FEATURE_COLS)} features")

print(f"\nASTRA-11 — Threshold Rules:")
print(f"  4-class threshold rules set and validated")
agree = (train[LABEL_COL] == train['Threshold_Label']).sum()
print(f"  Agreement with Nadine's Motor_State: {agree/len(train)*100:.1f}%")

print(f"\nASTRA-10 — Model Comparison:")
print(f"  {'Model':<25} {'Accuracy':>9} {'F1 Macro':>9} {'FAlarms':>9}")
print(f"  {'-'*56}")
for m, r in comparison.items():
    acc = r.get('accuracy', None)
    acc_str = f"{acc:.4f}" if acc else "   N/A  "
    print(f"  {m:<25} {acc_str:>9} {r['f1_macro']:>9.4f} {r['false_alarms']:>9,}")
print(f"  Best model selected: {best_name}")

print(f"\nASTRA-12 — Validation:")
print(f"  Confusion matrix, per-class F1, feature importance,")
print(f"  learning curve — all saved to outputs/")

print(f"\nInference Latency : P99 = {bench['p99']} ms → {result_str}")

print(f"\nAll outputs saved to: outputs/")
print(f"  Model artifacts:")
print(f"    condition_classifier.joblib  ← main model for El Shaddai & Shania")
print(f"    isolation_forest.joblib      ← secondary anomaly detector")
print(f"    condition_scaler.joblib      ← scaler, MUST be used with both models")
print(f"  Visualizations:")
print(f"    01_eda_sensor_per_fault.png")
print(f"    02_astra11_threshold_label_comparison.png")
print(f"    03_heatmap_sensor_per_condition.png")
print(f"    04_astra12_confusion_matrices.png")
print(f"    05_astra12_model_comparison.png")
print(f"    06_astra12_feature_importance.png")
print(f"    07_astra12_learning_curve.png")
print(f"    08_astra12_per_class_f1.png")
print(f"    09_inference_latency.png")
print(f"  Dataset:")
print(f"    raw_data_inference_results.csv  ← labeled raw data for Shania")

print()
print("ASTRA-10 : Train Anomaly Detection Model          — DONE")
print("ASTRA-11 : Set Threshold Rules for Anomaly Alerts — DONE")
print("ASTRA-12 : Test and Validate Anomaly Detection    — DONE")
