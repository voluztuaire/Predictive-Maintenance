"""
astra_config.py — Shared constants & pure functions
=====================================================
Extracted from astra_anomaly_detection.py (single source of truth for
values — keep both in sync if thresholds/features change there).

WHY THIS FILE EXISTS:
astra_anomaly_detection.py is a top-to-bottom script (loads CSVs,
trains models, saves plots, ~1-2 min runtime side effects). Importing
it directly from expert_validation.py / retrain_pipeline.py would
re-run all of that on every import. This module holds only the pure,
side-effect-free pieces (constants + functions with no I/O) so other
backend modules can reuse them cheaply and consistently.

No behavior change vs astra_anomaly_detection.py — this is a copy of
the same definitions, not a rewrite.
"""
import numpy as np
import pandas as pd

# ---- Feature / label config ----
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
FAULT_ORDER   = ['Normal', 'Rotor Bar', 'Bearing Wear', 'Misalignment', 'Stator Winding', 'Other']


# ---- ASTRA-11 Threshold Rules ----
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
}


# ---- Pure helper functions ----
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


# ---- Feature engineering ----
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

FEATURE_COLS = SENSOR_COLS + [
    'Voltage_Imbalance_Pct',
    'Current_Imbalance_Pct',
    'Vibration_Total',
    'Voltage_Mean',
    'Current_Mean',
    'RPM_Deviation',
]
