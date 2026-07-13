"""
feature_utils.py
------------------
Shared feature-engineering logic used by BOTH the offline training pipeline
(01_prepare_features.py) and the live inference/forecast path (forecast_engine.py,
api.py). Keeping this in one place means a live prediction always uses exactly
the same feature definitions the models were trained on.
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [
    'Voltage_L1', 'Voltage_L2', 'Voltage_L3',
    'Frequency', 'Power_Factor', 'Temperature',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Rotational_Speed',
]
ENGINEERED_BASE = ['Voltage_Imbalance', 'Vibration_Total', 'Voltage_Mean', 'RPM_Deviation']
FEATURE_COLS = SENSOR_COLS + ENGINEERED_BASE  # the 14 features teammate specified

SHORT_WINDOW = 4     # 1h at 15-min sampling
LONG_WINDOW = 96      # 24h at 15-min sampling


def add_teammate_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """The 4 engineered features from the handoff spec."""
    df = df.copy()
    df['Voltage_Imbalance'] = df[['Voltage_L1', 'Voltage_L2', 'Voltage_L3']].std(axis=1)
    df['Vibration_Total'] = np.sqrt(df['Vibration_X']**2 + df['Vibration_Y']**2 + df['Vibration_Z']**2)
    df['Voltage_Mean'] = df[['Voltage_L1', 'Voltage_L2', 'Voltage_L3']].mean(axis=1)
    df['RPM_Deviation'] = abs(df['Rotational_Speed'] - 1500)
    return df


def add_scaled_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Applies the shared condition_scaler.joblib to the 14 base features."""
    df = df.copy()
    scaled = scaler.transform(df[FEATURE_COLS])
    for i, col in enumerate(FEATURE_COLS):
        df[f'{col}_scaled'] = scaled[:, i]
    return df


def _slope(x):
    if len(x) < 2:
        return 0.0
    idx = np.arange(len(x))
    return np.polyfit(idx, x, 1)[0]


def add_temporal_features_single_motor(g: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean/std (1h) and trend slope (24h) for each scaled feature.
    Operates on ONE motor's data already sorted by time. Use this (not the
    grouped version) in live inference, where you're only ever scoring one
    motor's stream at a time.
    """
    g = g.copy()
    cols_to_track = [f'{c}_scaled' for c in FEATURE_COLS]
    for col in cols_to_track:
        g[f'{col}_roll_mean_1h'] = g[col].rolling(SHORT_WINDOW, min_periods=1).mean()
        g[f'{col}_roll_std_1h'] = g[col].rolling(SHORT_WINDOW, min_periods=1).std().fillna(0)
        g[f'{col}_trend_24h'] = g[col].rolling(LONG_WINDOW, min_periods=4).apply(_slope, raw=True).fillna(0)
    return g


def add_temporal_features_grouped(df: pd.DataFrame) -> pd.DataFrame:
    """Same as above but grouped by Motor_ID -- used for the offline training set."""
    df = df.sort_values(['Motor_ID', 'Timestamp']).reset_index(drop=True)
    out = [add_temporal_features_single_motor(g) for _, g in df.groupby('Motor_ID')]
    return pd.concat(out, ignore_index=True)


def get_all_feature_columns():
    cols = []
    for c in FEATURE_COLS:
        sc = f"{c}_scaled"
        cols += [sc, f"{sc}_roll_mean_1h", f"{sc}_roll_std_1h", f"{sc}_trend_24h"]
    return cols


def build_features_for_motor(raw_df: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Full pipeline for ONE motor's raw sensor readings (sorted by Timestamp):
    engineered features -> scaling -> rolling/trend. This is the single
    function live inference and forecasting both call, so there's exactly
    one definition of "how do we turn raw sensors into model features."
    """
    df = add_teammate_engineered_features(raw_df)
    df = add_scaled_features(df, scaler)
    df = add_temporal_features_single_motor(df)
    return df
