"""
forecast_engine.py
--------------------
Forecasts a motor's sensors forward in time, then chains those forecasted
sensor values through the same feature engineering + trained models to
project Health Score and RUL into the future.

Why Holt-Winters (statsmodels) instead of a deep forecasting model: this
data has a clear trend component (degradation) and a clear daily seasonal
cycle (ambient temperature/load following a 24h pattern) -- exactly what
Holt-Winters is designed for. It's fast (fits in milliseconds per sensor),
needs no GPU/training-from-scratch, and gives an interpretable trend +
seasonal decomposition for free. A deep sequence model would need far more
run-to-failure histories than a typical fleet has to outperform this.

Requires at least 48h (192 samples) of history per motor to fit a seasonal
model reliably; falls back to simple linear trend extrapolation if less
history is available.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")  # statsmodels convergence warnings on short/noisy series

SAMPLES_PER_HOUR = 4      # 15-min sampling
SEASONAL_PERIOD = 96      # 24h daily cycle
MIN_SAMPLES_FOR_SEASONAL = 192   # 48h minimum for a seasonal fit to be meaningful

SENSOR_COLS = [
    'Voltage_L1', 'Voltage_L2', 'Voltage_L3',
    'Frequency', 'Power_Factor', 'Temperature',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z',
    'Rotational_Speed',
]


def forecast_single_sensor(series: np.ndarray, horizon: int) -> np.ndarray:
    """Forecasts one sensor `horizon` steps ahead. Falls back gracefully
    for short or degenerate (near-constant) series."""
    n = len(series)

    if np.std(series) < 1e-6:
        # constant series (e.g. Power_Factor sometimes is exactly 1.0) -- just repeat it
        return np.full(horizon, series[-1])

    if n < MIN_SAMPLES_FOR_SEASONAL:
        # not enough history for a seasonal model -- linear trend extrapolation
        idx = np.arange(n)
        slope, intercept = np.polyfit(idx, series, 1)
        future_idx = np.arange(n, n + horizon)
        return slope * future_idx + intercept

    try:
        model = ExponentialSmoothing(
            series, trend="add", seasonal="add",
            seasonal_periods=SEASONAL_PERIOD, initialization_method="estimated",
        ).fit(optimized=True)
        return model.forecast(horizon)
    except Exception:
        # if Holt-Winters fails to converge (can happen on noisy/short data),
        # fall back to trend extrapolation rather than crashing
        idx = np.arange(n)
        slope, intercept = np.polyfit(idx, series, 1)
        future_idx = np.arange(n, n + horizon)
        return slope * future_idx + intercept


def forecast_all_sensors(history_df: pd.DataFrame, horizon_hours: float) -> pd.DataFrame:
    """
    history_df: recent readings for ONE motor, columns = SENSOR_COLS + 'Timestamp',
                sorted ascending by time, ideally >= 48h of 15-min samples.
    horizon_hours: how far ahead to forecast.

    Returns a DataFrame of horizon rows with forecasted sensor values and
    a continued Timestamp index.
    """
    horizon_steps = int(horizon_hours * SAMPLES_PER_HOUR)
    last_ts = pd.to_datetime(history_df['Timestamp'].iloc[-1])
    future_timestamps = [last_ts + pd.Timedelta(minutes=15 * (i + 1)) for i in range(horizon_steps)]

    forecasted = {'Timestamp': future_timestamps}
    for col in SENSOR_COLS:
        forecasted[col] = forecast_single_sensor(history_df[col].values, horizon_steps)

    return pd.DataFrame(forecasted)


def forecast_health_and_rul(history_df: pd.DataFrame, horizon_hours: float,
                              scaler, rul_model, health_model) -> pd.DataFrame:
    """
    The full chain: forecast sensors forward -> engineer features on
    (history + forecast) so rolling/trend windows have real context ->
    run the trained RUL/Health models on the forecasted portion only.

    Returns a DataFrame with one row per forecasted timestep:
    Timestamp, predicted_RUL_hours, predicted_Health_Score, plus the
    forecasted raw sensor values (useful for showing "why" on a dashboard).
    """
    from feature_utils import build_features_for_motor, get_all_feature_columns

    sensor_forecast = forecast_all_sensors(history_df, horizon_hours)

    # Concatenate real history + forecasted sensors so rolling/trend features
    # computed on the forecast portion have real preceding context, not zeros.
    combined = pd.concat([history_df, sensor_forecast], ignore_index=True)
    combined = build_features_for_motor(combined, scaler)

    feature_cols = get_all_feature_columns()
    forecast_portion = combined.iloc[len(history_df):].reset_index(drop=True)

    X = forecast_portion[feature_cols].fillna(0).values
    forecast_portion['predicted_RUL_hours'] = rul_model.predict(X)
    forecast_portion['predicted_Health_Score'] = health_model.predict(X)

    keep_cols = ['Timestamp'] + SENSOR_COLS + ['predicted_RUL_hours', 'predicted_Health_Score']
    return forecast_portion[keep_cols]
