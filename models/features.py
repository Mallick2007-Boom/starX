"""
Feature engineering pipeline for polar research station microgrid forecasting.
Extracts domain-driven thermal, operational, cyclical temporal, and lag features.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


FEATURE_COLS_CRITICAL: List[str] = [
    "outdoor_temp_c",
    "heating_deficit",
    "wind_chill_cooling",
    "temp_lag_1h",
    "temp_lag_24h",
    "temp_rolling_mean_24h",
    "sin_hour",
    "cos_hour",
    "sin_dayofyear",
    "cos_dayofyear",
    "wind_speed_mps",
    "daylight_hours",
]

FEATURE_COLS_FLEXIBLE: List[str] = [
    "crew_headcount",
    "is_work_hour",
    "crew_work_interaction",
    "daylight_hours",
    "solar_available_kw",
    "sin_hour",
    "cos_hour",
    "day_of_week",
    "sin_dayofyear",
    "cos_dayofyear",
    "wind_speed_mps",
]


def extract_features(
    df: pd.DataFrame,
    is_training: bool = True,
    fill_lags_from: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Transform raw polar telemetry into rich predictive feature set.

    Args:
        df: DataFrame containing at least 'timestamp' and exogenous weather/crew columns.
        is_training: Whether historical target values are present.
        fill_lags_from: Optional recent historical DataFrame used to compute
                        lag features for future forecast horizons.

    Returns:
        DataFrame augmented with engineered features.
    """
    data = df.copy()

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
        data["timestamp"] = pd.to_datetime(data["timestamp"])

    # Temporal features
    hours = data["timestamp"].dt.hour.values
    dayofyear = data["timestamp"].dt.dayofyear.values
    data["hour"] = hours
    data["day_of_week"] = data["timestamp"].dt.dayofweek.values

    # Cyclical sine/cosine transforms
    data["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    data["sin_dayofyear"] = np.sin(2 * np.pi * dayofyear / 365.25)
    data["cos_dayofyear"] = np.cos(2 * np.pi * dayofyear / 365.25)

    # Operational schedule: Station work hours (08:00 - 18:00)
    data["is_work_hour"] = ((hours >= 8) & (hours <= 18)).astype(float)

    # Crew work activity interaction
    if "crew_headcount" in data.columns:
        data["crew_work_interaction"] = (
            data["crew_headcount"] * data["is_work_hour"]
        )
    else:
        data["crew_headcount"] = 35.0
        data["crew_work_interaction"] = 35.0 * data["is_work_hour"]

    # Thermal physics features
    temp = data["outdoor_temp_c"].values
    data["heating_deficit"] = np.maximum(0.0, -temp)

    # Wind chill convective building heat loss
    wind = data["wind_speed_mps"].values if "wind_speed_mps" in data.columns else np.zeros_like(temp)
    data["wind_chill_cooling"] = data["heating_deficit"] * (1.0 + 0.03 * wind)

    # Solar availability default if missing
    if "solar_available_kw" not in data.columns:
        data["solar_available_kw"] = 0.0

    # Lags and rolling features
    # If fill_lags_from is provided (e.g. recent history before forecast horizon), concatenate for lag calculation
    if fill_lags_from is not None and len(fill_lags_from) > 0:
        combined_temp = pd.concat(
            [fill_lags_from["outdoor_temp_c"], data["outdoor_temp_c"]],
            ignore_index=True,
        )
        offset = len(fill_lags_from)
        n_points = len(data)

        temp_series = combined_temp
        data["temp_lag_1h"] = temp_series.shift(1).iloc[offset : offset + n_points].values
        data["temp_lag_24h"] = temp_series.shift(24).iloc[offset : offset + n_points].values
        data["temp_rolling_mean_24h"] = (
            temp_series.rolling(24, min_periods=1)
            .mean()
            .iloc[offset : offset + n_points]
            .values
        )
    else:
        # Standard within-dataset lags
        temp_series = data["outdoor_temp_c"]
        data["temp_lag_1h"] = temp_series.shift(1).bfill().values
        data["temp_lag_24h"] = temp_series.shift(24).bfill().values
        data["temp_rolling_mean_24h"] = (
            temp_series.rolling(24, min_periods=1).mean().values
        )

    # Fill any remaining NaNs safely
    data["temp_lag_1h"] = data["temp_lag_1h"].bfill().fillna(data["outdoor_temp_c"])
    data["temp_lag_24h"] = data["temp_lag_24h"].bfill().fillna(data["outdoor_temp_c"])
    data["temp_rolling_mean_24h"] = (
        data["temp_rolling_mean_24h"].bfill().fillna(data["outdoor_temp_c"])
    )

    return data


def prepare_training_matrices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Extract feature matrices (X_crit, y_crit) and (X_flex, y_flex) for training.
    """
    featured = extract_features(df, is_training=True)

    X_crit = featured[FEATURE_COLS_CRITICAL].copy()
    y_crit = featured["load_critical_kw"].copy()

    X_flex = featured[FEATURE_COLS_FLEXIBLE].copy()
    y_flex = featured["load_flexible_kw"].copy()

    return X_crit, y_crit, X_flex, y_flex
