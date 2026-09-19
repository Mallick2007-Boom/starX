"""
Gradient Boosting Load Forecaster for Polar Research Station Microgrid.
Provides 24h to 72h horizon forecasts for critical life-support heating load
and flexible science/operational load with feature explainability and evaluation.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import BaseLoadForecaster
from .features import (
    FEATURE_COLS_CRITICAL,
    FEATURE_COLS_FLEXIBLE,
    extract_features,
    prepare_training_matrices,
)


class GradientBoostingLoadForecaster(BaseLoadForecaster):
    """
    Lightweight, high-accuracy gradient boosting forecaster for polar loads.
    Trains separate gradient boosting models for critical and flexible loads.
    """

    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.model_critical = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=0.1,
            random_state=self.random_state,
        )

        self.model_flexible = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=0.1,
            random_state=self.random_state,
        )

        self.is_fitted = False
        self.feature_importance_cache: Optional[Dict[str, Dict[str, float]]] = None
        self._last_training_df: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame) -> "GradientBoostingLoadForecaster":
        """
        Fit gradient boosting models on historical telemetry data.
        Fast training (< 2 seconds on 1 year of hourly data).
        """
        X_crit, y_crit, X_flex, y_flex = prepare_training_matrices(train_df)

        self.model_critical.fit(X_crit, y_crit)
        self.model_flexible.fit(X_flex, y_flex)

        self.is_fitted = True
        self._last_training_df = train_df.copy()
        self.feature_importance_cache = None

        return self

    def get_feature_importance(
        self, eval_df: Optional[pd.DataFrame] = None, n_repeats: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute permutation feature importance for both critical and flexible models.
        Reveals the physical and operational drivers of the forecast.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing feature importance.")

        if self.feature_importance_cache is not None and eval_df is None:
            return self.feature_importance_cache

        df_source = eval_df if eval_df is not None else self._last_training_df
        if df_source is None:
            raise ValueError("No data available to calculate feature importance.")

        # Sample up to 2000 points for fast computation during hackathon demo
        sample_df = df_source.sample(n=min(2000, len(df_source)), random_state=self.random_state)
        X_crit, y_crit, X_flex, y_flex = prepare_training_matrices(sample_df)

        res_crit = permutation_importance(
            self.model_critical,
            X_crit,
            y_crit,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_absolute_error",
        )

        res_flex = permutation_importance(
            self.model_flexible,
            X_flex,
            y_flex,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_absolute_error",
        )

        imp_crit = {}
        for col, mean_score in zip(FEATURE_COLS_CRITICAL, res_crit.importances_mean):
            imp_crit[col] = float(max(0.0, mean_score))

        imp_flex = {}
        for col, mean_score in zip(FEATURE_COLS_FLEXIBLE, res_flex.importances_mean):
            imp_flex[col] = float(max(0.0, mean_score))

        # Normalize to percentage shares
        sum_crit = sum(imp_crit.values()) or 1.0
        sum_flex = sum(imp_flex.values()) or 1.0

        normalized_crit = {k: round(v / sum_crit * 100.0, 2) for k, v in sorted(imp_crit.items(), key=lambda x: x[1], reverse=True)}
        normalized_flex = {k: round(v / sum_flex * 100.0, 2) for k, v in sorted(imp_flex.items(), key=lambda x: x[1], reverse=True)}

        results = {
            "load_critical_drivers": normalized_crit,
            "load_flexible_drivers": normalized_flex,
        }

        if eval_df is None:
            self.feature_importance_cache = results

        return results

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate forecasting accuracy on held-out test data (MAE, RMSE, R^2).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation.")

        X_crit, y_crit, X_flex, y_flex = prepare_training_matrices(test_df)

        pred_crit = self.model_critical.predict(X_crit)
        pred_flex = self.model_flexible.predict(X_flex)

        # Baseline safety clip
        pred_crit = np.maximum(35.0, pred_crit)
        pred_flex = np.maximum(8.0, pred_flex)

        mae_crit = mean_absolute_error(y_crit, pred_crit)
        rmse_crit = np.sqrt(mean_squared_error(y_crit, pred_crit))
        r2_crit = r2_score(y_crit, pred_crit)

        mae_flex = mean_absolute_error(y_flex, pred_flex)
        rmse_flex = np.sqrt(mean_squared_error(y_flex, pred_flex))
        r2_flex = r2_score(y_flex, pred_flex)

        total_actual = y_crit + y_flex
        total_pred = pred_crit + pred_flex
        mae_total = mean_absolute_error(total_actual, total_pred)
        rmse_total = np.sqrt(mean_squared_error(total_actual, total_pred))

        return {
            "load_critical_kw": {
                "MAE": round(float(mae_crit), 3),
                "RMSE": round(float(rmse_crit), 3),
                "R2": round(float(r2_crit), 4),
            },
            "load_flexible_kw": {
                "MAE": round(float(mae_flex), 3),
                "RMSE": round(float(rmse_flex), 3),
                "R2": round(float(r2_flex), 4),
            },
            "total_station_load_kw": {
                "MAE": round(float(mae_total), 3),
                "RMSE": round(float(rmse_total), 3),
            },
        }

    def forecast_load(
        self,
        current_data: pd.DataFrame,
        horizon_hours: int = 48,
        future_exogenous: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce future load forecasts for horizon_hours (24 to 72 hours).

        Args:
            current_data: Recent telemetry dataframe ending at the current time.
            horizon_hours: Number of hours to forecast ahead (24-72 hours).
            future_exogenous: Optional future weather/crew forecast. If not provided,
                              future conditions are extrapolated using recent state
                              and polar astronomical cycles.

        Returns:
            DataFrame with timestamps and forecasted loads (critical, flexible, total).
        """
        if not self.is_fitted:
            raise RuntimeError("Forecaster has not been fitted. Call fit() first.")

        data = current_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        last_timestamp = data["timestamp"].max()

        # Build future dataframe for the horizon
        if future_exogenous is not None and len(future_exogenous) >= horizon_hours:
            future_df = future_exogenous.iloc[:horizon_hours].copy()
            if not pd.api.types.is_datetime64_any_dtype(future_df["timestamp"]):
                future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
        else:
            # Generate future timestamps
            future_times = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=horizon_hours,
                freq="h",
            )
            # Extrapolate exogenous variables from last known state
            last_row = data.iloc[-1]
            future_df = pd.DataFrame({"timestamp": future_times})

            # Extrapolate temperature with diurnal variation
            base_temp = last_row.get("outdoor_temp_c", -25.0)
            hours_ahead = np.arange(1, horizon_hours + 1)
            future_hours = future_times.hour.values
            diurnal_swing = 2.5 * np.sin(2 * np.pi * (future_hours - 9) / 24.0)
            future_df["outdoor_temp_c"] = base_temp + diurnal_swing

            # Crew headcount assumed constant across the 48-72h forecast
            future_df["crew_headcount"] = last_row.get("crew_headcount", 35)

            # Wind speed persistence
            future_df["wind_speed_mps"] = last_row.get("wind_speed_mps", 8.0)

            # Solar and daylight persistence
            future_df["daylight_hours"] = last_row.get("daylight_hours", 12.0)
            future_df["solar_available_kw"] = last_row.get("solar_available_kw", 0.0)

        # Feature extraction using recent history for lag computation
        history_window = data.tail(48)
        featured_future = extract_features(
            future_df, is_training=False, fill_lags_from=history_window
        )

        X_crit = featured_future[FEATURE_COLS_CRITICAL]
        X_flex = featured_future[FEATURE_COLS_FLEXIBLE]

        pred_crit = self.model_critical.predict(X_crit)
        pred_flex = self.model_flexible.predict(X_flex)

        # Apply physics safety bounds
        pred_crit = np.maximum(35.0, pred_crit)
        pred_flex = np.maximum(8.0, pred_flex)
        pred_total = pred_crit + pred_flex

        forecast_df = pd.DataFrame(
            {
                "timestamp": future_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "load_critical_pred_kw": np.round(pred_crit, 2),
                "load_flexible_pred_kw": np.round(pred_flex, 2),
                "total_load_pred_kw": np.round(pred_total, 2),
                "forecast_outdoor_temp_c": np.round(future_df["outdoor_temp_c"], 2),
                "crew_headcount": future_df["crew_headcount"].values,
            }
        )

        return forecast_df


# Global cached singleton forecaster for fast interactive API/hackathon queries
_CACHED_FORECASTER: Optional[GradientBoostingLoadForecaster] = None


def get_default_forecaster(data_path: Optional[str] = None) -> GradientBoostingLoadForecaster:
    """Load or initialize and train the default gradient boosting forecaster."""
    global _CACHED_FORECASTER
    if _CACHED_FORECASTER is not None and _CACHED_FORECASTER.is_fitted:
        return _CACHED_FORECASTER

    from pathlib import Path
    if data_path is None:
        data_path = Path(__file__).resolve().parent.parent / "data_generator" / "data" / "polar_station_energy_hourly.csv"

    df = pd.read_csv(data_path)
    forecaster = GradientBoostingLoadForecaster()
    forecaster.fit(df)
    _CACHED_FORECASTER = forecaster
    return _CACHED_FORECASTER


def forecast_load(
    current_data: pd.DataFrame,
    horizon_hours: int = 48,
    future_exogenous: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper function: Forecasts critical and flexible loads
    for the next horizon_hours (24 to 72 hours).

    Args:
        current_data: Recent telemetry data (or full historical dataframe).
        horizon_hours: Lookahead hours (default: 48, supports 24-72 hours).
        future_exogenous: Optional future weather/crew forecast.

    Returns:
        DataFrame with predictions:
            - timestamp
            - load_critical_pred_kw
            - load_flexible_pred_kw
            - total_load_pred_kw
    """
    forecaster = get_default_forecaster()
    return forecaster.forecast_load(
        current_data=current_data,
        horizon_hours=horizon_hours,
        future_exogenous=future_exogenous,
    )
