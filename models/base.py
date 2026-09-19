"""
Base interface for polar microgrid load forecasting models.
Enables swapping between lightweight gradient boosting, LSTM, or Transformer models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd


class BaseLoadForecaster(ABC):
    """Abstract base class for polar station critical and flexible load forecasters."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "BaseLoadForecaster":
        """
        Train the forecasting model on historical telemetry data.

        Args:
            train_df: DataFrame containing historical timestamps, weather,
                      crew headcount, and loads.
        Returns:
            self
        """
        pass

    @abstractmethod
    def forecast_load(
        self,
        current_data: pd.DataFrame,
        horizon_hours: int = 48,
        future_exogenous: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Forecast critical and flexible loads for the next 24 to 72 hours.

        Args:
            current_data: Recent historical telemetry (at least 24-48 hours)
                          used for lag features and state initialization.
            horizon_hours: Forecast lookahead horizon (typically 24, 48, or 72 hours).
            future_exogenous: Optional DataFrame containing known/forecasted weather
                              (outdoor_temp_c, wind_speed_mps, daylight_hours)
                              and crew_headcount for the horizon period.

        Returns:
            DataFrame with columns:
                - timestamp
                - load_critical_pred_kw
                - load_flexible_pred_kw
                - total_load_pred_kw
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Return explainability metrics showing the relative importance
        of features for critical load and flexible load predictions.
        """
        pass
