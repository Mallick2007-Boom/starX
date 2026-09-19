"""
Renewable Generation Forecasting Module for Polar Research Station Microgrid.
Predicts solar PV availability and wind turbine power generation with physics-informed
blade icing derating to feed Module C (the microgrid fuel optimizer).
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def icing_loss_factor(
    temp_c: Union[float, np.ndarray, pd.Series],
    humidity: Union[float, np.ndarray, pd.Series],
    wind_speed: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray]:
    """
    Physics-informed aerodynamic derating heuristic for polar wind turbines.

    Atmospheric Physics Citation & Scientific Justification:
    --------------------------------------------------------
    In cold-climate polar environments, blade icing occurs when supercooled liquid
    water droplets in freezing fog or marine clouds collide with the leading edge
    of rotating turbine blades (rime ice and glaze accretion). According to empirical
    polar field studies and cold-climate guidelines (IEA Wind TCP Task 19: "Wind Energy
    in Cold Climates" and Antarctic research at Ross Island / McMurdo):
      1. Peak icing risk occurs between -20°C and -1°C. Supercooled liquid water droplets
         freeze instantly on blade impact, disrupting the aerofoil boundary layer,
         drastically reducing lift and increasing drag.
      2. High relative humidity (RH >= 70%) indicates freezing fog/cloud saturation.
      3. Elevated wind speeds accelerate droplet impingement rates and heat transfer,
         hastening rime accretion until storm cut-out.
      4. At temperatures below -25°C to -30°C, extreme atmospheric dryness (Clausius-Clapeyron)
         causes moisture to exist purely as dry ice crystals which bounce off blades,
         reducing rime accretion rates.
      5. At temperatures above 0°C, freezing does not occur.

    This heuristic model calculates the aerodynamic derating factor:
      - 1.00 : Clean blades (optimal aerodynamic efficiency, 100% nominal power)
      - 0.15 : Severe icing derating (up to 85% aerodynamic power loss)

    Args:
        temp_c: Ambient outdoor temperature (°C)
        humidity: Relative humidity (%)
        wind_speed: Ambient wind speed (m/s)

    Returns:
        derating_factor: Multiplier in [0.15, 1.00] to scale raw wind power.
    """
    is_scalar = np.isscalar(temp_c)
    t = np.asarray(temp_c, dtype=float)
    h = np.asarray(humidity, dtype=float)
    w = np.asarray(wind_speed, dtype=float)

    # 1. Temperature susceptibility window:
    # Maximum icing susceptibility centered around -10°C to -5°C (freezing fog zone)
    # Tapers off above -1°C (melting) and below -22°C (sublimation/dry ice)
    temp_risk = np.zeros_like(t)
    in_icing_temp = (t >= -22.0) & (t <= -1.0)
    # Bell curve centered at -8°C
    temp_risk[in_icing_temp] = np.exp(-0.5 * ((t[in_icing_temp] - (-8.0)) / 6.0) ** 2)

    # 2. Moisture susceptibility:
    # Freezing fog requires high relative humidity (> 60%)
    humidity_risk = np.clip((h - 60.0) / 30.0, 0.0, 1.0)

    # 3. Wind exposure factor:
    # Moderate to strong winds (5-18 m/s) drive impingement rate of supercooled droplets
    wind_exposure = np.clip(w / 12.0, 0.2, 1.0)

    # Combined icing accretion severity index [0.0 to 1.0]
    icing_severity = temp_risk * humidity_risk * wind_exposure

    # Derate power up to 85% (factor drops from 1.0 down to 0.15)
    max_derate = 0.85
    factor = 1.0 - (max_derate * icing_severity)
    factor = np.clip(factor, 0.15, 1.0)

    if is_scalar:
        return float(factor.item())
    return factor


class RenewablesForecaster:
    """
    Forecasting engine for polar solar PV and wind microgrid generation.
    Incorporates astronomical daylight gating and aerodynamic icing derating.
    """

    def __init__(
        self,
        pv_capacity_kw: float = 120.0,
        wind_rated_capacity_kw: float = 100.0,
        wind_cut_in_mps: float = 3.0,
        wind_rated_mps: float = 12.0,
        wind_cut_out_mps: float = 25.0,
    ):
        self.pv_capacity_kw = pv_capacity_kw
        self.wind_rated_capacity_kw = wind_rated_capacity_kw
        self.wind_cut_in_mps = wind_cut_in_mps
        self.wind_rated_mps = wind_rated_mps
        self.wind_cut_out_mps = wind_cut_out_mps

        # Lightweight regressor for solar availability outside polar night
        self.solar_model = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42,
        )
        self.is_fitted = False

    def _extract_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract daylight and astronomical features for solar prediction."""
        feat = pd.DataFrame(index=df.index)
        ts = pd.to_datetime(df["timestamp"])

        feat["daylight_hours"] = df["daylight_hours"].values
        feat["hour"] = ts.dt.hour.values
        feat["dayofyear"] = ts.dt.dayofyear.values
        feat["sin_hour"] = np.sin(2 * np.pi * feat["hour"] / 24.0)
        feat["cos_hour"] = np.cos(2 * np.pi * feat["hour"] / 24.0)
        feat["sin_day"] = np.sin(2 * np.pi * feat["dayofyear"] / 365.25)
        feat["cos_day"] = np.cos(2 * np.pi * feat["dayofyear"] / 365.25)

        # Weather attenuation if available
        if "wind_speed_mps" in df.columns:
            feat["wind_speed_mps"] = df["wind_speed_mps"].values
        else:
            feat["wind_speed_mps"] = 8.0

        if "outdoor_temp_c" in df.columns:
            feat["outdoor_temp_c"] = df["outdoor_temp_c"].values
        else:
            feat["outdoor_temp_c"] = -25.0

        return feat

    def calculate_raw_wind_kw(
        self, wind_speed_mps: Union[float, np.ndarray, pd.Series]
    ) -> Union[float, np.ndarray]:
        """
        Compute theoretical turbine power curve (pre-icing) from wind speed.
        Cubic power ramp between cut-in and rated speed, flat at rated, 0 at cutout.
        """
        is_scalar = np.isscalar(wind_speed_mps)
        v = np.asarray(wind_speed_mps, dtype=float)
        p = np.zeros_like(v)

        v_in = self.wind_cut_in_mps
        v_rated = self.wind_rated_mps
        v_out = self.wind_cut_out_mps
        p_rated = self.wind_rated_capacity_kw

        # Cubic power ramp
        cubic_mask = (v >= v_in) & (v < v_rated)
        p[cubic_mask] = p_rated * ((v[cubic_mask] - v_in) / (v_rated - v_in)) ** 3

        # Rated plateau
        rated_mask = (v >= v_rated) & (v <= v_out)
        p[rated_mask] = p_rated

        # Above cutout: storm survival brake -> 0 kW
        p[v > v_out] = 0.0

        if is_scalar:
            return float(p.item())
        return p

    def fit(self, train_df: pd.DataFrame) -> "RenewablesForecaster":
        """
        Fit solar regression model on daytime observations.
        During polar night, solar is 0 by physical law and requires no fitting.
        """
        # Train solar model on non-polar night records
        daytime_mask = (train_df["daylight_hours"] > 0) & (train_df["solar_available_kw"] > 0)
        if daytime_mask.sum() > 50:
            X_solar = self._extract_solar_features(train_df[daytime_mask])
            y_solar = train_df.loc[daytime_mask, "solar_available_kw"].values
            self.solar_model.fit(X_solar, y_solar)

        self.is_fitted = True
        return self

    def predict_solar_kw(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict solar PV generation. Strictly enforces 0 kW during polar night.
        """
        n_steps = len(df)
        pred_solar = np.zeros(n_steps)

        # 1. Hard polar night & nighttime mask
        daylight = df["daylight_hours"].values
        ts = pd.to_datetime(df["timestamp"])
        hour = ts.dt.hour.values

        # Polar night filter: daylight == 0 strictly yields 0 kW
        is_polar_night = daylight == 0.0

        # Daylight active filter
        active_daylight = (~is_polar_night) & (daylight > 0.0)

        if active_daylight.any() and self.is_fitted:
            X_solar = self._extract_solar_features(df[active_daylight])
            preds = self.solar_model.predict(X_solar)
            # Clip between 0 and PV nameplate capacity
            pred_solar[active_daylight] = np.clip(preds, 0.0, self.pv_capacity_kw)

        # Force strict 0.0 during polar night
        pred_solar[is_polar_night] = 0.0
        return np.round(pred_solar, 2)

    def predict_wind_kw(
        self,
        wind_speed_mps: np.ndarray,
        temp_c: np.ndarray,
        humidity_pct: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate raw wind power, icing loss factor, and post-icing effective wind power.

        Returns:
            (wind_raw_kw, icing_factor, wind_effective_kw)
        """
        if humidity_pct is None:
            # Approximate humidity based on polar temperature:
            # Mild polar maritime air (-10°C to 0°C) is moist (75-90%)
            # Deep winter continental air (-50°C) is very dry (35-50%)
            temp_norm = np.clip((temp_c - (-55.0)) / 55.0, 0.0, 1.0)
            humidity_pct = 40.0 + 45.0 * temp_norm

        wind_raw_kw = self.calculate_raw_wind_kw(wind_speed_mps)
        icing_factor = icing_loss_factor(temp_c, humidity_pct, wind_speed_mps)
        wind_effective_kw = wind_raw_kw * icing_factor

        return (
            np.round(wind_raw_kw, 2),
            np.round(icing_factor, 4),
            np.round(wind_effective_kw, 2),
        )

    def forecast_renewables(
        self,
        current_data: pd.DataFrame,
        horizon_hours: int = 48,
        future_weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Forecast solar PV and wind generation (post-icing) for the next 24 to 72 hours.
        """
        data = current_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["timestamp"]):
            data["timestamp"] = pd.to_datetime(data["timestamp"])

        last_timestamp = data["timestamp"].max()

        # Build future time-series horizon
        if future_weather is not None and len(future_weather) >= horizon_hours:
            future_df = future_weather.iloc[:horizon_hours].copy()
            if not pd.api.types.is_datetime64_any_dtype(future_df["timestamp"]):
                future_df["timestamp"] = pd.to_datetime(future_df["timestamp"])
        else:
            future_times = pd.date_range(
                start=last_timestamp + pd.Timedelta(hours=1),
                periods=horizon_hours,
                freq="h",
            )
            last_row = data.iloc[-1]
            future_df = pd.DataFrame({"timestamp": future_times})

            # Extrapolate weather
            base_temp = last_row.get("outdoor_temp_c", -25.0)
            hours_ahead = future_times.hour.values
            diurnal_swing = 2.5 * np.sin(2 * np.pi * (hours_ahead - 9) / 24.0)
            future_df["outdoor_temp_c"] = base_temp + diurnal_swing
            future_df["wind_speed_mps"] = last_row.get("wind_speed_mps", 8.5)
            future_df["daylight_hours"] = last_row.get("daylight_hours", 12.0)
            if "relative_humidity_pct" in last_row:
                future_df["relative_humidity_pct"] = last_row["relative_humidity_pct"]
            else:
                future_df["relative_humidity_pct"] = 75.0

        # Predict solar
        solar_pred_kw = self.predict_solar_kw(future_df)

        # Predict wind
        temp = future_df["outdoor_temp_c"].values
        wind_speed = future_df["wind_speed_mps"].values
        humidity = (
            future_df["relative_humidity_pct"].values
            if "relative_humidity_pct" in future_df.columns
            else None
        )
        wind_raw_kw, icing_factors, wind_kw = self.predict_wind_kw(
            wind_speed, temp, humidity
        )

        total_renewable_kw = np.round(solar_pred_kw + wind_kw, 2)

        return pd.DataFrame(
            {
                "timestamp": future_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                "solar_kw": solar_pred_kw,
                "wind_raw_kw": wind_raw_kw,
                "icing_factor": icing_factors,
                "wind_kw": wind_kw,  # post-icing effective wind generation
                "total_renewable_kw": total_renewable_kw,
                "forecast_temp_c": np.round(temp, 2),
                "forecast_wind_speed_mps": np.round(wind_speed, 2),
            }
        )


# Global cached instance
_CACHED_RENEWABLES_FORECASTER: Optional[RenewablesForecaster] = None


def get_default_renewables_forecaster(
    data_path: Optional[str] = None,
) -> RenewablesForecaster:
    """Initialize and train default renewables forecaster."""
    global _CACHED_RENEWABLES_FORECASTER
    if _CACHED_RENEWABLES_FORECASTER is not None and _CACHED_RENEWABLES_FORECASTER.is_fitted:
        return _CACHED_RENEWABLES_FORECASTER

    from pathlib import Path
    if data_path is None:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "data_generator"
            / "data"
            / "polar_station_energy_hourly.csv"
        )

    df = pd.read_csv(data_path)
    forecaster = RenewablesForecaster()
    forecaster.fit(df)
    _CACHED_RENEWABLES_FORECASTER = forecaster
    return _CACHED_RENEWABLES_FORECASTER


def forecast_renewables(
    current_data: pd.DataFrame,
    horizon_hours: int = 48,
    future_weather: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience function: Forecast solar_kw and post-icing wind_kw
    for the next 24 to 72 hours to feed the fuel optimizer (Module C).

    Args:
        current_data: Recent telemetry observations.
        horizon_hours: Forecasting lookahead horizon (default: 48).
        future_weather: Optional DataFrame with forecasted weather variables.

    Returns:
        DataFrame with columns:
            - timestamp
            - solar_kw
            - wind_raw_kw
            - icing_factor
            - wind_kw (post-icing effective wind power)
            - total_renewable_kw
    """
    forecaster = get_default_renewables_forecaster()
    return forecaster.forecast_renewables(
        current_data=current_data,
        horizon_hours=horizon_hours,
        future_weather=future_weather,
    )
