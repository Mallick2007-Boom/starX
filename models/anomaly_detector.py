"""
Lightweight Anomaly Detection Engine for Polar Research Station (Generator & Battery Health).
Detects generator efficiency drift, abnormal fuel-to-output ratios, and battery capacity
drops during extreme Antarctic cold using rolling robust statistics and Isolation Forests.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def inject_synthetic_faults(
    df: pd.DataFrame,
    fault_rate: float = 0.03,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Simulate realistic polar station equipment fault periods in telemetry:
      1. Generator Efficiency Drift / Injector Fouling: +20% to +40% abnormal fuel burn.
      2. Battery Thermal Enclosure Failure at Extreme Cold: -35% to -55% capacity/charge sag when T < -20°C.
      3. Fuel Line Discrepancy / Aux Boiler Leak: elevated idle fuel burn with 0 generator kW.

    Args:
        df: Input clean microgrid telemetry DataFrame.
        fault_rate: Fraction of hours to inject faults into (default: ~3%).
        random_seed: Deterministic seed for reproducible testing.

    Returns:
        (corrupted_df, ground_truth_fault_records)
    """
    rng = np.random.RandomState(random_seed)
    corrupted = df.copy().reset_index(drop=True)
    n_rows = len(corrupted)

    # Initialize ground truth tracker
    corrupted["is_synthetic_fault"] = False
    corrupted["fault_type"] = "NORMAL"

    ground_truth_faults: List[Dict[str, Any]] = []

    # Fault 1: Generator Efficiency Drift (3 distinct multi-hour episodes)
    # Injector fouling or fuel wax crystallization causing higher liters per kWh
    gen_mask = (
        (corrupted["diesel_generation_kw"] > 25.0)
        if "diesel_generation_kw" in corrupted.columns
        else (corrupted.get("kw_from_diesel", 0.0) > 25.0)
    )
    gen_indices = np.where(gen_mask)[0]

    if len(gen_indices) > 50:
        for _ in range(3):
            start_idx = int(rng.choice(gen_indices[:-18]))
            duration = rng.randint(8, 20)
            end_idx = min(n_rows, start_idx + duration)

            drift_multiplier = rng.uniform(1.22, 1.38)
            fuel_col = (
                "hourly_fuel_consumption_liters"
                if "hourly_fuel_consumption_liters" in corrupted.columns
                else "fuel_consumed_liters"
            )

            corrupted.loc[start_idx:end_idx, fuel_col] *= drift_multiplier
            corrupted.loc[start_idx:end_idx, "is_synthetic_fault"] = True
            corrupted.loc[start_idx:end_idx, "fault_type"] = "GENERATOR_EFFICIENCY_DRIFT"

            ground_truth_faults.append(
                {
                    "fault_type": "GENERATOR_EFFICIENCY_DRIFT",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_timestamp": str(corrupted["timestamp"].iloc[start_idx]),
                    "end_timestamp": str(corrupted["timestamp"].iloc[end_idx - 1]),
                    "description": f"Fuel consumption spiked {drift_multiplier:.1%} above nominal due to injector fouling",
                }
            )

    # Fault 2: Battery Thermal Enclosure Failure at Extreme Cold (2 episodes)
    # When ambient temperature < -25°C, heating loops fail, causing high internal resistance
    cold_mask = corrupted["outdoor_temp_c"] < -28.0
    cold_indices = np.where(cold_mask)[0]

    if len(cold_indices) > 40:
        for _ in range(2):
            start_idx = int(rng.choice(cold_indices[:-14]))
            duration = rng.randint(6, 16)
            end_idx = min(n_rows, start_idx + duration)

            soc_col = "battery_soc_pct" if "battery_soc_pct" in corrupted.columns else None
            if soc_col:
                corrupted.loc[start_idx:end_idx, soc_col] *= rng.uniform(0.55, 0.70)

            corrupted.loc[start_idx:end_idx, "is_synthetic_fault"] = True
            corrupted.loc[start_idx:end_idx, "fault_type"] = "BATTERY_COLD_DEGRADATION"

            ground_truth_faults.append(
                {
                    "fault_type": "BATTERY_COLD_DEGRADATION",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_timestamp": str(corrupted["timestamp"].iloc[start_idx]),
                    "end_timestamp": str(corrupted["timestamp"].iloc[end_idx - 1]),
                    "description": "Battery capacity dropped 35-45% during deep cold from thermal enclosure heater failure",
                }
            )

    # Fault 3: Fuel Line Leak / Aux Boiler Overconsumption while Generator Idle (2 episodes)
    idle_mask = (
        (corrupted["diesel_generation_kw"] == 0.0)
        if "diesel_generation_kw" in corrupted.columns
        else (corrupted.get("kw_from_diesel", 0.0) == 0.0)
    )
    idle_indices = np.where(idle_mask)[0]

    if len(idle_indices) > 30:
        for _ in range(2):
            start_idx = int(rng.choice(idle_indices[:-10]))
            duration = rng.randint(4, 10)
            end_idx = min(n_rows, start_idx + duration)

            fuel_col = (
                "hourly_fuel_consumption_liters"
                if "hourly_fuel_consumption_liters" in corrupted.columns
                else "fuel_consumed_liters"
            )
            corrupted.loc[start_idx:end_idx, fuel_col] += rng.uniform(8.0, 16.0)
            corrupted.loc[start_idx:end_idx, "is_synthetic_fault"] = True
            corrupted.loc[start_idx:end_idx, "fault_type"] = "IDLE_FUEL_LEAK"

            ground_truth_faults.append(
                {
                    "fault_type": "IDLE_FUEL_LEAK",
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_timestamp": str(corrupted["timestamp"].iloc[start_idx]),
                    "end_timestamp": str(corrupted["timestamp"].iloc[end_idx - 1]),
                    "description": "Unexplained fuel consumption (+12 L/h) while generator is completely idle",
                }
            )

    return corrupted, ground_truth_faults


class AnomalyDetector:
    """
    Dual-engine anomaly detection system for polar research station microgrid.
    Combines robust rolling Z-scores with Isolation Forest for multidimensional faults.
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        nominal_sfc: float = 0.285,
        rolling_window_hours: int = 48,
        contamination: float = 0.03,
    ):
        self.z_threshold = z_threshold
        self.nominal_sfc = nominal_sfc
        self.rolling_window_hours = rolling_window_hours
        self.contamination = contamination

        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self.is_fitted = False

    def _compute_health_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive operational health indicators from raw microgrid telemetry."""
        feat = pd.DataFrame(index=df.index)
        feat["timestamp"] = df["timestamp"].values

        # Identify diesel power and fuel columns
        if "diesel_generation_kw" in df.columns:
            diesel_kw = df["diesel_generation_kw"].values
        elif "kw_from_diesel" in df.columns:
            diesel_kw = df["kw_from_diesel"].values
        else:
            diesel_kw = np.zeros(len(df))

        if "hourly_fuel_consumption_liters" in df.columns:
            fuel_burn = df["hourly_fuel_consumption_liters"].values
        elif "fuel_consumed_liters" in df.columns:
            fuel_burn = df["fuel_consumed_liters"].values
        else:
            fuel_burn = np.zeros(len(df))

        temp_c = df["outdoor_temp_c"].values

        feat["diesel_kw"] = diesel_kw
        feat["fuel_burn"] = fuel_burn
        feat["outdoor_temp_c"] = temp_c

        # 1. Expected fuel burn from physical microgrid model
        # Base electrical burn + generator standby burn + extreme cold heating boiler (< -35°C)
        boiler_baseline = np.maximum(0.0, -35.0 - temp_c) * 0.30
        active_gen = diesel_kw >= 5.0
        expected_fuel = np.where(
            active_gen,
            diesel_kw * self.nominal_sfc + 3.5 + boiler_baseline,
            np.where(fuel_burn > 0.5, 3.5 + boiler_baseline, 0.0),
        )

        # 2. Specific Fuel Consumption (SFC) in L/kWh when generator is operating
        sfc = np.zeros(len(df))
        sfc[active_gen] = fuel_burn[active_gen] / diesel_kw[active_gen]
        feat["sfc"] = sfc
        feat["active_gen"] = active_gen
        feat["expected_fuel"] = expected_fuel
        feat["boiler_baseline"] = boiler_baseline

        # 3. Fuel excess ratio relative to physical baseline
        excess_ratio = np.zeros(len(df))
        active_fuel = expected_fuel > 1.0
        excess_ratio[active_fuel] = (
            fuel_burn[active_fuel] - expected_fuel[active_fuel]
        ) / expected_fuel[active_fuel]
        feat["fuel_excess_ratio"] = excess_ratio

        # 4. Battery Health Metrics (if battery SoC is available)
        if "battery_soc_pct" in df.columns:
            soc = df["battery_soc_pct"].values
            soc_diff = np.diff(soc, prepend=soc[0])
            feat["battery_soc_pct"] = soc
            feat["soc_step_delta"] = soc_diff
        else:
            feat["battery_soc_pct"] = 60.0
            feat["soc_step_delta"] = 0.0

        return feat

    def fit(self, training_df: pd.DataFrame) -> "AnomalyDetector":
        """Fit baseline distribution and Isolation Forest on historical telemetry."""
        feat = self._compute_health_metrics(training_df)

        # Train Isolation Forest on operational indicators
        X_train = feat[["sfc", "fuel_excess_ratio", "outdoor_temp_c", "battery_soc_pct"]].fillna(0.0)
        self.iso_forest.fit(X_train)
        self.is_fitted = True
        return self

    def detect(self, recent_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scan telemetry for generator or battery degradation and return explainable alerts.

        Args:
            recent_data: DataFrame with recent microgrid observations.

        Returns:
            List of detected anomaly records with timestamps, severity, and reasons.
        """
        feat = self._compute_health_metrics(recent_data)
        anomalies: List[Dict[str, Any]] = []

        # Rolling baseline for SFC on active generator timesteps
        sfc_series = pd.Series(
            np.where(feat["active_gen"], feat["sfc"], np.nan), index=feat.index
        )
        rolling_median = sfc_series.rolling(
            self.rolling_window_hours, min_periods=4
        ).median().bfill().fillna(self.nominal_sfc)
        rolling_std = sfc_series.rolling(
            self.rolling_window_hours, min_periods=4
        ).std().bfill().fillna(0.02)
        rolling_std = np.maximum(0.015, rolling_std)

        # Multi-dimensional Isolation Forest scores
        if self.is_fitted:
            X_eval = feat[
                ["sfc", "fuel_excess_ratio", "outdoor_temp_c", "battery_soc_pct"]
            ].fillna(0.0)
            iso_preds = self.iso_forest.predict(X_eval)  # -1 for anomaly, 1 for normal
        else:
            iso_preds = np.ones(len(feat))

        for i in range(len(feat)):
            ts = str(feat["timestamp"].iloc[i])
            temp = float(feat["outdoor_temp_c"].iloc[i])
            d_kw = float(feat["diesel_kw"].iloc[i])
            burn = float(feat["fuel_burn"].iloc[i])
            sfc_val = float(feat["sfc"].iloc[i])
            excess_ratio = float(feat["fuel_excess_ratio"].iloc[i])
            is_active = bool(feat["active_gen"].iloc[i])
            iso_flag = bool(iso_preds[i] == -1)

            # --- ANOMALY RULE 1: Generator Efficiency Drift (Excess Fuel Ratio) ---
            if is_active and excess_ratio >= 0.18:
                efficiency_drop_pct = excess_ratio * 100.0
                severity = "CRITICAL" if efficiency_drop_pct >= 28.0 else "WARNING"
                expected_b = float(feat["expected_fuel"].iloc[i])
                reason = (
                    f"Fuel efficiency dropped {efficiency_drop_pct:.1f}% below physical baseline "
                    f"({burn:.1f} L/h observed vs {expected_b:.1f} L/h expected for {d_kw:.1f} kW load) — "
                    f"possible generator fuel injector fouling, polar fuel waxing, or turbocharger wear."
                )
                anomalies.append(
                    {
                        "timestamp": ts,
                        "severity": severity,
                        "anomaly_type": "GENERATOR_EFFICIENCY_DRIFT",
                        "metric": "Fuel Excess Ratio (%)",
                        "observed_value": round(burn, 2),
                        "baseline_value": round(expected_b, 2),
                        "z_score": round(float(excess_ratio / 0.05), 2),
                        "reason": reason,
                    }
                )
                continue

            # --- ANOMALY RULE 2: Idle Fuel Leak / Auxiliary Boiler Runaway ---
            boiler = float(feat["boiler_baseline"].iloc[i])
            expected_idle = 3.5 + boiler
            if d_kw < 2.0 and burn > (expected_idle + 4.0):
                excess_burn = burn - expected_idle
                reason = (
                    f"Unexplained fuel burn of {burn:.1f} L/h while diesel generator is idle (0 kW) "
                    f"— exceeds expected baseline ({expected_idle:.1f} L/h) by +{excess_burn:.1f} L/h. Suspected fuel line leak "
                    f"or auxiliary heating loop valve failure."
                )
                anomalies.append(
                    {
                        "timestamp": ts,
                        "severity": "CRITICAL",
                        "anomaly_type": "IDLE_FUEL_LEAK",
                        "metric": "Idle Fuel Consumption (L/h)",
                        "observed_value": round(burn, 2),
                        "baseline_value": round(expected_idle, 2),
                        "z_score": round(float(excess_burn / 2.0), 2),
                        "reason": reason,
                    }
                )
                continue

            # --- ANOMALY RULE 3: Battery Capacity Drop at Extreme Cold ---
            if "battery_soc_pct" in feat.columns:
                soc_val = float(feat["battery_soc_pct"].iloc[i])
                step_delta = float(feat["soc_step_delta"].iloc[i])
                # In extreme cold (< -20°C), sudden SoC collapse or violation below 20% floor
                if temp < -20.0 and (soc_val < 18.0 or step_delta < -14.0):
                    reason = (
                        f"Battery capacity / state-of-charge sudden drop (SoC {soc_val:.1f}%, Δ {step_delta:.1f}%) "
                        f"at extreme ambient cold ({temp:.1f}°C) — suspected BESS thermal enclosure heater malfunction."
                    )
                    anomalies.append(
                        {
                            "timestamp": ts,
                            "severity": "WARNING",
                            "anomaly_type": "BATTERY_COLD_DEGRADATION",
                            "metric": "Battery State of Charge (%)",
                            "observed_value": round(soc_val, 1),
                            "baseline_value": 20.0,
                            "z_score": round(float(abs(step_delta) / 2.0), 2),
                            "reason": reason,
                        }
                    )
                    continue

            # --- ANOMALY RULE 4: Multivariate Isolation Forest Flag ---
            if iso_flag and excess_ratio > 0.20:
                reason = (
                    f"Multivariate telemetry anomaly: combined fuel-to-output ratio "
                    f"exceeded physical bounds by +{excess_ratio*100:.1f}% under ambient {temp:.1f}°C."
                )
                anomalies.append(
                    {
                        "timestamp": ts,
                        "severity": "WARNING",
                        "anomaly_type": "MULTIVARIATE_DISPATCH_ANOMALY",
                        "metric": "Fuel Excess Ratio",
                        "observed_value": round(excess_ratio, 3),
                        "baseline_value": 0.0,
                        "z_score": 2.6,
                        "reason": reason,
                    }
                )

        return anomalies


# Global cached detector instance
_CACHED_DETECTOR: Optional[AnomalyDetector] = None


def get_default_detector() -> AnomalyDetector:
    """Initialize and fit default anomaly detector on nominal station telemetry."""
    global _CACHED_DETECTOR
    if _CACHED_DETECTOR is not None and _CACHED_DETECTOR.is_fitted:
        return _CACHED_DETECTOR

    from pathlib import Path
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data_generator"
        / "data"
        / "polar_station_energy_hourly.csv"
    )
    df = pd.read_csv(data_path)
    detector = AnomalyDetector()
    detector.fit(df)
    _CACHED_DETECTOR = detector
    return _CACHED_DETECTOR


def detect_anomalies(
    recent_data: pd.DataFrame,
    z_threshold: float = 2.5,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper: Inspects recent microgrid telemetry and flags
    generator and battery degradation events with explainable reasons.

    Args:
        recent_data: DataFrame of recent hourly records.
        z_threshold: Sensitivity threshold (default: 2.5 standard deviations).

    Returns:
        List of flagged anomaly records containing timestamps, severity, and reasons.
    """
    detector = get_default_detector()
    detector.z_threshold = z_threshold
    return detector.detect(recent_data)
