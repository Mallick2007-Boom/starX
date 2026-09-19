"""
BOREAS Antarctic Digital Twin Engine (Module: digital_twin.py)
Maintains high-fidelity synchronized representations of environmental physics,
energy microgrid generation/storage, and station habitat life-support telemetry.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .optimizer import MicrogridState, OptimizerConfig, optimize_hour
from .risk_engine import RiskAssessment, evaluate_mission_risk


@dataclass
class EnvironmentState:
    outdoor_temp_c: float
    wind_speed_mps: float
    relative_humidity_pct: float
    solar_elevation_deg: float
    daylight_hours: float
    wind_icing_factor: float
    weather_condition: str
    storm_status: str


@dataclass
class EnergyState:
    solar_generation_kw: float
    wind_generation_raw_kw: float
    effective_wind_generation_kw: float
    renewables_total_kw: float
    battery_soc_pct: float
    battery_power_kw: float
    battery_kwh: float
    battery_mode: str
    diesel_generation_kw: float
    diesel_fuel_reserve_liters: float
    hourly_fuel_burn_liters: float
    total_station_demand_kw: float
    critical_demand_kw: float
    flexible_demand_kw: float
    curtailed_flexible_kw: float
    critical_load_met_pct: float


@dataclass
class StationState:
    crew_population: int
    active_equipment: Dict[str, bool]
    heating_thermal_deficit_kw: float
    equipment_health: Dict[str, str]
    operating_mode: str
    risk: RiskAssessment


class DigitalTwin:
    """
    Antarctic Research Facility Virtual Twin.
    Synchronizes physical environmental sensors with microgrid power electronics
    and life-support habitat telemetry.
    """

    def __init__(self, data_df: Optional[pd.DataFrame] = None):
        self.df = data_df
        self.current_step = 4716  # Default: mid-winter blizzard index
        self.load_switches = {
            "drill": True,
            "lidar": True,
            "skidoo": True,
            "laundry": True,
            "sauna": True,
        }
        self.generator_1_online = True
        self.generator_2_online = True
        self.battery_online = True
        self.cold_battery_mode = False
        self.survival_mode = False

    def set_data(self, df: pd.DataFrame):
        self.df = df

    def get_snapshot(
        self,
        step_index: Optional[int] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extracts a comprehensive digital twin snapshot at step_index,
        incorporating any physical or operational overrides.
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("Digital Twin has no telemetry dataset loaded.")

        idx = step_index if step_index is not None and 0 <= step_index < len(self.df) else self.current_step
        idx = max(0, min(idx, len(self.df) - 1))
        row = self.df.iloc[idx].to_dict()

        # Extract base physical telemetry
        temp_c = float(row.get("outdoor_temp_c", -35.0))
        wind_mps = float(row.get("wind_speed_mps", 8.0))
        humidity = float(row.get("relative_humidity_pct", 75.0))
        sun_elev = float(row.get("sun_elevation_deg", 0.0))
        daylight = float(row.get("daylight_hours", 0.0))
        icing = float(row.get("wind_icing_factor", 1.0))
        crew = int(row.get("crew_headcount", 35))

        p_crit = float(row.get("load_critical_kw", 70.0))
        p_flex = float(row.get("load_flexible_kw", 35.0))
        p_solar = float(row.get("solar_available_kw", 0.0))
        raw_wind = float(row.get("wind_available_kw", wind_mps * 12.0))
        p_wind_eff = float(row.get("wind_effective_kw", raw_wind * icing))
        fuel_reserve = float(row.get("fuel_reserve_liters", row.get("diesel_fuel_reserve_liters", 240000.0)))
        bat_kwh = float(row.get("battery_kwh", 180.0))

        # Apply overrides if passed
        ov = overrides or {}
        if "outdoor_temp_c" in ov:
            temp_c = float(ov["outdoor_temp_c"])
        if "wind_speed_mps" in ov:
            wind_mps = float(ov["wind_speed_mps"])
        if "wind_icing_factor" in ov:
            icing = float(ov["wind_icing_factor"])
        if "generator_1_online" in ov:
            self.generator_1_online = bool(ov["generator_1_online"])
        if "generator_2_online" in ov:
            self.generator_2_online = bool(ov["generator_2_online"])
        if "battery_online" in ov:
            self.battery_online = bool(ov["battery_online"])
        if "cold_battery_mode" in ov:
            self.cold_battery_mode = bool(ov["cold_battery_mode"])
        if "survival_mode" in ov:
            self.survival_mode = bool(ov["survival_mode"])
        if "load_switches" in ov:
            self.load_switches.update(ov["load_switches"])

        p_wind_eff = raw_wind * icing

        # Weather classification
        if wind_mps >= 22.0:
            storm_status = "BLIZZARD_ACTIVE"
            weather_condition = "Katabatic Blizzard"
        elif wind_mps >= 15.0 or icing <= 0.40:
            storm_status = "BLIZZARD_WATCH"
            weather_condition = "Freezing Fog & High Gale"
        elif daylight == 0.0:
            storm_status = "CLEAR"
            weather_condition = "Polar Night (Total Darkness)"
        elif sun_elev > 10.0:
            storm_status = "CLEAR"
            weather_condition = "Continuous Midnight Sun"
        else:
            storm_status = "CLEAR"
            weather_condition = "Cold Antarctic Overcast"

        # Days until resupply
        hours_remaining = max(1, len(self.df) - idx)
        days_until_resupply = max(1.0, hours_remaining / 24.0)

        # Solve live dispatch with current digital twin state
        cfg = OptimizerConfig(
            generator_1_available=self.generator_1_online,
            generator_2_available=self.generator_2_online,
            battery_available=self.battery_online,
            cold_battery_protection_active=self.cold_battery_mode,
            survival_mode_active=self.survival_mode,
            operator_load_switches=self.load_switches,
        )
        state = MicrogridState(
            current_battery_kwh=bat_kwh if self.battery_online else 0.0,
            current_fuel_reserve_liters=fuel_reserve,
            days_until_resupply=days_until_resupply,
            current_temp_c=temp_c,
        )
        forecast = {
            "load_critical_kw": p_crit,
            "load_flexible_kw": p_flex,
            "renewables_kw": p_solar + p_wind_eff,
            "future_renewables_mean_6h": p_wind_eff,
        }
        dispatch = optimize_hour(state, forecast, cfg)

        # Evaluate risk
        burn_recent = dispatch["fuel_consumed_liters"]
        daily_burn = max(24.0, burn_recent * 24.0)
        risk = evaluate_mission_risk(
            battery_soc_pct=dispatch["battery_soc_pct"],
            predicted_min_soc_24h=dispatch["battery_soc_pct"],
            fuel_reserve_liters=dispatch["fuel_reserve_remaining_liters"],
            daily_burn_liters=daily_burn,
            days_until_resupply=days_until_resupply,
            critical_load_kw=p_crit,
            renewable_kw=p_solar + p_wind_eff,
            wind_icing_factor=icing,
            outdoor_temp_c=temp_c,
            generator_1_online=self.generator_1_online,
            generator_2_online=self.generator_2_online,
            battery_online=self.battery_online,
            severe_storm_active=(storm_status == "BLIZZARD_ACTIVE"),
        )

        return {
            "timestamp": str(row.get("timestamp")),
            "step_index": idx,
            "total_steps": len(self.df),
            "environment": {
                "outdoor_temp_c": round(temp_c, 1),
                "wind_speed_mps": round(wind_mps, 1),
                "relative_humidity_pct": round(humidity, 1),
                "solar_elevation_deg": round(sun_elev, 1),
                "daylight_hours": round(daylight, 1),
                "wind_icing_factor": round(icing, 3),
                "weather_condition": weather_condition,
                "storm_status": storm_status,
            },
            "energy": {
                "solar_generation_kw": round(p_solar, 1),
                "wind_generation_raw_kw": round(raw_wind, 1),
                "wind_effective_kw": round(p_wind_eff, 1),
                "renewables_total_kw": round(p_solar + p_wind_eff, 1),
                "battery_soc_pct": round(dispatch["battery_soc_pct"], 1),
                "battery_power_kw": round(dispatch["battery_power_kw"], 1),
                "battery_kwh": round(dispatch["battery_kwh"], 1),
                "battery_mode": dispatch["battery_charge_or_discharge"],
                "battery_online": self.battery_online,
                "cold_battery_protection_active": dispatch["cold_battery_protection_active"],
                "diesel_generation_kw": round(dispatch["kw_from_diesel"], 1),
                "diesel_fuel_reserve_liters": round(dispatch["fuel_reserve_remaining_liters"], 1),
                "hourly_fuel_burn_liters": round(dispatch["fuel_consumed_liters"], 2),
                "total_station_demand_kw": round(p_crit + p_flex - dispatch["curtailed_flexible_kw"], 1),
                "critical_demand_kw": round(p_crit, 1),
                "flexible_demand_kw": round(p_flex - dispatch["curtailed_flexible_kw"], 1),
                "curtailed_flexible_kw": round(dispatch["curtailed_flexible_kw"], 1),
                "critical_load_met_pct": dispatch["critical_load_met_pct"],
                "diesel_generator_1_online": self.generator_1_online,
                "diesel_generator_2_online": self.generator_2_online,
            },
            "station": {
                "crew_population": crew,
                "active_load_switches": dict(self.load_switches),
                "heating_thermal_deficit_kw": round(p_crit * 0.70, 1),
                "equipment_health": {
                    "generator_1": "ONLINE (Primary 100kW)" if self.generator_1_online else "OFFLINE / FAILED",
                    "generator_2": "ONLINE (Backup 100kW)" if self.generator_2_online else "OFFLINE / FAILED",
                    "battery_bess": "HEALTHY (300 kWh LiFePO4)" if self.battery_online else "OFFLINE / INVERTER TRIP",
                    "wind_turbines": f"OPERATIONAL (Icing: {int(icing*100)}%)",
                    "solar_pv": "OPERATIONAL" if p_solar > 0 or daylight > 0 else "POLAR NIGHT ZERO CLAMP",
                },
                "operating_mode": dispatch["operating_mode"],
                "days_until_summer_resupply": round(days_until_resupply, 1),
            },
            "risk": {
                "level": risk.risk_level,
                "score": risk.risk_score,
                "primary_causes": risk.primary_causes,
                "recommended_action": risk.recommended_action,
                "autonomy_days": risk.autonomy_days,
                "resupply_buffer_margin_pct": risk.resupply_buffer_margin_pct,
            },
            "decision": {
                "action": "DISPATCH BATTERY" if dispatch["battery_power_kw"] > 0 else ("CHARGE BATTERY" if dispatch["battery_power_kw"] < 0 else ("DISPATCH DIESEL" if dispatch["kw_from_diesel"] > 0 else "RENEWABLE DIRECT")),
                "reason": dispatch["reason"],
            },
        }
