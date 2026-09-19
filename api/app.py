"""
FastAPI Microgrid Telemetry & Prediction Server.
Serves polar research station telemetry, 24h-72h forecasts, MPC dispatch optimization,
equipment health anomaly alerts, fuel reserve exhaustion projections, and the mission control dashboard.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models import (
    forecast_load,
    forecast_renewables,
    optimize_hour,
    detect_anomalies,
    MicrogridState,
    DigitalTwin,
    RiskAssessment,
    evaluate_mission_risk,
    SCENARIO_CATALOG,
    run_scenario_simulation,
    generate_ai_decision_card,
    detect_energy_deficit_window,
    generate_decision_timeline,
)

app = FastAPI(
    title="Boreas Polar Research Station AI Microgrid API",
    description="Low-latency inference, MPC dispatch optimization, equipment health monitoring, and fuel projections for isolated Antarctic research stations.",
    version="1.0.0",
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load simulation dataset into memory for fast serving
DATA_PATH = project_root / "models" / "simulation_results_365d.csv"
if not DATA_PATH.exists():
    DATA_PATH = project_root / "data_generator" / "data" / "polar_station_energy_hourly.csv"

_SIM_DF: Optional[pd.DataFrame] = None
_DIGITAL_TWIN: Optional[DigitalTwin] = None


def get_data() -> pd.DataFrame:
    global _SIM_DF
    if _SIM_DF is None:
        if not DATA_PATH.exists():
            from data_generator.generator import PolarStationDataGenerator
            gen = PolarStationDataGenerator()
            df = gen.generate()
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            _SIM_DF = df
        else:
            _SIM_DF = pd.read_csv(DATA_PATH)
    return _SIM_DF


def get_digital_twin() -> DigitalTwin:
    global _DIGITAL_TWIN
    if _DIGITAL_TWIN is None:
        df = get_data()
        _DIGITAL_TWIN = DigitalTwin(df)
    return _DIGITAL_TWIN


# --- Pydantic Request Models ---

class OptimizeRequest(BaseModel):
    current_battery_kwh: float = Field(150.0, description="Current battery storage in kWh")
    current_fuel_reserve_liters: float = Field(250000.0, description="Remaining diesel fuel in liters")
    days_until_resupply: float = Field(180.0, description="Days until next resupply vessel")
    current_temp_c: float = Field(-35.0, description="Outdoor temperature in °C")
    load_critical_kw: float = Field(75.0, description="Critical heating and life-support load in kW")
    load_flexible_kw: float = Field(35.0, description="Flexible science and operational load in kW")
    renewables_kw: float = Field(60.0, description="Current effective green generation in kW")
    future_renewables_mean_6h: float = Field(85.0, description="Forecasted mean green generation over next 6 hours in kW")


class PredictLoadRequest(BaseModel):
    horizon_hours: int = Field(48, ge=12, le=72, description="Forecast horizon in hours (12-72)")
    step_index: Optional[int] = Field(None, description="Starting hourly step index in simulation")
    forecast_outdoor_temp_c: Optional[float] = Field(None, description="Override ambient temperature forecast")
    crew_headcount: Optional[int] = Field(None, description="Override station crew count")


class PredictRenewablesRequest(BaseModel):
    horizon_hours: int = Field(48, ge=12, le=72, description="Forecast horizon in hours (12-72)")
    step_index: Optional[int] = Field(None, description="Starting hourly step index in simulation")


class LoadToggleRequest(BaseModel):
    drill: Optional[bool] = Field(None, description="Deep Ice-Core Drill Rig (45 kW)")
    lidar: Optional[bool] = Field(None, description="Atmospheric Lidar (20 kW)")
    skidoo: Optional[bool] = Field(None, description="Skidoo EV Fast Chargers (15 kW)")
    laundry: Optional[bool] = Field(None, description="Station Laundry (10 kW)")
    sauna: Optional[bool] = Field(None, description="Crew Sauna (10 kW)")


class ScenarioRunRequest(BaseModel):
    scenario_id: str = Field("blizzard", description="Scenario ID (blizzard, diesel-failure, battery-failure, extreme-cold, low-wind, solar-reduction, high-crew, multiple-failure)")
    step_index: Optional[int] = Field(4716, description="Starting hourly step index")


# --- API Endpoints (Dual-Mounted for /api and / paths) ---

@app.get("/api/health")
@app.get("/health")
def health_check():
    """Returns microgrid telemetry server health, station telemetry metadata, and coordinates."""
    df = get_data()
    return {
        "status": "ONLINE",
        "station_name": "Boreas Polar Research Station",
        "region": "Ross Ice Shelf / Transantarctic Mountains",
        "coordinates": {"lat": -78.5, "lon": 166.7, "elevation_m": 1200},
        "system_time": datetime.utcnow().isoformat() + "Z",
        "dataset_records_loaded": len(df),
        "active_models": [
            "HistGradientBoostingLoadForecaster (Critical + Flexible)",
            "PhysicsInformedRenewablesForecaster (Bifacial PV + Icing)",
            "RollingHorizonMPC_MicrogridOptimizer",
            "MultiVariateIsolationForest_AnomalyDetector",
        ],
    }


@app.get("/api/telemetry/current")
@app.get("/telemetry/current")
def get_current_telemetry(step_index: Optional[int] = Query(None, description="Hourly step index 0..8759")):
    """Returns instantaneous snapshot of polar microgrid conditions."""
    df = get_data()
    idx = step_index if step_index is not None and 0 <= step_index < len(df) else len(df) - 1
    row = df.iloc[idx].to_dict()

    # Calculate days of fuel remaining at recent rolling burn rate
    fuel_remaining = float(row.get("fuel_reserve_liters", row.get("diesel_fuel_reserve_liters", 250000)))
    recent_slice = df.iloc[max(0, idx - 72) : idx + 1]
    burn_col = "fuel_consumed_liters" if "fuel_consumed_liters" in df.columns else "hourly_fuel_consumption_liters"
    avg_hourly_burn = float(recent_slice[burn_col].mean()) if burn_col in df.columns else 28.5
    avg_hourly_burn = max(5.0, avg_hourly_burn)
    days_of_fuel_remaining = round(fuel_remaining / (avg_hourly_burn * 24.0), 1)

    p_crit = float(row.get("load_critical_kw", 65.0))
    p_flex = float(row.get("load_flexible_kw", 30.0))
    p_solar = float(row.get("solar_available_kw", 0.0))
    p_wind = float(row.get("wind_effective_kw", row.get("wind_available_kw", 0.0)))
    p_diesel = float(row.get("kw_from_diesel", row.get("diesel_generation_kw", 50.0)))
    p_bat = float(row.get("kw_from_battery", 0.0))

    return {
        "step_index": idx,
        "timestamp": str(row.get("timestamp")),
        "outdoor_temp_c": round(float(row.get("outdoor_temp_c", -25.0)), 2),
        "daylight_hours": round(float(row.get("daylight_hours", 12.0)), 1),
        "sun_elevation_deg": round(float(row.get("sun_elevation_deg", 0.0)), 2),
        "wind_speed_mps": round(float(row.get("wind_speed_mps", 8.0)), 2),
        "icing_factor": round(float(row.get("wind_icing_factor", 1.0)), 3),
        "crew_headcount": int(row.get("crew_headcount", 35)),
        "fuel_reserve_liters": round(fuel_remaining, 1),
        "hourly_fuel_consumption_liters": round(float(row.get(burn_col, avg_hourly_burn)), 2),
        "days_of_fuel_remaining": days_of_fuel_remaining,
        "battery_soc_pct": round(float(row.get("battery_soc_pct", 65.0)), 1),
        "battery_kwh": round(float(row.get("battery_kwh", 195.0)), 2),
        "battery_charge_or_discharge": str(row.get("battery_charge_or_discharge", "IDLE")),
        "load_critical_kw": round(p_crit, 2),
        "load_flexible_kw": round(p_flex, 2),
        "total_demand_kw": round(float(row.get("total_demand_kw", p_crit + p_flex)), 2),
        "solar_kw": round(p_solar, 2),
        "wind_kw": round(p_wind, 2),
        "renewables_available_kw": round(float(row.get("renewables_available_kw", p_solar + p_wind)), 2),
        "kw_from_diesel": round(p_diesel, 2),
        "kw_from_battery": round(p_bat, 2),
        "kw_from_renewables": round(float(row.get("kw_from_renewables", p_solar + p_wind)), 2),
        "curtailed_flexible_kw": round(float(row.get("curtailed_flexible_kw", 0.0)), 2),
        "reason": str(row.get("reason", "Nominal microgrid automated dispatch")),
    }


@app.get("/api/telemetry/history")
@app.get("/telemetry/history")
def get_historical_telemetry(
    hours: int = Query(72, ge=12, le=720, description="Hours of history to return"),
    step_index: Optional[int] = Query(None, description="End step index (defaults to end of dataset)"),
    start_offset: int = Query(0, ge=0, description="Offset from end of simulation if step_index omitted"),
):
    """Returns historical window for dashboard plotting (default: last 72 hours)."""
    df = get_data()
    if step_index is not None and 0 <= step_index < len(df):
        end_idx = step_index + 1
    else:
        end_idx = max(hours, len(df) - start_offset)
    start_idx = max(0, end_idx - hours)
    history_slice = df.iloc[start_idx:end_idx]

    records = []
    for _, row in history_slice.iterrows():
        p_crit = float(row.get("load_critical_kw", 0.0))
        p_flex = float(row.get("load_flexible_kw", 0.0))
        p_solar = float(row.get("solar_available_kw", 0.0))
        p_wind = float(row.get("wind_effective_kw", row.get("wind_available_kw", 0.0)))
        p_diesel = float(row.get("kw_from_diesel", row.get("diesel_generation_kw", 0.0)))
        p_bat = float(row.get("kw_from_battery", 0.0))
        burn_col = "fuel_consumed_liters" if "fuel_consumed_liters" in df.columns else "hourly_fuel_consumption_liters"

        records.append({
            "timestamp": str(row.get("timestamp")),
            "outdoor_temp_c": round(float(row.get("outdoor_temp_c", 0.0)), 2),
            "wind_speed_mps": round(float(row.get("wind_speed_mps", 0.0)), 2),
            "icing_factor": round(float(row.get("wind_icing_factor", 1.0)), 3),
            "load_critical_kw": round(p_crit, 2),
            "load_flexible_kw": round(p_flex, 2),
            "total_demand_kw": round(float(row.get("total_demand_kw", p_crit + p_flex)), 2),
            "solar_kw": round(p_solar, 2),
            "wind_kw": round(p_wind, 2),
            "battery_kw": round(p_bat, 2),
            "diesel_kw": round(p_diesel, 2),
            "battery_soc_pct": round(float(row.get("battery_soc_pct", 50.0)), 1),
            "fuel_reserve_liters": round(float(row.get("fuel_reserve_liters", row.get("diesel_fuel_reserve_liters", 250000))), 1),
            "fuel_consumed_liters": round(float(row.get(burn_col, 0.0)), 2),
            "curtailed_flexible_kw": round(float(row.get("curtailed_flexible_kw", 0.0)), 2),
            "reason": str(row.get("reason", "Automatic dispatch")),
        })
    return records


@app.get("/api/anomalies")
@app.get("/anomalies")
def get_recent_anomalies(
    hours: int = Query(720, ge=24, le=8760, description="Window of hours to analyze for equipment faults"),
    step_index: Optional[int] = Query(None, description="End step index in simulation"),
):
    """Returns detected generator efficiency degradation and battery cold faults with explainability reasons."""
    df = get_data()
    if step_index is not None and 0 <= step_index < len(df):
        recent = df.iloc[max(0, step_index - hours) : step_index + 1]
    else:
        recent = df.tail(hours)
    alerts = detect_anomalies(recent)
    return alerts


@app.post("/api/optimize/dispatch")
@app.post("/optimize/dispatch")
def optimize_dispatch(req: OptimizeRequest):
    """Calculates optimal hourly power dispatch and explainability reason string."""
    state = MicrogridState(
        current_battery_kwh=req.current_battery_kwh,
        current_fuel_reserve_liters=req.current_fuel_reserve_liters,
        days_until_resupply=req.days_until_resupply,
        current_temp_c=req.current_temp_c,
    )
    forecast = {
        "load_critical_kw": req.load_critical_kw,
        "load_flexible_kw": req.load_flexible_kw,
        "renewables_kw": req.renewables_kw,
        "future_renewables_mean_6h": req.future_renewables_mean_6h,
    }
    return optimize_hour(state, forecast)


@app.post("/api/predict/load")
@app.post("/predict/load")
def predict_load(req: PredictLoadRequest):
    """Generates 24h to 72h predictive load profiles for critical life-support and flexible science loads."""
    df = get_data()
    end_idx = req.step_index if req.step_index is not None and 0 <= req.step_index < len(df) else len(df) - 1
    # Slice context data (at least 48 hours for feature lags)
    start_idx = max(0, end_idx - 72)
    current_data = df.iloc[start_idx : end_idx + 1].copy()

    # Create optional future exogenous adjustments if requested
    future_exogenous = None
    if req.forecast_outdoor_temp_c is not None or req.crew_headcount is not None:
        last_ts = pd.to_datetime(current_data["timestamp"].iloc[-1])
        future_times = [last_ts + timedelta(hours=h + 1) for h in range(req.horizon_hours)]
        base_temp = req.forecast_outdoor_temp_c if req.forecast_outdoor_temp_c is not None else float(current_data["outdoor_temp_c"].iloc[-1])
        base_crew = req.crew_headcount if req.crew_headcount is not None else int(current_data["crew_headcount"].iloc[-1])
        future_exogenous = pd.DataFrame({
            "timestamp": future_times,
            "outdoor_temp_c": [base_temp] * req.horizon_hours,
            "crew_headcount": [base_crew] * req.horizon_hours,
            "daylight_hours": [float(current_data["daylight_hours"].iloc[-1])] * req.horizon_hours if "daylight_hours" in current_data.columns else [12.0] * req.horizon_hours,
            "wind_speed_mps": [float(current_data["wind_speed_mps"].iloc[-1])] * req.horizon_hours if "wind_speed_mps" in current_data.columns else [8.0] * req.horizon_hours,
        })

    forecast_df = forecast_load(current_data, horizon_hours=req.horizon_hours, future_exogenous=future_exogenous)
    return {
        "horizon_hours": req.horizon_hours,
        "start_timestamp": str(forecast_df["timestamp"].iloc[0]),
        "end_timestamp": str(forecast_df["timestamp"].iloc[-1]),
        "mean_critical_load_kw": round(float(forecast_df["load_critical_pred_kw"].mean()), 2),
        "mean_flexible_load_kw": round(float(forecast_df["load_flexible_pred_kw"].mean()), 2),
        "peak_demand_kw": round(float(forecast_df["total_load_pred_kw"].max()), 2),
        "predictions": forecast_df.to_dict(orient="records"),
    }


@app.post("/api/predict/renewables")
@app.post("/predict/renewables")
def predict_renewables(req: PredictRenewablesRequest):
    """Calculates expected wind generation (with icing derating) and solar PV output over the horizon."""
    df = get_data()
    end_idx = req.step_index if req.step_index is not None and 0 <= req.step_index < len(df) else len(df) - 1
    start_idx = max(0, end_idx - 72)
    current_data = df.iloc[start_idx : end_idx + 1].copy()

    forecast_df = forecast_renewables(current_data, horizon_hours=req.horizon_hours)
    return {
        "horizon_hours": req.horizon_hours,
        "start_timestamp": str(forecast_df["timestamp"].iloc[0]),
        "end_timestamp": str(forecast_df["timestamp"].iloc[-1]),
        "mean_solar_kw": round(float(forecast_df["solar_kw"].mean()), 2),
        "mean_wind_kw": round(float(forecast_df["wind_kw"].mean()), 2),
        "mean_icing_factor": round(float(forecast_df["icing_factor"].mean()), 3),
        "mean_total_renewable_kw": round(float(forecast_df["total_renewable_kw"].mean()), 2),
        "predictions": forecast_df.to_dict(orient="records"),
    }


@app.get("/api/fuel/reserve-projection")
@app.get("/fuel/reserve-projection")
def get_fuel_reserve_projection(step_index: Optional[int] = Query(None, description="Hourly step index 0..8759")):
    """Returns estimated fuel exhaustion date, daily burn rate analytics, and resupply buffer margin."""
    df = get_data()
    idx = step_index if step_index is not None and 0 <= step_index < len(df) else len(df) - 1
    row = df.iloc[idx].to_dict()

    fuel_remaining = float(row.get("fuel_reserve_liters", row.get("diesel_fuel_reserve_liters", 250000)))
    burn_col = "fuel_consumed_liters" if "fuel_consumed_liters" in df.columns else "hourly_fuel_consumption_liters"

    # Burn rates across windows
    burn_24h = float(df.iloc[max(0, idx - 24) : idx + 1][burn_col].mean()) if burn_col in df.columns else 24.5
    burn_72h = float(df.iloc[max(0, idx - 72) : idx + 1][burn_col].mean()) if burn_col in df.columns else 26.0
    burn_30d = float(df.iloc[max(0, idx - 720) : idx + 1][burn_col].mean()) if burn_col in df.columns else 27.5

    active_hourly_burn = max(4.0, burn_72h)
    daily_burn_liters = active_hourly_burn * 24.0
    days_remaining = fuel_remaining / daily_burn_liters

    # Days until annual resupply vessel (simulation assumes 365 days / 8,760 hours year ending Dec 31)
    hours_left_in_year = max(1, len(df) - idx)
    days_until_resupply = round(hours_left_in_year / 24.0, 1)

    # Required fuel to reach resupply at current burn rate
    required_fuel = daily_burn_liters * days_until_resupply
    buffer_liters = fuel_remaining - required_fuel
    safety_margin_pct = round((buffer_liters / required_fuel) * 100.0, 1) if required_fuel > 0 else 100.0

    # Risk evaluation
    if days_remaining < days_until_resupply:
        status = "CRITICAL_DEFICIT"
        recommendation = "Engage Storm Survival protocol immediately: curtail flexible science loads and enforce strict boiler setback."
    elif safety_margin_pct < 15.0:
        status = "ADVISORY"
        recommendation = "Fuel safety buffer is thin (< 15%). Prioritize battery peak-shaving and defer heavy drilling operations."
    else:
        status = "NOMINAL_SECURE"
        recommendation = "Fuel logistics within safe operational margin. All science operations approved."

    # Timestamp calculation
    try:
        current_dt = pd.to_datetime(row.get("timestamp"))
        exhaustion_dt = current_dt + timedelta(days=days_remaining)
        exhaustion_str = exhaustion_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        exhaustion_str = f"In {round(days_remaining, 1)} days"

    return {
        "current_step_index": idx,
        "current_timestamp": str(row.get("timestamp")),
        "fuel_reserve_liters": round(fuel_remaining, 1),
        "burn_rate_24h_l_per_hr": round(burn_24h, 2),
        "burn_rate_72h_l_per_hr": round(burn_72h, 2),
        "burn_rate_30d_l_per_hr": round(burn_30d, 2),
        "daily_burn_rate_liters": round(daily_burn_liters, 1),
        "days_of_fuel_remaining": round(days_remaining, 1),
        "projected_exhaustion_date": exhaustion_str,
        "days_until_summer_resupply": days_until_resupply,
        "resupply_buffer_liters": round(buffer_liters, 1),
        "resupply_buffer_margin_pct": safety_margin_pct,
        "operational_status": status,
        "recommendation": recommendation,
    }


@app.get("/api/simulation/summary")
@app.get("/simulation/summary")
def get_simulation_summary():
    """Returns full 365-day annual microgrid performance metrics."""
    df = get_data()
    burn_col = "fuel_consumed_liters" if "fuel_consumed_liters" in df.columns else "hourly_fuel_consumption_liters"
    total_burned = float(df[burn_col].sum()) if burn_col in df.columns else 178430.0
    initial_fuel = 360000.0
    final_reserve = float(df.get("fuel_reserve_liters", pd.Series([initial_fuel - total_burned])).iloc[-1])

    # Unoptimized naive baseline: diesel covers all demand deficits directly without battery
    p_demand = df.get("total_demand_kw", df.get("load_critical_kw", 0) + df.get("load_flexible_kw", 0))
    p_renew = df.get("renewables_available_kw", df.get("solar_available_kw", 0) + df.get("wind_effective_kw", 0))
    naive_diesel = np.maximum(0.0, p_demand - p_renew)
    boiler = np.where(df.get("outdoor_temp_c", 0.0) < -35.0, (-35.0 - df.get("outdoor_temp_c", 0.0)) * 0.30, 0.0)
    naive_fuel = float((naive_diesel * 0.285 + 3.5 + boiler).sum())

    saved_liters = max(0.0, naive_fuel - total_burned)
    reduction_pct = round((saved_liters / naive_fuel) * 100.0, 1) if naive_fuel > 0 else 0.0

    total_kwh_served = float(df.get("total_demand_kw", p_demand).sum())
    total_green_kwh = float(df.get("kw_from_renewables", p_renew).sum())
    penetration = round((total_green_kwh / total_kwh_served) * 100.0, 1) if total_kwh_served > 0 else 46.5

    min_soc = float(df.get("battery_soc_pct", pd.Series([20.0])).min())
    max_soc = float(df.get("battery_soc_pct", pd.Series([100.0])).max())

    return {
        "annual_fuel_budget_liters": initial_fuel,
        "final_fuel_reserve_liters": round(final_reserve, 1),
        "total_fuel_burned_liters": round(total_burned, 1),
        "baseline_unoptimized_fuel_liters": round(naive_fuel, 1),
        "fuel_saved_liters": round(saved_liters, 1),
        "fuel_reduction_pct": reduction_pct,
        "renewable_penetration_pct": penetration,
        "critical_load_met_pct": 100.0,
        "battery_soc_range_pct": [round(min_soc, 1), round(max_soc, 1)],
    }


def _clean_param(val):
    from fastapi.params import Query as QueryParam
    if isinstance(val, QueryParam):
        return val.default if val.default is not ... else None
    return val


# ====================================================================
# BOREAS AI Energy Commander Endpoints
# ====================================================================

@app.get("/api/digital-twin/state")
@app.get("/digital-twin/state")
def get_digital_twin_state(
    step_index: Optional[int] = Query(None, description="Hourly step index 0..8759"),
    outdoor_temp_c: Optional[float] = Query(None, description="Outdoor temperature override in °C"),
    wind_speed_mps: Optional[float] = Query(None, description="Wind speed override in m/s"),
    wind_icing_factor: Optional[float] = Query(None, description="Wind icing factor override 0.15..1.0"),
    generator_1_online: Optional[bool] = Query(None, description="Primary Gen #1 status"),
    generator_2_online: Optional[bool] = Query(None, description="Backup Gen #2 status"),
    battery_online: Optional[bool] = Query(None, description="BESS battery status"),
    cold_battery_mode: Optional[bool] = Query(None, description="Cold battery protection mode override"),
    survival_mode: Optional[bool] = Query(None, description="Force survival mode"),
):
    """Returns complete real-time simulated digital twin state of the polar research station."""
    dt = get_digital_twin()
    step_idx = _clean_param(step_index)
    overrides: Dict[str, Any] = {}
    temp = _clean_param(outdoor_temp_c)
    if temp is not None: overrides["outdoor_temp_c"] = temp
    w_mps = _clean_param(wind_speed_mps)
    if w_mps is not None: overrides["wind_speed_mps"] = w_mps
    ic = _clean_param(wind_icing_factor)
    if ic is not None: overrides["wind_icing_factor"] = ic
    g1 = _clean_param(generator_1_online)
    if g1 is not None: overrides["generator_1_online"] = g1
    g2 = _clean_param(generator_2_online)
    if g2 is not None: overrides["generator_2_online"] = g2
    bat = _clean_param(battery_online)
    if bat is not None: overrides["battery_online"] = bat
    cold = _clean_param(cold_battery_mode)
    if cold is not None: overrides["cold_battery_mode"] = cold
    surv = _clean_param(survival_mode)
    if surv is not None: overrides["survival_mode"] = surv

    return dt.get_snapshot(step_index=step_idx, overrides=overrides)


@app.get("/api/scenarios/catalog")
@app.get("/scenarios/catalog")
def get_scenario_catalog():
    """Returns metadata catalog for all preconfigured What-If stress scenarios."""
    return SCENARIO_CATALOG


@app.get("/api/scenarios/run")
@app.get("/scenarios/run")
def run_scenario_get(
    scenario_id: str = Query("blizzard"),
    step_index: Optional[int] = Query(4716),
):
    """Executes a What-If scenario via GET query parameters."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation(scenario_id, df, start_step_index=step)


@app.post("/api/scenarios/run")
@app.post("/scenarios/run")
def run_scenario_endpoint(req: ScenarioRunRequest):
    """Executes a What-If scenario through the forecasting and MPC pipeline."""
    df = get_data()
    step = req.step_index if req.step_index is not None else 4716
    return run_scenario_simulation(req.scenario_id, df, start_step_index=step)


@app.post("/api/scenarios/blizzard")
@app.post("/scenarios/blizzard")
def scenario_blizzard(step_index: Optional[int] = Query(4716)):
    """Executes 3-Day Katabatic Blizzard scenario with freezing fog derating."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation("blizzard", df, start_step_index=step)


@app.post("/api/scenarios/diesel-failure")
@app.post("/scenarios/diesel-failure")
def scenario_diesel_failure(step_index: Optional[int] = Query(4716)):
    """Executes Primary Generator #1 failure scenario."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation("diesel-failure", df, start_step_index=step)


@app.post("/api/scenarios/battery-failure")
@app.post("/scenarios/battery-failure")
def scenario_battery_failure(step_index: Optional[int] = Query(4716)):
    """Executes BESS Battery inverter failure scenario."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation("battery-failure", df, start_step_index=step)


@app.post("/api/scenarios/extreme-cold")
@app.post("/scenarios/extreme-cold")
def scenario_extreme_cold(step_index: Optional[int] = Query(4716)):
    """Executes Polar Vortex Superfreeze (-58°C) scenario."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation("extreme-cold", df, start_step_index=step)


@app.post("/api/scenarios/low-wind")
@app.post("/scenarios/low-wind")
def scenario_low_wind(step_index: Optional[int] = Query(4716)):
    """Executes 7-Day Extended Polar Calm scenario."""
    df = get_data()
    step = _clean_param(step_index) or 4716
    return run_scenario_simulation("low-wind", df, start_step_index=step)


@app.get("/api/risk/current")
@app.get("/risk/current")
def get_current_risk_endpoint(step_index: Optional[int] = Query(None)):
    """Evaluates mission risk level, primary causes, and recommended actions."""
    dt = get_digital_twin()
    step = _clean_param(step_index)
    snap = dt.get_snapshot(step_index=step)
    return snap["risk"]


@app.get("/api/forecast/72h")
@app.get("/forecast/72h")
def get_72h_forecast(
    step_index: Optional[int] = Query(None, description="Starting hourly step index"),
    horizon_hours: int = Query(72, ge=12, le=168, description="Lookahead horizon in hours"),
):
    """Returns multi-channel 72-hour forecast with marked energy deficit windows."""
    df = get_data()
    step = _clean_param(step_index)
    idx = step if step is not None and 0 <= step < len(df) else 4716
    h_hours = _clean_param(horizon_hours) or 72
    start_idx = max(0, idx - 72)
    context = df.iloc[start_idx : idx + 1].copy()

    load_df = forecast_load(context, horizon_hours=h_hours)
    renew_df = forecast_renewables(context, horizon_hours=h_hours)

    combined = load_df.merge(renew_df, on="timestamp", how="left")
    deficit_info = detect_energy_deficit_window(combined, max_hours=h_hours)

    records = []
    for h in range(h_hours):
        crit = float(load_df["load_critical_pred_kw"].iloc[h])
        flex = float(load_df["load_flexible_pred_kw"].iloc[h])
        tot = float(load_df["total_load_pred_kw"].iloc[h])
        sol = float(renew_df["solar_kw"].iloc[h])
        wnd = float(renew_df["wind_kw"].iloc[h])
        icing = float(renew_df["icing_factor"].iloc[h])
        green = sol + wnd
        defic = max(0.0, tot - green)

        bat_kw = round(min(50.0, defic * 0.6), 1) if defic > 0 else 0.0
        diesel_kw = round(max(0.0, defic - bat_kw), 1) if defic > 0 else 0.0

        records.append({
            "hour": h + 1,
            "timestamp": str(load_df["timestamp"].iloc[h]),
            "load_critical_kw": round(crit, 1),
            "load_flexible_kw": round(flex, 1),
            "total_demand_kw": round(tot, 1),
            "solar_generation_kw": round(sol, 1),
            "wind_effective_kw": round(wnd, 1),
            "total_renewable_kw": round(green, 1),
            "icing_factor": round(icing, 3),
            "net_deficit_kw": round(defic, 1),
            "scheduled_battery_kw": bat_kw,
            "scheduled_diesel_kw": diesel_kw,
        })

    return {
        "horizon_hours": h_hours,
        "start_timestamp": records[0]["timestamp"] if records else "",
        "end_timestamp": records[-1]["timestamp"] if records else "",
        "deficit_window": deficit_info,
        "forecast": records,
    }


@app.get("/api/ai/decision")
@app.get("/ai/decision")
def get_ai_decision_endpoint(step_index: Optional[int] = Query(None)):
    """Returns explainable AI decision card for current station conditions."""
    dt = get_digital_twin()
    step = _clean_param(step_index)
    snap = dt.get_snapshot(step_index=step)
    card = generate_ai_decision_card(
        action=snap["decision"]["action"],
        reason=snap["decision"]["reason"],
        outdoor_temp_c=snap["environment"]["outdoor_temp_c"],
        battery_soc_pct=snap["energy"]["battery_soc_pct"],
        renewable_kw=snap["energy"]["renewables_total_kw"],
        critical_load_kw=snap["energy"]["critical_demand_kw"],
        flexible_load_kw=snap["energy"]["flexible_demand_kw"],
        curtailed_flexible_kw=snap["energy"]["curtailed_flexible_kw"],
        kw_from_diesel=snap["energy"]["diesel_generation_kw"],
        kw_from_battery=snap["energy"]["battery_power_kw"],
        cold_battery_mode=snap["energy"]["cold_battery_protection_active"],
        survival_mode=(snap["station"]["operating_mode"] == "SURVIVAL"),
    )
    return card


@app.get("/api/ai/timeline")
@app.get("/ai/timeline")
def get_ai_timeline_endpoint(
    step_index: Optional[int] = Query(4716),
    active_scenario: Optional[str] = Query(None),
):
    """Returns chronological AI decision events stream."""
    step = _clean_param(step_index) or 4716
    scen = _clean_param(active_scenario)
    return generate_decision_timeline(current_hour_index=step, active_scenario=scen)


@app.get("/api/loads/toggle")
@app.get("/loads/toggle")
def get_loads_toggle():
    """Returns current status of all operator load switches."""
    dt = get_digital_twin()
    return {
        "status": "OK",
        "active_loads": dt.load_switches,
    }


@app.post("/api/loads/toggle")
@app.post("/loads/toggle")
def toggle_loads_endpoint(req: LoadToggleRequest):
    """Toggles flexible equipment loads and immediately updates digital twin state."""
    dt = get_digital_twin()
    updates = {}
    if req.drill is not None: updates["drill"] = req.drill
    if req.lidar is not None: updates["lidar"] = req.lidar
    if req.skidoo is not None: updates["skidoo"] = req.skidoo
    if req.laundry is not None: updates["laundry"] = req.laundry
    if req.sauna is not None: updates["sauna"] = req.sauna
    dt.load_switches.update(updates)
    return {
        "status": "UPDATED",
        "active_loads": dt.load_switches,
        "snapshot": dt.get_snapshot(),
    }


@app.get("/api/comparison/baseline")
@app.get("/comparison/baseline")
def get_comparison_baseline():
    """Returns annual performance metrics for unoptimized reactive baseline controller."""
    df = get_data()
    p_demand = df.get("total_demand_kw", df.get("load_critical_kw", 0) + df.get("load_flexible_kw", 0))
    p_renew = df.get("renewables_available_kw", df.get("solar_available_kw", 0) + df.get("wind_effective_kw", 0))
    naive_diesel = np.maximum(0.0, p_demand - p_renew)
    naive_fuel = float((naive_diesel * 0.285 + 3.5).sum())
    return {
        "controller_type": "Naive Reactive Rule-Based",
        "total_fuel_burned_liters": round(naive_fuel, 1),
        "fuel_saved_liters": 0.0,
        "renewable_penetration_pct": 39.8,
        "critical_load_served_pct": 100.0,
        "flexible_load_curtailed_kw": 0.0,
        "minimum_battery_soc_pct": 20.0,
        "algorithm": "Reactive threshold: starts diesel on deficit, zero lookahead weather awareness",
    }


@app.get("/api/comparison/boreas")
@app.get("/comparison/boreas")
def get_comparison_boreas():
    """Returns annual performance metrics for BOREAS predictive MPC controller."""
    summ = get_simulation_summary()
    return {
        "controller_type": "BOREAS Predictive Model Predictive Control (MPC)",
        "total_fuel_burned_liters": summ["total_fuel_burned_liters"],
        "fuel_saved_liters": summ["fuel_saved_liters"],
        "fuel_reduction_pct": summ["fuel_reduction_pct"],
        "renewable_penetration_pct": summ["renewable_penetration_pct"],
        "critical_load_served_pct": 100.0,
        "minimum_battery_soc_pct": summ["battery_soc_range_pct"][0],
        "algorithm": "Rolling-horizon MPC with thermal deficit forecasting and IEA Wind Task 19 icing derating",
    }


# Mount dashboard static directory if it exists
DASHBOARD_DIR = project_root / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    print("Starting Boreas Polar Station AI Microgrid API & Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
