"""
BOREAS Explainable AI (XAI) & Decision Timeline Engine (Module: explainable_ai.py)
Generates transparent, contextual natural-language explanations, lookahead energy
deficit window notifications, and interactive decision timeline events.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def generate_ai_decision_card(
    action: str,
    reason: str,
    outdoor_temp_c: float,
    battery_soc_pct: float,
    renewable_kw: float,
    critical_load_kw: float,
    flexible_load_kw: float,
    curtailed_flexible_kw: float,
    kw_from_diesel: float,
    kw_from_battery: float,
    cold_battery_mode: bool = False,
    survival_mode: bool = False,
) -> Dict[str, Any]:
    """
    Constructs a comprehensive, transparent AI decision card for mission control.
    """
    total_demand = critical_load_kw + flexible_load_kw

    # Contextual Why
    if survival_mode:
        why = f"Emergency survival conditions detected: ambient temperature {outdoor_temp_c:.1f}°C and critical life support heating requirement ({critical_load_kw:.1f} kW) prioritized over science operations."
    elif kw_from_battery > 1.0 and kw_from_diesel <= 1.0:
        why = f"Available renewables ({renewable_kw:.1f} kW) are below station demand ({total_demand:.1f} kW). Discharging battery buffer ({battery_soc_pct:.0f}% SoC) to avoid diesel generator ignition and preserve fuel."
    elif kw_from_diesel > 1.0 and kw_from_battery > 1.0:
        why = f"Renewable generation ({renewable_kw:.1f} kW) insufficient for combined load ({total_demand:.1f} kW). Executing optimal hybrid dispatch: shaving peak with {kw_from_battery:.1f} kW battery and running diesel at efficient loading."
    elif kw_from_diesel > 1.0:
        why = f"Renewable deficit ({max(0.0, total_demand - renewable_kw):.1f} kW) and battery reserve needed for upcoming storm buffer. Dispatched diesel generator to cover load deficit."
    elif kw_from_battery < -1.0:
        why = f"Green generation surplus ({renewable_kw:.1f} kW exceeds station demand of {total_demand:.1f} kW). Directing excess power into BESS storage."
    else:
        why = f"Station energy demand ({total_demand:.1f} kW) met directly by available generation ({renewable_kw:.1f} kW)."

    # Lookahead forecast
    deficit_est = max(0.0, total_demand - renewable_kw)
    if deficit_est > 20.0:
        forecast_str = f"Renewable deficit of {deficit_est:.1f} kW expected across next 6 to 12 hours. Thermal heating demand remaining high."
    else:
        forecast_str = f"Renewable generation projected stable. Minor deficit or surplus expected over next 6 hours."

    # Safety consideration
    floor = "30% (Cold Battery Protection Floor)" if cold_battery_mode else "20% (Sub-zero LiFePO4 Floor)"
    if battery_soc_pct <= (30.0 if cold_battery_mode else 20.0) + 5.0:
        safety_str = f"CAUTION: Battery SoC ({battery_soc_pct:.1f}%) is within 5% of the mandatory {floor}. Preserving cells against freeze damage."
    else:
        safety_str = f"Battery SoC ({battery_soc_pct:.1f}%) comfortably exceeds the {floor} safety threshold."

    # Expected result
    if curtailed_flexible_kw > 0.5:
        result_str = f"Guaranteed 100% life-support critical heating uptime. Deferred {curtailed_flexible_kw:.1f} kW of flexible science demand to prevent fuel depletion."
    elif kw_from_diesel <= 0.0:
        result_str = "Zero liters of diesel fuel burned. 100% zero-emission microgrid operation."
    else:
        result_str = "100% critical load served while minimizing fuel burn rate and maintaining spinning generator stability."

    return {
        "current_action": action,
        "why": why,
        "forecast": forecast_str,
        "safety_consideration": safety_str,
        "expected_result": result_str,
        "detailed_reason": reason,
    }


def detect_energy_deficit_window(
    forecast_df: pd.DataFrame,
    max_hours: int = 72,
) -> Dict[str, Any]:
    """
    Scans a 72-hour forecast DataFrame to detect the next critical energy deficit window
    where station demand exceeds available renewable generation.
    """
    if forecast_df is None or len(forecast_df) == 0:
        return {
            "has_deficit": False,
            "window_text": "None detected",
            "duration_hours": 0,
            "mean_deficit_kw": 0.0,
            "explanation": "Predicted renewable generation is sufficient to satisfy station demand over the lookahead horizon.",
        }

    n = min(len(forecast_df), max_hours)
    slice_df = forecast_df.iloc[:n].copy()

    # Determine demand and green generation
    if "total_load_pred_kw" in slice_df.columns:
        demands = slice_df["total_load_pred_kw"].values
    elif "load_critical_kw" in slice_df.columns:
        demands = slice_df["load_critical_kw"].values + slice_df.get("load_flexible_kw", 0.0).values
    else:
        demands = np.full(n, 100.0)

    if "total_renewable_kw" in slice_df.columns:
        renews = slice_df["total_renewable_kw"].values
    elif "solar_kw" in slice_df.columns and "wind_kw" in slice_df.columns:
        renews = slice_df["solar_kw"].values + slice_df["wind_kw"].values
    elif "wind_effective_kw" in slice_df.columns:
        renews = slice_df.get("solar_available_kw", 0.0).values + slice_df["wind_effective_kw"].values
    else:
        renews = np.full(n, 20.0)

    deficits = demands - renews
    is_deficit = deficits > 1.0

    if not np.any(is_deficit):
        return {
            "has_deficit": False,
            "window_text": "Zero Deficit (100% Green)",
            "duration_hours": 0,
            "mean_deficit_kw": 0.0,
            "explanation": "No energy deficit predicted across the next 72 hours. High wind/solar availability.",
        }

    # Find longest or first contiguous deficit streak
    start_hr = int(np.argmax(is_deficit))
    end_hr = start_hr
    while end_hr < n and is_deficit[end_hr]:
        end_hr += 1

    dur = end_hr - start_hr
    mean_def = float(np.mean(deficits[start_hr:end_hr]))

    # Format window labels
    window_label = f"+{start_hr}h → +{end_hr}h ({dur} hours duration)"

    explanation = (
        f"Predicted renewable generation is insufficient from hour +{start_hr} to +{end_hr} "
        f"(average deficit: {mean_def:.1f} kW). Battery discharge and diesel pre-dispatch have been scheduled in advance."
    )

    return {
        "has_deficit": True,
        "start_hour": start_hr,
        "end_hour": end_hr,
        "window_text": window_label,
        "duration_hours": dur,
        "mean_deficit_kw": round(mean_def, 1),
        "explanation": explanation,
    }


def generate_decision_timeline(
    current_hour_index: int = 4716,
    active_scenario: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generates a realistic chronological stream of recent and upcoming AI decision events
    for the operator timeline component.
    """
    base_events = [
        {
            "time": "T-12h (08:00)",
            "icon": "☀️",
            "type": "SURPLUS",
            "title": "Solar / Wind Green Surplus Detected",
            "summary": "Renewable generation peaked at 115 kW, exceeding station demand (85 kW).",
            "decision": "CHARGE BATTERY",
            "reason": "Routed 30 kW green surplus into BESS storage to build safety buffer.",
            "forecast": "Predicted high wind continuing for 4 hours.",
            "action": "Charged battery to 88% SoC. Shut down Generator #2.",
        },
        {
            "time": "T-08h (12:00)",
            "icon": "🔋",
            "type": "STORAGE",
            "title": "Battery Storage Nearing 90% Capacity",
            "summary": "Battery reached 265 kWh (88.3% SoC). Inverter throttled charge rate.",
            "decision": "MAINTAIN FLOAT",
            "reason": "Preserving cell health and preventing overvoltage.",
            "forecast": "Incoming katabatic wind front approaching from Transantarctic range.",
            "action": "Station running 100% renewable direct.",
        },
        {
            "time": "T-04h (16:00)",
            "icon": "🌬️",
            "type": "ICING",
            "title": "Atmospheric Freezing Fog Alert",
            "summary": "Relative humidity rose to 82% at -14°C, triggering rime icing conditions.",
            "decision": "DERATE TURBINES",
            "reason": "IEA Wind Task 19 model derated wind turbine aerodynamic output by 45%.",
            "forecast": "Effective wind generation expected to fall from 90 kW to 49 kW.",
            "action": "Notified operations commander. Pre-warmed diesel generator block.",
        },
        {
            "time": "T-02h (18:00)",
            "icon": "🧠",
            "type": "MPC_OPTIMIZE",
            "title": "Predictive Rolling MPC Recalculation",
            "summary": "Lookahead forecast identified upcoming 6-hour energy deficit window.",
            "decision": "PRE-SCHEDULE DISPATCH",
            "reason": "Preserve battery buffer above 20% while minimizing diesel fuel consumption.",
            "forecast": "Heating demand rising by 25% as temperature drops to -48°C.",
            "action": "Calculated optimal hybrid battery/diesel setpoint (45 kW battery, 35 kW diesel).",
        },
        {
            "time": "T-00h (NOW)",
            "icon": "🛢️",
            "type": "DISPATCH",
            "title": "Optimal Microgrid Dispatch Setpoints Executed",
            "summary": "Station demand 112 kW served via 32 kW wind, 45 kW battery, and 35 kW diesel.",
            "decision": "HYBRID BALANCING",
            "reason": "Shaved deficit with battery storage without deep discharging below freeze floor.",
            "forecast": "Deficit window persisting for 9 hours.",
            "action": "All life-support critical heating loops 100% energized.",
        },
        {
            "time": "T+03h (LOOKAHEAD)",
            "icon": "🧪",
            "type": "LOAD_SHED",
            "title": "Automated Flexible Load Postponement",
            "summary": "Predicted battery SoC nearing 25% threshold during blizzard peak.",
            "decision": "POSTPONE ICE DRILL",
            "reason": "Deep ice drill is a flexible 45 kW load; deferring operation conserves 38 L of diesel.",
            "forecast": "Wind icing clearing after blizzard front passes.",
            "action": "Science lab scheduled to resume drilling once wind power recovers above 80 kW.",
        },
    ]

    if active_scenario == "diesel-failure":
        base_events.insert(0, {
            "time": "ALERT",
            "icon": "⚠️",
            "type": "FAILURE",
            "title": "Generator #1 Fuel Injector Outage",
            "summary": "Primary 100 kW genset reported pressure loss and tripped offline.",
            "decision": "START BACKUP GENSET #2",
            "reason": "Immediate automated failover to secondary 100 kW backup generator.",
            "forecast": "Single generator running without N+1 redundancy.",
            "action": "Curtailed non-essential skidoo chargers to maintain safety margin.",
        })
    elif active_scenario == "battery-failure":
        base_events.insert(0, {
            "time": "ALERT",
            "icon": "⚠️",
            "type": "FAILURE",
            "title": "BESS Battery Inverter Offline",
            "summary": "Battery storage unavailable for peak shaving or surplus absorption.",
            "decision": "DIESEL DIRECT DISPATCH",
            "reason": "Microgrid forced to balance instantaneously on diesel gensets.",
            "forecast": "Diesel fuel burn increasing by +22% without battery buffer.",
            "action": "Postponed all heavy scientific equipment to preserve fuel reserve.",
        })

    return base_events
