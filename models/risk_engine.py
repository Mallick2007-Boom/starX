"""
BOREAS Mission Risk Engine (Module: risk_engine.py)
Evaluates current and lookahead polar station telemetry across physical and logistics
constraints to produce transparent, physics-grounded operational risk levels:
NORMAL, WATCH, WARNING, or SURVIVAL.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RiskAssessment:
    """Structured assessment output from the BOREAS Mission Risk Engine."""
    risk_level: str                      # NORMAL | WATCH | WARNING | SURVIVAL
    risk_score: float                    # 0 to 100 composite constraint-stress index
    primary_causes: List[str]            # Interpretable physical and logistics drivers
    recommended_action: str              # Concrete operational command recommendation
    autonomy_days: float                 # Estimated days of fuel remaining
    resupply_buffer_margin_pct: float    # Safety buffer above required resupply fuel
    battery_projected_min_soc: float     # Lowest predicted battery SoC in lookahead window


def evaluate_mission_risk(
    battery_soc_pct: float,
    predicted_min_soc_24h: float,
    fuel_reserve_liters: float,
    daily_burn_liters: float,
    days_until_resupply: float,
    critical_load_kw: float,
    renewable_kw: float,
    wind_icing_factor: float,
    outdoor_temp_c: float,
    generator_1_online: bool = True,
    generator_2_online: bool = True,
    battery_online: bool = True,
    severe_storm_active: bool = False,
) -> RiskAssessment:
    """
    Evaluates polar research station risk based on non-linear physical conditions.
    Never uses black-box arbitrariness; grounded in fuel buffer, thermal limits,
    battery chemistry floors, and equipment availability.
    """
    causes: List[str] = []
    stress_points: float = 0.0

    # 1. Fuel Logistics & Autonomy Margin
    daily_burn = max(10.0, daily_burn_liters)
    autonomy_days = fuel_reserve_liters / daily_burn
    required_fuel = daily_burn * max(1.0, days_until_resupply)
    buffer_liters = fuel_reserve_liters - required_fuel
    resupply_buffer_pct = (buffer_liters / required_fuel) * 100.0 if required_fuel > 0 else 100.0

    if autonomy_days < days_until_resupply:
        causes.append(f"Fuel exhaustion projected in {autonomy_days:.1f} days, BEFORE annual resupply vessel ({days_until_resupply:.0f} days)")
        stress_points += 45.0
    elif resupply_buffer_pct < 10.0:
        causes.append(f"Critical fuel safety buffer ({resupply_buffer_pct:.1f}% margin remaining)")
        stress_points += 25.0
    elif resupply_buffer_pct < 20.0:
        causes.append(f"Narrow fuel safety buffer ({resupply_buffer_pct:.1f}% margin above resupply)")
        stress_points += 10.0

    # 2. Battery Storage & Freeze Limits
    min_soc = min(battery_soc_pct, predicted_min_soc_24h)
    if not battery_online:
        causes.append("Battery Energy Storage System (BESS) is OFFLINE / Tripped")
        stress_points += 30.0
    elif min_soc <= 20.5:
        causes.append(f"Battery SoC approaching sub-zero physical degradation floor ({min_soc:.1f}%)")
        stress_points += 30.0
    elif min_soc <= 30.0:
        causes.append(f"Battery storage constrained ({min_soc:.1f}% minimum projected)")
        stress_points += 15.0

    # 3. Renewable Collapse & Severe Aerodynamic Icing
    if wind_icing_factor <= 0.35:
        causes.append(f"Severe turbine blade rime icing detected ({int((1 - wind_icing_factor)*100)}% power loss)")
        stress_points += 20.0
    elif wind_icing_factor <= 0.60:
        causes.append(f"Moderate blade icing derating active ({int((1 - wind_icing_factor)*100)}% power loss)")
        stress_points += 10.0

    deficit = max(0.0, critical_load_kw - renewable_kw)
    if renewable_kw <= 5.0:
        causes.append("Zero/negligible renewable generation (polar night / calm blizzard)")
        stress_points += 15.0
    elif deficit > 50.0:
        causes.append(f"Substantial renewable deficit ({deficit:.1f} kW critical load unserved by green power)")
        stress_points += 10.0

    # 4. Extreme Cold & Heating Deficit
    if outdoor_temp_c <= -50.0:
        causes.append(f"Extreme polar deep freeze ({outdoor_temp_c:.1f}°C) driving high heating load and boiler fuel burn")
        stress_points += 20.0
    elif outdoor_temp_c <= -40.0:
        causes.append(f"Severe cold ({outdoor_temp_c:.1f}°C) requiring Cold Battery Protection mode")
        stress_points += 10.0

    # 5. Generator Equipment Failures
    if not generator_1_online and not generator_2_online:
        causes.append("CATASTROPHIC: Dual generator outage! Station running on emergency battery buffer")
        stress_points += 60.0
    elif not generator_1_online:
        causes.append("Primary Generator #1 is OFFLINE. Operating on single backup genset (no N+1 redundancy)")
        stress_points += 25.0
    elif not generator_2_online:
        causes.append("Secondary Generator #2 is OFFLINE. Redundancy degraded")
        stress_points += 15.0

    if severe_storm_active:
        causes.append("Active Katabatic Blizzard event: gale-force winds and zero visibility")
        stress_points += 20.0

    # Classify overall Risk Level
    stress_points = min(100.0, stress_points)

    if (
        (autonomy_days < days_until_resupply)
        or (not generator_1_online and not generator_2_online)
        or (min_soc <= 20.0 and deficit > 40.0)
        or (stress_points >= 65.0)
    ):
        risk_level = "SURVIVAL"
        recommended_action = (
            "ENGAGE SURVIVAL PROTOCOL: Immediately shed all flexible science operations (drill rigs, lidars, chargers). "
            "Enforce strict habitat heating setback and prioritize diesel/battery preservation."
        )
    elif stress_points >= 40.0:
        risk_level = "WARNING"
        recommended_action = (
            "ENERGY WARNING: Constrain flexible science loads, raise battery reserve floor to 30%, "
            "and prepare backup generator for automated sync."
        )
    elif stress_points >= 20.0:
        risk_level = "WATCH"
        recommended_action = (
            "STORM WATCH: Monitor renewable generation deficit and pre-charge battery buffer during available wind intervals."
        )
    else:
        risk_level = "NORMAL"
        recommended_action = (
            "NOMINAL CONDITIONS: Station operating within safe energy margins. All science and operational loads approved."
        )

    if not causes:
        causes.append("Nominal renewable penetration and fuel reserves within safe logistical buffers.")

    return RiskAssessment(
        risk_level=risk_level,
        risk_score=round(stress_points, 1),
        primary_causes=causes,
        recommended_action=recommended_action,
        autonomy_days=round(autonomy_days, 1),
        resupply_buffer_margin_pct=round(resupply_buffer_pct, 1),
        battery_projected_min_soc=round(min_soc, 1),
    )
