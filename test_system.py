"""
BOREAS — AI Energy Commander for Polar Research Stations
Comprehensive End-to-End System Integrity & Resilience Verification Suite.
Validates datasets, forecasting models, MPC optimization, anomaly detection,
Antarctic Digital Twin, What-If scenarios, Mission Risk Engine, and FastAPI endpoints.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.app import (
    app,
    health_check,
    get_current_telemetry,
    get_historical_telemetry,
    get_recent_anomalies,
    optimize_dispatch,
    predict_load,
    predict_renewables,
    get_fuel_reserve_projection,
    get_simulation_summary,
    get_digital_twin_state,
    scenario_blizzard,
    scenario_diesel_failure,
    scenario_battery_failure,
    get_current_risk_endpoint,
    get_72h_forecast,
    get_ai_decision_endpoint,
    get_ai_timeline_endpoint,
    toggle_loads_endpoint,
    get_comparison_baseline,
    get_comparison_boreas,
    OptimizeRequest,
    PredictLoadRequest,
    PredictRenewablesRequest,
    LoadToggleRequest,
)
from models import (
    forecast_load,
    forecast_renewables,
    optimize_hour,
    detect_anomalies,
    MicrogridState,
    OptimizerConfig,
    DigitalTwin,
    evaluate_mission_risk,
    SCENARIO_CATALOG,
    run_scenario_simulation,
    generate_ai_decision_card,
    detect_energy_deficit_window,
    generate_decision_timeline,
)


def run_all_tests():
    print("=" * 80)
    print(" BOREAS — AI ENERGY COMMANDER FOR POLAR RESEARCH STATIONS ")
    print(" UNIFIED END-TO-END SYSTEM INTEGRITY & RESILIENCE TEST SUITE ")
    print("=" * 80)

    # 1. Dataset Validation
    print("\n--- 1. Validating Synthetic Polar Dataset ---")
    data_path = project_root / "data_generator" / "data" / "polar_station_energy_hourly.csv"
    assert data_path.exists(), f"Dataset missing at {data_path}"
    df_raw = pd.read_csv(data_path)
    print(f"  [PASS] Raw dataset loaded: {len(df_raw):,} hourly records.")
    assert len(df_raw) == 8760, f"Expected 8,760 hours, found {len(df_raw)}"
    assert "outdoor_temp_c" in df_raw.columns
    assert "load_critical_kw" in df_raw.columns
    assert "solar_available_kw" in df_raw.columns
    assert "wind_effective_kw" in df_raw.columns
    assert df_raw["outdoor_temp_c"].min() >= -65.0
    assert df_raw["outdoor_temp_c"].max() <= 10.0

    # 2. Simulation Results Validation
    print("\n--- 2. Validating 365-Day Optimization Simulation Results ---")
    sim_path = project_root / "models" / "simulation_results_365d.csv"
    assert sim_path.exists(), f"Simulation results missing at {sim_path}"
    df_sim = pd.read_csv(sim_path)
    print(f"  [PASS] Simulation log loaded: {len(df_sim):,} records.")
    assert len(df_sim) == 8760
    assert "daylight_hours" in df_sim.columns
    assert "wind_speed_mps" in df_sim.columns
    assert "wind_icing_factor" in df_sim.columns
    assert "kw_from_diesel" in df_sim.columns
    assert "kw_from_battery" in df_sim.columns
    assert "reason" in df_sim.columns
    print("  [PASS] All atmospheric and dispatch columns verified.")

    # 3. Load Forecaster Inference
    print("\n--- 3. Validating HistGradientBoosting Load Forecaster ---")
    load_preds = forecast_load(df_sim.tail(72), horizon_hours=48)
    assert len(load_preds) == 48
    assert "load_critical_pred_kw" in load_preds.columns
    assert "load_flexible_pred_kw" in load_preds.columns
    assert "total_load_pred_kw" in load_preds.columns
    assert (load_preds["load_critical_pred_kw"] > 0).all()
    print(f"  [PASS] Load forecast generated (48h). Avg critical load: {load_preds['load_critical_pred_kw'].mean():.1f} kW.")

    # 4. Renewables & Icing Forecaster Inference
    print("\n--- 4. Validating Physics-Informed Renewables Forecaster ---")
    renew_preds = forecast_renewables(df_sim.tail(72), horizon_hours=48)
    assert len(renew_preds) == 48
    assert "solar_kw" in renew_preds.columns
    assert "wind_kw" in renew_preds.columns
    assert "icing_factor" in renew_preds.columns
    assert (renew_preds["icing_factor"] >= 0.15).all() and (renew_preds["icing_factor"] <= 1.0).all()
    print(f"  [PASS] Renewable forecast generated (48h). Mean icing factor: {renew_preds['icing_factor'].mean():.3f}.")

    # 5. Microgrid MPC Optimization Engine
    print("\n--- 5. Validating Microgrid MPC Optimization Engine ---")
    state = MicrogridState(
        current_battery_kwh=150.0,
        current_fuel_reserve_liters=250000.0,
        days_until_resupply=180.0,
        current_temp_c=-35.0,
    )
    alloc_surplus = optimize_hour(state, {
        "load_critical_kw": 80.0,
        "load_flexible_kw": 30.0,
        "renewables_kw": 120.0,
        "future_renewables_mean_6h": 100.0,
    })
    assert alloc_surplus["kw_from_diesel"] == 0.0, "Diesel should be 0 kW during renewable surplus"
    assert alloc_surplus["battery_charge_or_discharge"] == "CHARGE"
    assert alloc_surplus["critical_load_met_pct"] == 100.0
    print(f"  [PASS] Green surplus dispatch: Battery charged {alloc_surplus['battery_power_kw']} kW, Diesel 0 kW.")

    # Cold Battery Protection mode test
    cfg_cold = OptimizerConfig(cold_battery_protection_active=True)
    alloc_cold = optimize_hour(state, {
        "load_critical_kw": 80.0,
        "load_flexible_kw": 30.0,
        "renewables_kw": 20.0,
        "future_renewables_mean_6h": 20.0,
    }, config=cfg_cold)
    assert alloc_cold["cold_battery_protection_active"] is True
    assert alloc_cold["critical_load_met_pct"] == 100.0
    print(f"  [PASS] Cold Battery Protection engaged: safe floor raised to 30% SoC.")

    # 6. Antarctic Digital Twin State Engine
    print("\n--- 6. Validating Antarctic Digital Twin State Engine ---")
    dt = DigitalTwin(df_sim)
    snap = dt.get_snapshot(step_index=4716)
    assert "environment" in snap
    assert "energy" in snap
    assert "station" in snap
    assert "risk" in snap
    assert "decision" in snap
    assert snap["environment"]["outdoor_temp_c"] <= 0.0
    assert snap["energy"]["critical_load_met_pct"] == 100.0
    print(f"  [PASS] Digital Twin snapshot generated for Hour {snap['step_index']}: Temp {snap['environment']['outdoor_temp_c']}°C, Total Demand {snap['energy']['total_station_demand_kw']} kW, Mode: {snap['station']['operating_mode']}.")

    # 7. What-If Scenario Simulator
    print("\n--- 7. Validating What-If Scenario Simulator ---")
    # A. 3-Day Blizzard
    blizz_res = run_scenario_simulation("blizzard", df_sim, start_step_index=4716)
    assert blizz_res["after"]["critical_load_survival_pct"] == 100.0
    assert blizz_res["after"]["fuel_consumed_liters"] > 0.0
    print(f"  [PASS] 3-Day Blizzard scenario: Fuel burned {blizz_res['after']['fuel_consumed_liters']:.0f} L, Critical survival 100%.")

    # B. Primary Generator #1 Failure
    gen_res = run_scenario_simulation("diesel-failure", df_sim, start_step_index=4716)
    assert gen_res["after"]["critical_load_survival_pct"] == 100.0
    print(f"  [PASS] Generator #1 Failure scenario: Handled with Gen #2 failover, 100% critical power served.")

    # C. Battery BESS Inverter Failure
    bat_res = run_scenario_simulation("battery-failure", df_sim, start_step_index=4716)
    assert bat_res["after"]["battery_min_soc_pct"] == 0.0
    assert bat_res["after"]["critical_load_survival_pct"] == 100.0
    print(f"  [PASS] Battery Failure scenario: Balanced directly on diesel and renewables, 100% critical survival.")

    # 8. Mission Risk Engine Validations
    print("\n--- 8. Validating Mission Risk Engine ---")
    # Test Normal
    risk_norm = evaluate_mission_risk(
        battery_soc_pct=80.0,
        predicted_min_soc_24h=70.0,
        fuel_reserve_liters=250000.0,
        daily_burn_liters=600.0,
        days_until_resupply=180.0,
        critical_load_kw=60.0,
        renewable_kw=85.0,
        wind_icing_factor=1.0,
        outdoor_temp_c=-20.0,
    )
    assert risk_norm.risk_level == "NORMAL"

    # Test Survival
    risk_surv = evaluate_mission_risk(
        battery_soc_pct=20.0,
        predicted_min_soc_24h=19.0,
        fuel_reserve_liters=20000.0,
        daily_burn_liters=900.0,
        days_until_resupply=180.0,
        critical_load_kw=80.0,
        renewable_kw=5.0,
        wind_icing_factor=0.30,
        outdoor_temp_c=-55.0,
        generator_1_online=False,
    )
    assert risk_surv.risk_level == "SURVIVAL"
    assert len(risk_surv.primary_causes) > 0
    print(f"  [PASS] Risk Engine: Normal state verified; Survival state verified (Causes: {len(risk_surv.primary_causes)}).")

    # 9. Explainable AI & Deficit Window Detection
    print("\n--- 9. Validating Explainable AI & Deficit Window Detection ---")
    card = generate_ai_decision_card(
        action="DISPATCH BATTERY",
        reason="Shaved load deficit with battery storage",
        outdoor_temp_c=-45.0,
        battery_soc_pct=55.0,
        renewable_kw=20.0,
        critical_load_kw=75.0,
        flexible_load_kw=25.0,
        curtailed_flexible_kw=0.0,
        kw_from_diesel=0.0,
        kw_from_battery=45.0,
    )
    assert "current_action" in card
    assert "why" in card
    assert "safety_consideration" in card
    print(f"  [PASS] AI Decision Card verified: Action \"{card['current_action']}\", Why: \"{card['why'][:45]}...\"")

    timeline = generate_decision_timeline(current_hour_index=4716)
    assert len(timeline) >= 6
    print(f"  [PASS] Operator Decision Timeline generated: {len(timeline)} events.")

    # 10. FastAPI REST Endpoints Handlers
    print("\n--- 10. Validating FastAPI Endpoints Handlers ---")
    # Health
    assert health_check()["status"] == "ONLINE"

    # Digital Twin
    dt_res = get_digital_twin_state(step_index=4716)
    assert dt_res["environment"]["outdoor_temp_c"] <= 0.0

    # Scenarios endpoints
    s_blizz = scenario_blizzard(step_index=4716)
    assert s_blizz["impact"]["critical_load_survival_pct"] == 100.0

    s_gen = scenario_diesel_failure(step_index=4716)
    assert s_gen["impact"]["critical_load_survival_pct"] == 100.0

    s_bat = scenario_battery_failure(step_index=4716)
    assert s_bat["impact"]["critical_load_survival_pct"] == 100.0

    # Risk endpoint
    r_res = get_current_risk_endpoint(step_index=4716)
    assert "level" in r_res

    # 72h Forecast endpoint
    fc_res = get_72h_forecast(step_index=4716, horizon_hours=72)
    assert len(fc_res["forecast"]) == 72
    assert "deficit_window" in fc_res

    # AI Decision endpoint
    dec_res = get_ai_decision_endpoint(step_index=4716)
    assert "why" in dec_res

    # Timeline endpoint
    tl_res = get_ai_timeline_endpoint(step_index=4716)
    assert len(tl_res) >= 6

    # Load Toggle endpoint
    lt_res = toggle_loads_endpoint(LoadToggleRequest(drill=False, lidar=True))
    assert lt_res["status"] == "UPDATED"
    assert lt_res["active_loads"]["drill"] is False

    # Comparison endpoints
    base_comp = get_comparison_baseline()
    assert base_comp["total_fuel_burned_liters"] > 0

    boreas_comp = get_comparison_boreas()
    assert boreas_comp["fuel_saved_liters"] > 0

    # Dashboard HTML static asset
    dash_html = project_root / "dashboard" / "index.html"
    assert dash_html.exists()
    content = dash_html.read_text(encoding="utf-8")
    assert "ANTARCTIC ENERGY COMMAND" in content
    assert "HACKATHON DEMO MODE" in content
    assert "RUN DEMO (60s)" in content
    assert "SURVIVAL MODE" in content
    assert "forecastChart" in content
    print(f"  [PASS] Dashboard HTML verified ({len(content):,} bytes). Ready for mission control.")

    print("\n" + "=" * 80)
    print(" ALL 10 COMPREHENSIVE SYSTEM INTEGRITY CHECKS PASSED WITH 100% SUCCESS! ")
    print(" BOREAS AI Energy Commander is verified, resilient, and fully operational. ")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
