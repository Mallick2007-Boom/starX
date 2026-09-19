"""
Runner and Verification Suite for Polar Microgrid Optimization Engine.
Runs a 365-day hourly rolling simulation, verifies hard survival constraints,
computes diesel fuel savings vs baseline, and logs human-readable reason strings.
"""

from pathlib import Path
import sys
import time
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.optimizer import (
    MicrogridState,
    OptimizerConfig,
    optimize_hour,
    run_simulation,
)


def main():
    print("=" * 75)
    print(" Polar Research Station AI Microgrid: Energy Optimization Engine ")
    print("=" * 75)

    # 1. Test single-hour optimization across distinct polar operational regimes
    print("\n--- 1. Testing optimize_hour() with Explainable Reason Strings ---")

    test_state = MicrogridState(
        current_battery_kwh=180.0,
        current_fuel_reserve_liters=250_000.0,
        days_until_resupply=120.0,
        current_temp_c=-15.0,
    )

    # Scenario A: Summer Renewable Surplus (100% Green)
    forecast_surplus = {
        "load_critical_kw": 50.0,
        "load_flexible_kw": 40.0,
        "renewables_kw": 140.0,  # 50 kW surplus
        "future_renewables_mean_6h": 120.0,
    }
    alloc_a = optimize_hour(test_state, forecast_surplus)
    print("\n[Scenario A: High Renewable Surplus]")
    print(f"  Demand: 90 kW | Renewables: 140 kW")
    print(f"  Allocation: Diesel={alloc_a['kw_from_diesel']} kW | Renewables={alloc_a['kw_from_renewables']} kW | Battery={alloc_a['kw_from_battery']} kW ({alloc_a['battery_charge_or_discharge']})")
    print(f"  Reason: \"{alloc_a['reason']}\"")
    assert alloc_a["kw_from_diesel"] == 0.0, "Diesel should be off during surplus"
    assert alloc_a["battery_charge_or_discharge"] == "CHARGE"

    # Scenario B: High Wind Incoming -> Draw from Battery to Save Fuel
    forecast_deficit_windy = {
        "load_critical_kw": 80.0,
        "load_flexible_kw": 30.0,
        "renewables_kw": 70.0,   # 40 kW deficit
        "future_renewables_mean_6h": 110.0,  # strong wind forecast next 6h
    }
    alloc_b = optimize_hour(test_state, forecast_deficit_windy)
    print("\n[Scenario B: Moderate Deficit with Favorable Wind Forecast]")
    print(f"  Demand: 110 kW | Renewables: 70 kW | Deficit: 40 kW")
    print(f"  Allocation: Diesel={alloc_b['kw_from_diesel']} kW | Renewables={alloc_b['kw_from_renewables']} kW | Battery={alloc_b['kw_from_battery']} kW ({alloc_b['battery_charge_or_discharge']})")
    print(f"  Reason: \"{alloc_b['reason']}\"")
    assert alloc_b["kw_from_diesel"] == 0.0, "Battery should cover deficit when wind is incoming"
    assert alloc_b["battery_charge_or_discharge"] == "DISCHARGE"

    # Scenario C: Deep Polar Night Blizzard (Zero Solar, Low Wind)
    forecast_winter_storm = {
        "load_critical_kw": 135.0,
        "load_flexible_kw": 25.0,
        "renewables_kw": 15.0,   # severe deficit 145 kW
        "future_renewables_mean_6h": 12.0,  # lull in storm
    }
    low_bat_state = MicrogridState(
        current_battery_kwh=65.0,  # near 20% floor
        current_fuel_reserve_liters=180_000.0,
        days_until_resupply=180.0,
        current_temp_c=-58.0,
    )
    alloc_c = optimize_hour(low_bat_state, forecast_winter_storm)
    print("\n[Scenario C: Mid-Winter Blizzard at -58°C with Battery Near Minimum]")
    print(f"  Demand: 160 kW | Renewables: 15 kW | Deficit: 145 kW")
    print(f"  Allocation: Diesel={alloc_c['kw_from_diesel']} kW | Renewables={alloc_c['kw_from_renewables']} kW | Battery={alloc_c['kw_from_battery']} kW ({alloc_c['battery_charge_or_discharge']})")
    print(f"  Reason: \"{alloc_c['reason']}\"")
    assert alloc_c["kw_from_diesel"] > 100.0
    assert alloc_c["critical_load_met_pct"] == 100.0

    # 2. Run Full 365-Day (8,760-Hour) Microgrid Rolling Simulation
    print("\n" + "-" * 75)
    print(" 2. Running 365-Day Rolling-Horizon Microgrid Simulation ")
    print("-" * 75)

    t0 = time.time()
    sim_df, summary = run_simulation(
        days=365,
        initial_battery_kwh=150.0,
        initial_fuel_liters=360_000.0,
        safety_buffer_pct=0.10,
    )
    sim_time = time.time() - t0

    print(f"Simulation completed in {sim_time:.2f} seconds ({len(sim_df):,} hourly timesteps).")

    # 3. Hard Constraint Validations
    print("\n--- Hard Constraint Verification ---")
    assert summary["critical_load_survival_rate_pct"] == 100.0, "Critical load was curtailed!"
    print(f"  [PASS] Critical Load Security: 100.0% met across all 8,760 hours (Zero curtailment)")

    assert not summary["zero_fuel_violation"], "Fuel reserve dropped to zero!"
    final_fuel = summary["final_fuel_reserve_liters"]
    print(f"  [PASS] Fuel Depletion Guard: Fuel reserve stayed positive (Final Reserve: {final_fuel:,.0f} L)")

    assert summary["min_battery_soc_pct"] >= 20.0, f"Battery SoC violated 20% floor: {summary['min_battery_soc_pct']}%"
    assert summary["max_battery_soc_pct"] <= 100.0, f"Battery SoC exceeded 100%: {summary['max_battery_soc_pct']}%"
    print(f"  [PASS] Battery SoC Safety Range: [{summary['min_battery_soc_pct']:.1f}%, {summary['max_battery_soc_pct']:.1f}%]")

    # 4. Energy & Fuel Conservation Summary
    print("\n" + "=" * 75)
    print(" ANNUAL OPTIMIZATION PERFORMANCE SUMMARY ")
    print("=" * 75)
    print(f"Initial Annual Fuel Budget:       {summary['initial_fuel_liters']:>12,.0f} Liters")
    print(f"End-of-Year Safety Reserve:       {summary['final_fuel_reserve_liters']:>12,.0f} Liters")
    print(f"Total Diesel Burned (AI MPC):     {summary['total_fuel_burned_liters']:>12,.0f} Liters")
    print(f"Total Diesel Burned (Baseline):   {summary['naive_baseline_fuel_liters']:>12,.0f} Liters")
    print(f"Total Fuel Saved:                 {summary['fuel_saved_liters']:>12,.0f} Liters")
    print(f"Fuel Consumption Reduction:       {summary['fuel_savings_pct']:>11.1f}%")
    print(f"Renewable Energy Penetration:     {summary['renewable_penetration_pct']:>11.1f}%")

    # 5. Display Sample Hourly Decision Logs across the Year
    print("\n--- Sample Hourly Allocation Logs & Reason Strings ---")
    sample_indices = [50, 2500, 4800, 7200]
    preview_cols = [
        "timestamp",
        "outdoor_temp_c",
        "total_demand_kw",
        "renewables_available_kw",
        "kw_from_diesel",
        "kw_from_battery",
        "battery_soc_pct",
        "fuel_reserve_liters",
    ]
    print(sim_df[preview_cols].iloc[sample_indices].to_string(index=False))

    print("\nCorresponding Explainability Reasons:")
    for idx in sample_indices:
        ts = sim_df["timestamp"].iloc[idx]
        reason = sim_df["reason"].iloc[idx]
        print(f"  [{ts}] -> \"{reason}\"")

    # 6. Save simulation results for dashboard and API
    output_csv = project_root / "models" / "simulation_results_365d.csv"
    sim_df.to_csv(output_csv, index=False)
    print(f"\nSaved 365-day simulation logs to:\n  {output_csv}")

    print("\n" + "=" * 75)
    print(" Optimization Engine Validation Complete! All Tests Passed. ")
    print("=" * 75)


if __name__ == "__main__":
    main()
