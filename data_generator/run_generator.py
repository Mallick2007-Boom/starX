"""
CLI entry point to generate, validate, and export 1 year of hourly polar station synthetic data.
"""

from pathlib import Path
import sys
import numpy as np

# Ensure local imports work when executing script directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_generator.config import StationConfig
from data_generator.generator import PolarStationDataGenerator


def validate_dataset(df) -> bool:
    """Run physics and data integrity assertions on the generated dataset."""
    print("\n--- Running Physics & Domain Consistency Checks ---")
    all_passed = True

    # Check 1: Record count (8,760 hours for non-leap year)
    n_rows = len(df)
    assert n_rows in (8760, 8784), f"Unexpected row count: {n_rows}"
    print(f"  [PASS] Timestep count: {n_rows} hourly records")

    # Check 2: Temperature bounds (-62°C to +3°C)
    t_min = df["outdoor_temp_c"].min()
    t_max = df["outdoor_temp_c"].max()
    assert -65.0 <= t_min <= -50.0, f"Min temp outside expected polar range: {t_min}"
    assert -5.0 <= t_max <= 5.0, f"Max temp outside expected polar range: {t_max}"
    print(f"  [PASS] Outdoor Temp Range: {t_min:.1f}°C to {t_max:.1f}°C")

    # Check 3: Daylight hours range (0 for polar night, up to 24 in summer)
    d_min = df["daylight_hours"].min()
    d_max = df["daylight_hours"].max()
    polar_night_hours = (df["daylight_hours"] == 0).sum()
    polar_night_days = polar_night_hours / 24.0
    assert d_min == 0.0, f"Expected 0 daylight hours during polar night, got {d_min}"
    assert d_max == 24.0, f"Expected 24 daylight hours during polar day, got {d_max}"
    assert 90 <= polar_night_days <= 140, f"Polar night duration unexpected: {polar_night_days} days"
    print(f"  [PASS] Daylight Hours Range: {d_min:.1f}h to {d_max:.1f}h (Polar Night duration: {polar_night_days:.1f} days)")

    # Check 4: Solar power during polar night must be strictly zero
    polar_night_solar = df[df["daylight_hours"] == 0]["solar_available_kw"].max()
    assert polar_night_solar == 0.0, f"Solar power generated during polar night: {polar_night_solar} kW"
    solar_max = df["solar_available_kw"].max()
    print(f"  [PASS] Solar generation: 0.0 kW during polar night, peak summer: {solar_max:.1f} kW")

    # Check 5: Critical load is never zero and inversely correlates with outdoor temperature
    crit_min = df["load_critical_kw"].min()
    crit_max = df["load_critical_kw"].max()
    assert crit_min > 0.0, f"Critical load dropped to zero or below: {crit_min}"
    corr_temp_crit = np.corrcoef(df["outdoor_temp_c"], df["load_critical_kw"])[0, 1]
    assert corr_temp_crit < -0.85, f"Expected strong negative correlation with temp, got {corr_temp_crit}"
    print(f"  [PASS] Critical Load: min {crit_min:.1f} kW, max {crit_max:.1f} kW (Corr with temp: {corr_temp_crit:.3f})")

    # Check 6: Flexible load correlates positively with crew headcount
    flex_min = df["load_flexible_kw"].min()
    flex_max = df["load_flexible_kw"].max()
    corr_crew_flex = np.corrcoef(df["crew_headcount"], df["load_flexible_kw"])[0, 1]
    assert corr_crew_flex > 0.4, f"Expected positive correlation between crew and flexible load, got {corr_crew_flex}"
    print(f"  [PASS] Flexible Load: min {flex_min:.1f} kW, max {flex_max:.1f} kW (Corr with crew: {corr_crew_flex:.3f})")

    # Check 7: Crew headcount within range 15 to 80
    crew_min = df["crew_headcount"].min()
    crew_max = df["crew_headcount"].max()
    assert crew_min == 15 and crew_max == 80, f"Unexpected crew limits: {crew_min} to {crew_max}"
    print(f"  [PASS] Crew Headcount: {crew_min} (winter) to {crew_max} (summer)")

    # Check 8: Wind icing factor in range (0.05 to 1.0)
    ice_min = df["wind_icing_factor"].min()
    ice_max = df["wind_icing_factor"].max()
    assert 0.0 <= ice_min <= 1.0 and ice_max <= 1.0
    print(f"  [PASS] Wind Icing Factor: min derate {ice_min:.3f}, max nominal {ice_max:.3f}")

    # Check 9: Diesel fuel reserve starts at initial budget and depletes smoothly
    initial_fuel = df["diesel_fuel_reserve_liters"].iloc[0]
    final_fuel = df["diesel_fuel_reserve_liters"].iloc[-1]
    total_fuel_burned = df["hourly_fuel_consumption_liters"].sum()
    assert final_fuel < initial_fuel, "Fuel did not deplete over the year"
    assert final_fuel > 0, f"Fuel reserve ran completely out before resupply! Final: {final_fuel:.1f} L"
    print(f"  [PASS] Diesel Fuel Reserve: Initial {initial_fuel:,.0f} L -> End of Year {final_fuel:,.0f} L (Burned: {total_fuel_burned:,.0f} L)")

    print("--- All 9 Physics & Domain Integrity Checks Passed Successfully! ---\n")
    return all_passed


def main():
    print("=" * 65)
    print(" Polar Research Station AI Microgrid: Synthetic Data Generator ")
    print("=" * 65)

    config = StationConfig()
    print(f"Station Name:       {config.station_name}")
    print(f"Latitude / Elev:    {config.latitude}°S / {config.elevation_m:.0f}m")
    print(f"Simulation Year:    {config.year}")
    print(f"PV Capacity:        {config.pv_capacity_kw:.0f} kW")
    print(f"Wind Capacity:      {config.wind_rated_capacity_kw:.0f} kW")
    print(f"Initial Fuel:       {config.initial_diesel_reserve_liters:,.0f} Liters")
    print(f"Crew Range:         {config.min_crew_winter} (winter) - {config.max_crew_summer} (summer)")

    generator = PolarStationDataGenerator(config)
    print("\nGenerating 8,760 hourly records...")
    df = generator.generate()

    # Validate dataset
    validate_dataset(df)

    # Output directory
    output_dir = current_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "polar_station_energy_hourly.csv"

    # Export clean CSV
    df.to_csv(output_csv, index=False)
    print(f"Dataset successfully exported to:\n  {output_csv}")
    print(f"File size: {output_csv.stat().st_size / (1024 * 1024):.2f} MB")

    # Display sample preview
    print("\nFirst 3 rows:")
    print(df.iloc[:3][["timestamp", "outdoor_temp_c", "daylight_hours", "wind_speed_mps", "load_critical_kw", "load_flexible_kw", "solar_available_kw", "diesel_fuel_reserve_liters"]])

    print("\nMid-winter sample (July 15 12:00 - Polar Night):")
    mid_winter = df[df["timestamp"].str.startswith(f"{config.year}-07-15 12:00")]
    print(mid_winter[["timestamp", "outdoor_temp_c", "daylight_hours", "wind_speed_mps", "wind_icing_factor", "load_critical_kw", "solar_available_kw", "diesel_fuel_reserve_liters"]])


if __name__ == "__main__":
    main()
