"""
Configuration parameters for the Antarctic Polar Research Station.
Allows easy tuning of station sizing, climate, season lengths, equipment capacities, and fuel reserves.
"""

from dataclasses import dataclass


@dataclass
class StationConfig:
    """
    Tunable parameters for a fictional Antarctic polar research station
    calibrated against facilities like McMurdo, Amundsen-Scott, and Halley VI.
    """

    # Station Identity & Location
    station_name: str = "Boreas Polar Research Station"
    latitude: float = -78.5  # High Antarctic latitude (degrees South)
    longitude: float = 166.66  # Ross Island / McMurdo Sound sector (degrees East)
    elevation_m: float = 1200.0  # Elevation above sea level (meters)
    year: int = 2026
    random_seed: int = 42

    # Crew & Operational Capacity
    min_crew_winter: int = 15  # Skeleton overwintering crew (May - August)
    max_crew_summer: int = 80  # Full summer science & logistics complement (Dec - Jan)
    crew_summer_arrival_day: int = 295  # Late October (air bridge opens)
    crew_summer_departure_day: int = 60  # Late February / early March (winter closing)

    # Polar Day / Night & Seasonal Dates
    # Southern Hemisphere:
    # Polar night (~4 months): ~Day 120 (April 30) to ~Day 243 (August 31)
    # Polar day (~24h sun): ~Day 305 (Nov 1) to ~Day 31 (Jan 31)
    polar_night_start_day: int = 120
    polar_night_end_day: int = 243
    polar_day_start_day: int = 305
    polar_day_end_day: int = 31

    # Outdoor Temperature (°C)
    temp_summer_peak_c: float = -1.5  # Warmest summer afternoon peak (~0°C to -2°C)
    temp_winter_trough_c: float = -58.0  # Deep polar winter minimum (~ -55°C to -60°C)
    temp_synoptic_noise_std: float = 3.5  # Weather front variations (cold snaps, maritime lows)
    diurnal_temp_amplitude_c: float = 3.5  # Day/night temperature swing during shoulder seasons

    # Wind Parameters & Power Curve
    wind_mean_mps: float = 9.2  # Annual average wind speed (m/s)
    wind_k_shape: float = 2.05  # Weibull distribution shape parameter
    wind_cut_in_mps: float = 3.0  # Turbine start speed (m/s)
    wind_rated_mps: float = 12.0  # Turbine rated speed (m/s)
    wind_cut_out_mps: float = 25.0  # Turbine storm cutout speed (m/s)
    wind_rated_capacity_kw: float = 100.0  # Nameplate wind generation capacity (kW)

    # Blade Icing Derating Parameters
    # Severe blade icing occurs between -20°C and -1°C under high humidity
    icing_temp_min_c: float = -20.0
    icing_temp_max_c: float = -1.0
    icing_max_derate: float = 0.85  # Maximum wind power loss under severe icing (up to 85%)
    icing_recovery_rate: float = 0.12  # Rate at which blades shed ice when conditions improve

    # Solar Photovoltaic Array (kWp)
    pv_capacity_kw: float = 120.0  # Installed high-efficiency bifacial solar PV (kW)
    albedo_boost_factor: float = 1.25  # Snow ground reflection boost for bifacial PV
    cloud_attenuation_max: float = 0.70  # Max solar drop during dense blizzard / cloud cover

    # Electrical Loads (kW)
    # Critical Load: Life support, HVAC thermal loops, water generation plant, emergency comms
    load_critical_base_kw: float = 45.0  # Base critical load at 0°C (kW)
    load_critical_thermal_coeff: float = 1.55  # Additional kW per °C temperature drop below 0°C
    load_critical_noise_std: float = 2.2

    # Flexible Load: Science labs, lidars, ice drilling, workshops, non-essential amenities
    load_flexible_base_kw: float = 12.0  # Base lab idle load (kW)
    load_flexible_per_crew_kw: float = 0.60  # Additional kW per active crew member
    load_flexible_daylight_boost_kw: float = 18.0  # Active daytime science / logistics boost (kW)
    load_flexible_noise_std: float = 3.0

    # Diesel Generation & Fuel Logistics (Liters of Antarctic Grade AN-8 / Jet A-1)
    initial_diesel_reserve_liters: float = 360_000.0  # Annual resupply fuel budget (liters)
    diesel_efficiency_liters_per_kwh: float = 0.285  # Specific fuel consumption (L/kWh)
    diesel_standby_idle_liters_per_hour: float = 3.5  # Spinning reserve / generator baseline idle burn
    extreme_cold_boiler_liters_per_deg: float = 0.30  # Supplemental boiler burn when temp < -35°C
