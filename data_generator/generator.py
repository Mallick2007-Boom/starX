"""
Polar Research Station Synthetic Data Generator.
Simulates 1 year of hourly microgrid and environmental telemetry for an isolated
Antarctic research station with high physical realism.
"""

import math
from typing import Optional
import numpy as np
import pandas as pd

try:
    from .config import StationConfig
except (ImportError, ValueError):
    from config import StationConfig


class PolarStationDataGenerator:
    """
    Generates synthetic time-series data for a polar research station microgrid,
    accounting for Antarctic solar geometry, temperature cycles, katabatic winds,
    turbine icing derating, crew schedules, loads, and fuel depletion.
    """

    def __init__(self, config: Optional[StationConfig] = None):
        self.config = config or StationConfig()
        np.random.seed(self.config.random_seed)

    def _generate_time_index(self) -> pd.DatetimeIndex:
        """Create 1 full year of hourly timestamps."""
        start_date = f"{self.config.year}-01-01 00:00:00"
        # 365 days = 8760 hours (non-leap) or 8784 (leap)
        end_date = f"{self.config.year}-12-31 23:00:00"
        return pd.date_range(start=start_date, end=end_date, freq="h")

    def _calculate_solar_geometry(self, time_index: pd.DatetimeIndex):
        """
        Calculate solar declination, solar elevation angle, and daily daylight hours
        for high southern polar latitude.
        """
        day_of_year = time_index.dayofyear.values  # 1 to 365
        hour_of_day = time_index.hour.values  # 0 to 23

        # Solar declination angle in radians (Cooper's approximation)
        # Declination is negative in Southern Hemisphere summer (Dec/Jan)
        # -23.44 deg around Dec 21, +23.44 deg around June 21
        declination_deg = 23.44 * np.sin(2 * np.pi * (284 + day_of_year) / 365.25)
        declination_rad = np.radians(declination_deg)
        lat_rad = np.radians(self.config.latitude)

        # Solar hour angle (omega): 0 at solar noon (12:00), 15 deg per hour
        solar_noon_hour = 12.0
        hour_angle_deg = (hour_of_day - solar_noon_hour) * 15.0
        hour_angle_rad = np.radians(hour_angle_deg)

        # Sin(elevation) = sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(omega)
        sin_elevation = (
            np.sin(lat_rad) * np.sin(declination_rad)
            + np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad)
        )
        sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
        elevation_deg = np.degrees(np.arcsin(sin_elevation))

        # Analytical daily daylight hours calculation:
        # cos(omega_s) = -tan(lat)*tan(decl)
        cos_omega_s = -np.tan(lat_rad) * np.tan(declination_rad)
        # For high polar latitudes:
        # cos_omega_s >= 1.0: sun never rises -> Polar Night (0 hours)
        # cos_omega_s <= -1.0: sun never sets -> Polar Day / Midnight Sun (24 hours)
        # -1.0 < cos_omega_s < 1.0: daily sunrise/sunset transitions
        daily_daylight = np.zeros_like(cos_omega_s)
        polar_night_mask = cos_omega_s >= 1.0
        polar_day_mask = cos_omega_s <= -1.0
        transitional_mask = (~polar_night_mask) & (~polar_day_mask)

        daily_daylight[polar_night_mask] = 0.0
        daily_daylight[polar_day_mask] = 24.0
        daily_daylight[transitional_mask] = (
            2.0
            * np.degrees(np.arccos(np.clip(cos_omega_s[transitional_mask], -1.0, 1.0)))
            / 15.0
        )

        return elevation_deg, daily_daylight, polar_night_mask, polar_day_mask

    def _generate_temperature(
        self,
        time_index: pd.DatetimeIndex,
        polar_night_mask: np.ndarray,
        polar_day_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Generate outdoor temperature in °C.
        Seasonal sine wave (-60°C to 0°C) with diurnal swings during shoulder seasons
        and stochastic autoregressive weather fronts (blizzards / warm marine air incursions).
        """
        n_steps = len(time_index)
        day_of_year = time_index.dayofyear.values
        hour_of_day = time_index.hour.values

        # Southern Hemisphere seasonal cycle:
        # Peak warmth around Day 15 (mid-January), trough around Day 197 (mid-July)
        # Shift so cosine has maximum at day 15
        temp_mean = (
            self.config.temp_summer_peak_c + self.config.temp_winter_trough_c
        ) / 2.0
        temp_amp = (
            self.config.temp_summer_peak_c - self.config.temp_winter_trough_c
        ) / 2.0
        seasonal_temp = temp_mean + temp_amp * np.cos(
            2 * np.pi * (day_of_year - 15) / 365.0
        )

        # Diurnal temperature cycle:
        # Present during transitional shoulder seasons (sunrise/sunset), suppressed during polar night & polar day
        transitional_factor = 1.0 - (
            polar_night_mask.astype(float) * 0.85 + polar_day_mask.astype(float) * 0.70
        )
        diurnal_temp = (
            self.config.diurnal_temp_amplitude_c
            * np.sin(2 * np.pi * (hour_of_day - 9) / 24.0)
            * transitional_factor
        )

        # Autoregressive AR(1) synoptic weather fronts (blizzards, cold snaps, atmospheric rivers)
        # Persistence parameter phi = 0.985 corresponds to ~3-5 day weather front persistence
        phi = 0.985
        innovation_scale = self.config.temp_synoptic_noise_std * np.sqrt(1 - phi**2)
        white_noise = np.random.normal(0, innovation_scale, size=n_steps)
        synoptic_temp = np.zeros(n_steps)
        for t in range(1, n_steps):
            synoptic_temp[t] = phi * synoptic_temp[t - 1] + white_noise[t]

        # Combine
        temp_c = seasonal_temp + diurnal_temp + synoptic_temp

        # Hard physical clamp to polar station bounds (-62°C to +3°C)
        temp_c = np.clip(temp_c, -62.0, 2.5)
        return temp_c

    def _generate_wind_and_icing(
        self, time_index: pd.DatetimeIndex, temp_c: np.ndarray
    ):
        """
        Generate wind speed (m/s) and wind icing derating factor (0.0 to 1.0)
        based on temperature, atmospheric moisture, and storm systems.
        """
        n_steps = len(time_index)

        # Baseline wind speed from AR(1) coupled Weibull process
        # Mean ~ 9.2 m/s, gusting to 25-35 m/s during storms
        scale = self.config.wind_mean_mps / math.gamma(
            1 + 1 / self.config.wind_k_shape
        )
        base_weibull = np.random.weibull(self.config.wind_k_shape, size=n_steps) * scale

        # Smooth wind persistence using an exponential moving average
        wind_speed_mps = pd.Series(base_weibull).ewm(span=6).mean().to_numpy(copy=True)

        # Katabatic blizzard events: generate periodic intense gale events
        # 12-18 gale events per year lasting 12-36 hours
        num_storms = 16
        storm_starts = np.random.choice(n_steps - 48, size=num_storms, replace=False)
        for start in storm_starts:
            duration = np.random.randint(14, 38)
            storm_envelope = np.sin(np.pi * np.linspace(0, 1, duration))
            wind_speed_mps[start : start + duration] += (
                storm_envelope * np.random.uniform(10.0, 18.0)
            )

        wind_speed_mps = np.clip(wind_speed_mps, 0.2, 36.0)

        # Relative humidity model:
        # Warmer polar air (-15°C to 0°C) carries maritime moisture (high RH 75-95%)
        # Deep interior winter air (-50°C) is extremely dry (RH 35-55%)
        temp_normalized = (temp_c - (-60.0)) / 60.0  # 0 (at -60C) to 1 (at 0C)
        base_humidity = 40.0 + 45.0 * temp_normalized
        humidity_noise = np.random.normal(0, 5.0, size=n_steps)
        relative_humidity_pct = np.clip(base_humidity + humidity_noise, 25.0, 98.0)

        # Wind Icing Factor:
        # Icing occurs when -20°C <= temp <= -1°C and humidity >= 70%
        # Liquid water content in freezing fog coats blades with rime ice,
        # degrading aerodynamic profile.
        in_icing_temp_range = (temp_c >= self.config.icing_temp_min_c) & (
            temp_c <= self.config.icing_temp_max_c
        )
        high_humidity = relative_humidity_pct >= 70.0
        icing_active_conditions = in_icing_temp_range & high_humidity

        icing_severity = np.zeros(n_steps)
        current_ice = 0.0

        for t in range(n_steps):
            if icing_active_conditions[t]:
                # Ice accrues proportional to humidity and wind exposure
                accretion_rate = (
                    0.08
                    * (relative_humidity_pct[t] / 100.0)
                    * (wind_speed_mps[t] / 10.0)
                )
                current_ice = min(1.0, current_ice + accretion_rate)
            else:
                # Ice sheds or sublimates
                current_ice = max(
                    0.0, current_ice - self.config.icing_recovery_rate
                )
            icing_severity[t] = current_ice

        # Derating factor: 1.0 (clean blades) down to (1 - max_derate)
        wind_icing_factor = 1.0 - (self.config.icing_max_derate * icing_severity)
        wind_icing_factor = np.clip(wind_icing_factor, 0.05, 1.0)

        # Raw wind power potential (before icing derating) based on turbine power curve
        v_in = self.config.wind_cut_in_mps
        v_rated = self.config.wind_rated_mps
        v_out = self.config.wind_cut_out_mps
        p_rated = self.config.wind_rated_capacity_kw

        wind_available_kw = np.zeros(n_steps)
        # Below cut-in or above cut-out: 0 kW
        cubic_mask = (wind_speed_mps >= v_in) & (wind_speed_mps < v_rated)
        rated_mask = (wind_speed_mps >= v_rated) & (wind_speed_mps <= v_out)

        wind_available_kw[cubic_mask] = p_rated * (
            (wind_speed_mps[cubic_mask] - v_in) / (v_rated - v_in)
        ) ** 3
        wind_available_kw[rated_mask] = p_rated

        return (
            wind_speed_mps,
            wind_icing_factor,
            wind_available_kw,
            relative_humidity_pct,
        )

    def _generate_crew_headcount(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        """
        Generate station crew headcount:
        15 in polar winter (skeleton crew) up to 80 in peak summer (science teams).
        Smooth transition during flight windows (October arrival, February departure).
        """
        day_of_year = time_index.dayofyear.values
        crew = np.full(len(time_index), float(self.config.min_crew_winter))

        # Peak summer: Nov 1 to Jan 31 (days 305 to 365, and days 1 to 31)
        summer_peak_mask = (day_of_year >= 320) | (day_of_year <= 25)
        crew[summer_peak_mask] = float(self.config.max_crew_summer)

        # Summer ramp-up: Day 295 (late Oct) to Day 320 (mid Nov)
        ramp_up_mask = (day_of_year >= 295) & (day_of_year < 320)
        t_up = (day_of_year[ramp_up_mask] - 295) / 25.0
        crew[ramp_up_mask] = self.config.min_crew_winter + t_up * (
            self.config.max_crew_summer - self.config.min_crew_winter
        )

        # Summer departure ramp-down: Day 25 (late Jan) to Day 60 (late Feb)
        ramp_down_mask = (day_of_year > 25) & (day_of_year <= 60)
        t_down = (60 - day_of_year[ramp_down_mask]) / 35.0
        crew[ramp_down_mask] = self.config.min_crew_winter + t_down * (
            self.config.max_crew_summer - self.config.min_crew_winter
        )

        # Crew headcount rounded to integer
        return np.round(crew).astype(int)

    def _generate_solar_power(
        self,
        elevation_deg: np.ndarray,
        polar_night_mask: np.ndarray,
        wind_speed_mps: np.ndarray,
    ) -> np.ndarray:
        """
        Generate solar power in kW.
        Strictly 0 during polar night or when sun elevation <= 0.
        Follows solar elevation angle with bifacial snow albedo boost and blizzard cloud attenuation.
        """
        n_steps = len(elevation_deg)
        solar_available_kw = np.zeros(n_steps)

        # Sun above horizon and not polar night
        sun_up_mask = (elevation_deg > 0.0) & (~polar_night_mask)

        # Cloud and blizzard attenuation correlated with wind speed
        # High wind / storm days produce blowing snow and thick overcast
        blizzard_cloud_factor = 1.0 - np.clip(
            (wind_speed_mps - 10.0) / 20.0, 0.0, self.config.cloud_attenuation_max
        )

        # Direct irradiance model: sin(elevation) shaped with albedo boost
        sin_elev = np.sin(np.radians(elevation_deg[sun_up_mask]))
        raw_solar_kw = (
            self.config.pv_capacity_kw
            * (sin_elev**0.95)
            * self.config.albedo_boost_factor
            * blizzard_cloud_factor[sun_up_mask]
        )

        # Clip to array capacity
        solar_available_kw[sun_up_mask] = np.clip(
            raw_solar_kw, 0.0, self.config.pv_capacity_kw
        )

        # Guaranteed exact zero during polar night
        solar_available_kw[polar_night_mask] = 0.0
        return solar_available_kw

    def _generate_loads(
        self,
        temp_c: np.ndarray,
        crew_headcount: np.ndarray,
        time_index: pd.DatetimeIndex,
        elevation_deg: np.ndarray,
    ):
        """
        Generate critical load (kW) and flexible load (kW).
        - load_critical_kw: Life support, HVAC heating loops, water production.
          Never zero. Inversely correlates with outdoor temperature.
        - load_flexible_kw: Science labs, lidars, drill rigs, amenities.
          Correlates with crew headcount and diurnal activity/daylight.
        """
        n_steps = len(temp_c)
        hour_of_day = time_index.hour.values

        # 1. Critical Load (kW)
        # Heat loss is proportional to temperature delta: (T_indoor - T_outdoor)
        # Assuming indoor station setpoint ~ +18°C.
        # As T_outdoor drops from 0°C to -60°C, thermal loop demand ramps up.
        temp_deficit = np.maximum(0.0, -temp_c)  # 0 at 0°C, 60 at -60°C
        thermal_heating_kw = (
            self.config.load_critical_thermal_coeff * temp_deficit
        )

        critical_noise = np.random.normal(
            0, self.config.load_critical_noise_std, size=n_steps
        )
        load_critical_kw = (
            self.config.load_critical_base_kw
            + thermal_heating_kw
            + critical_noise
        )
        # Critical load must NEVER be zero or below minimum safety floor
        load_critical_kw = np.maximum(35.0, load_critical_kw)

        # 2. Flexible Load (kW)
        # Correlates with crew size and daily awake/work schedule
        crew_load = self.config.load_flexible_per_crew_kw * crew_headcount

        # Diurnal human work cycle: peak between 08:00 and 19:00
        daily_work_cycle = 0.5 * (
            1.0 + np.sin(2 * np.pi * (hour_of_day - 6) / 24.0)
        )
        daily_work_cycle = np.clip(daily_work_cycle, 0.1, 1.0)

        # Active daylight boost (solar elevation > 0 allows outdoor survey & remote sensor runs)
        daylight_active = np.clip(elevation_deg / 25.0, 0.0, 1.0)

        flexible_noise = np.random.normal(
            0, self.config.load_flexible_noise_std, size=n_steps
        )
        load_flexible_kw = (
            self.config.load_flexible_base_kw
            + crew_load * daily_work_cycle
            + self.config.load_flexible_daylight_boost_kw * daylight_active
            + flexible_noise
        )
        load_flexible_kw = np.maximum(8.0, load_flexible_kw)

        return load_critical_kw, load_flexible_kw

    def _simulate_diesel_reserve(
        self,
        load_critical_kw: np.ndarray,
        load_flexible_kw: np.ndarray,
        solar_available_kw: np.ndarray,
        wind_available_kw: np.ndarray,
        wind_icing_factor: np.ndarray,
        temp_c: np.ndarray,
    ):
        """
        Simulate hourly diesel fuel consumption and reserve depletion.
        Tracks net load after effective wind and solar generation,
        accounting for generator electrical efficiency and extreme cold heating boilers.
        """
        n_steps = len(load_critical_kw)
        diesel_fuel_reserve = np.zeros(n_steps)
        hourly_fuel_burn = np.zeros(n_steps)
        diesel_gen_kw = np.zeros(n_steps)

        # Effective wind power after icing derating
        wind_effective_kw = wind_available_kw * wind_icing_factor

        total_demand_kw = load_critical_kw + load_flexible_kw
        total_renewable_kw = wind_effective_kw + solar_available_kw

        # Microgrid dispatch: renewables first, diesel covers remaining deficit
        net_deficit_kw = np.maximum(0.0, total_demand_kw - total_renewable_kw)
        diesel_gen_kw[:] = net_deficit_kw

        # Fuel consumption model:
        # Electrical generation + baseline spinning reserve + extreme cold boiler (< -35°C)
        current_reserve = self.config.initial_diesel_reserve_liters

        for t in range(n_steps):
            # Electrical burn
            gen_burn = (
                diesel_gen_kw[t] * self.config.diesel_efficiency_liters_per_kwh
            )
            standby_burn = self.config.diesel_standby_idle_liters_per_hour

            # Supplemental thermal boiler burn when ambient temp < -35°C
            # (generator waste heat recovery cannot meet full hydronic demand)
            boiler_burn = 0.0
            if temp_c[t] < -35.0:
                boiler_burn = (
                    -35.0 - temp_c[t]
                ) * self.config.extreme_cold_boiler_liters_per_deg

            total_burn = gen_burn + standby_burn + boiler_burn
            hourly_fuel_burn[t] = total_burn
            current_reserve = max(0.0, current_reserve - total_burn)
            diesel_fuel_reserve[t] = current_reserve

        return diesel_fuel_reserve, hourly_fuel_burn, wind_effective_kw, diesel_gen_kw

    def generate(self) -> pd.DataFrame:
        """
        Execute synthetic data generation and return a clean pandas DataFrame
        with all requested polar station telemetry.
        """
        time_index = self._generate_time_index()

        # Solar geometry & polar cycles
        elevation_deg, daily_daylight, polar_night_mask, polar_day_mask = (
            self._calculate_solar_geometry(time_index)
        )

        # Environmental conditions
        temp_c = self._generate_temperature(
            time_index, polar_night_mask, polar_day_mask
        )
        (
            wind_speed_mps,
            wind_icing_factor,
            wind_available_kw,
            relative_humidity_pct,
        ) = self._generate_wind_and_icing(time_index, temp_c)

        # Station operations
        crew_headcount = self._generate_crew_headcount(time_index)

        # Renewable generation
        solar_available_kw = self._generate_solar_power(
            elevation_deg, polar_night_mask, wind_speed_mps
        )

        # Loads
        load_critical_kw, load_flexible_kw = self._generate_loads(
            temp_c, crew_headcount, time_index, elevation_deg
        )

        # Diesel fuel reserve and dispatch
        (
            diesel_fuel_reserve_liters,
            hourly_fuel_burn,
            wind_effective_kw,
            diesel_gen_kw,
        ) = self._simulate_diesel_reserve(
            load_critical_kw,
            load_flexible_kw,
            solar_available_kw,
            wind_available_kw,
            wind_icing_factor,
            temp_c,
        )

        # Assemble DataFrame with user requested columns
        df = pd.DataFrame(
            {
                "timestamp": time_index.strftime("%Y-%m-%d %H:%M:%S"),
                "outdoor_temp_c": np.round(temp_c, 2),
                "daylight_hours": np.round(daily_daylight, 2),
                "wind_speed_mps": np.round(wind_speed_mps, 2),
                "wind_icing_factor": np.round(wind_icing_factor, 4),
                "crew_headcount": crew_headcount,
                "load_critical_kw": np.round(load_critical_kw, 2),
                "load_flexible_kw": np.round(load_flexible_kw, 2),
                "solar_available_kw": np.round(solar_available_kw, 2),
                "wind_available_kw": np.round(wind_available_kw, 2),
                "diesel_fuel_reserve_liters": np.round(
                    diesel_fuel_reserve_liters, 2
                ),
                # Additional high-value microgrid metrics for AI forecasting & optimization
                "wind_effective_kw": np.round(wind_effective_kw, 2),
                "diesel_generation_kw": np.round(diesel_gen_kw, 2),
                "hourly_fuel_consumption_liters": np.round(
                    hourly_fuel_burn, 2
                ),
                "sun_elevation_deg": np.round(elevation_deg, 2),
                "relative_humidity_pct": np.round(relative_humidity_pct, 1),
            }
        )

        return df


if __name__ == "__main__":
    from pathlib import Path

    print("=" * 65)
    print(" Polar Research Station Synthetic Telemetry Generator ")
    print("=" * 65)
    config = StationConfig()
    generator = PolarStationDataGenerator(config)
    print(f"Generating 1 year ({config.year}) of hourly microgrid telemetry...")
    df = generator.generate()

    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "polar_station_energy_hourly.csv"
    df.to_csv(out_file, index=False)
    print(f"Successfully generated {len(df):,} hourly records.")
    print(f"Clean CSV saved to: {out_file}")
    print("\nDataset Preview (Summer vs Winter):")
    print(
        df.iloc[[0, 4700]][
            [
                "timestamp",
                "outdoor_temp_c",
                "daylight_hours",
                "wind_speed_mps",
                "wind_icing_factor",
                "crew_headcount",
                "load_critical_kw",
                "load_flexible_kw",
                "solar_available_kw",
                "wind_available_kw",
                "diesel_fuel_reserve_liters",
            ]
        ].to_string()
    )

