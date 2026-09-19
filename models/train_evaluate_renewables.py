"""
Evaluation and Validation Suite for Renewable Generation Forecaster.
Tests solar polar night suppression, wind aerodynamic curve, physics-informed
icing derating heuristics, and the forecast_renewables() 24h-72h pipeline.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.renewables_forecaster import (
    RenewablesForecaster,
    forecast_renewables,
    icing_loss_factor,
)


def main():
    print("=" * 70)
    print(" Polar Station AI Microgrid: Renewable Forecasting & Icing Model ")
    print("=" * 70)

    # 1. Test Icing Heuristic Sensitivity
    print("\n--- 1. Testing Physics-Informed Icing Derating Heuristic ---")
    temps = [-50.0, -35.0, -15.0, -8.0, -4.0, -0.5, 2.0]
    humidity = 90.0  # High freezing fog humidity
    wind = 10.0      # Typical moderate gale

    print(f"Icing factor response at RH={humidity:.0f}%, Wind={wind:.0f} m/s across temperatures:")
    for t in temps:
        factor = icing_loss_factor(t, humidity, wind)
        derate_pct = (1.0 - factor) * 100.0
        bar = "#" * int(derate_pct / 3.0)
        print(f"  Temp: {t:5.1f}°C  ->  Icing Factor: {factor:.3f} (Power Loss: {derate_pct:4.1f}%) {bar}")

    # Assertions on icing behavior
    assert icing_loss_factor(2.0, 90.0, 10.0) == 1.0, "Above 0°C should have no icing"
    assert icing_loss_factor(-8.0, 90.0, 10.0) < 0.40, "Severe icing should occur at -8°C with high humidity"
    assert icing_loss_factor(-50.0, 40.0, 10.0) > 0.95, "Extreme cold dry air should have minimal icing"
    print("  [PASS] All aerodynamic icing heuristic assertions verified.")

    # 2. Load Dataset & Train / Test Split
    data_path = project_root / "data_generator" / "data" / "polar_station_energy_hourly.csv"
    df = pd.read_csv(data_path)

    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    forecaster = RenewablesForecaster()
    forecaster.fit(train_df)

    # 3. Solar Forecast Evaluation
    print("\n--- 2. Solar Generation Forecast Evaluation ---")
    test_solar_pred = forecaster.predict_solar_kw(test_df)
    test_solar_actual = test_df["solar_available_kw"].values

    mae_solar = mean_absolute_error(test_solar_actual, test_solar_pred)
    rmse_solar = np.sqrt(mean_squared_error(test_solar_actual, test_solar_pred))
    r2_solar = r2_score(test_solar_actual, test_solar_pred)

    print(f"Held-Out Test Period (Nov 7 - Dec 31):")
    print(f"  - Solar MAE:  {mae_solar:.2f} kW")
    print(f"  - Solar RMSE: {rmse_solar:.2f} kW")
    print(f"  - Solar R²:   {r2_solar:.4f}")

    # Mid-winter polar night verification
    mid_winter_mask = (df["timestamp"].str.startswith("2026-07-"))
    winter_pred_solar = forecaster.predict_solar_kw(df[mid_winter_mask])
    assert np.all(winter_pred_solar == 0.0), "Solar forecast must be strictly 0.0 during polar night"
    print(f"  [PASS] Polar night constraint: strictly 0.0 kW throughout entire month of July ({len(winter_pred_solar)} hours).")

    # 4. Wind Forecast & Icing Derating Evaluation
    print("\n--- 3. Wind Power Forecast (Post-Icing) Evaluation ---")
    wind_speed = test_df["wind_speed_mps"].values
    temp = test_df["outdoor_temp_c"].values
    humidity = test_df["relative_humidity_pct"].values if "relative_humidity_pct" in test_df.columns else None

    raw_wind_pred, icing_pred, effective_wind_pred = forecaster.predict_wind_kw(
        wind_speed, temp, humidity
    )

    actual_raw_wind = test_df["wind_available_kw"].values
    actual_effective_wind = test_df["wind_effective_kw"].values

    mae_raw_wind = mean_absolute_error(actual_raw_wind, raw_wind_pred)
    mae_effective_wind = mean_absolute_error(actual_effective_wind, effective_wind_pred)
    rmse_effective_wind = np.sqrt(mean_squared_error(actual_effective_wind, effective_wind_pred))

    print(f"Wind Forecast Metrics:")
    print(f"  - Raw Wind Potential MAE:    {mae_raw_wind:.2f} kW")
    print(f"  - Post-Icing Effective MAE:  {mae_effective_wind:.2f} kW")
    print(f"  - Post-Icing Effective RMSE: {rmse_effective_wind:.2f} kW")

    # 5. Multi-Step Forecast Demonstration for Optimizer (Module C)
    print("\n--- 4. Multi-Step 72-Hour Ahead Forecast Demonstration ---")
    current_data = train_df.tail(48).copy()
    future_weather = test_df.iloc[:72].copy()

    forecast_72h = forecaster.forecast_renewables(
        current_data=current_data,
        horizon_hours=72,
        future_weather=future_weather,
    )

    print(f"Generated 72-Hour Renewable Generation Forecast (shape: {forecast_72h.shape}):")
    preview_cols = [
        "timestamp",
        "forecast_temp_c",
        "forecast_wind_speed_mps",
        "solar_kw",
        "wind_raw_kw",
        "icing_factor",
        "wind_kw",
        "total_renewable_kw",
    ]
    print(forecast_72h[preview_cols].iloc[[0, 12, 24, 36, 48, 71]].to_string(index=False))

    # 6. Test convenience wrapper function
    print("\n--- 5. Testing Generic Wrapper forecast_renewables() ---")
    quick_res = forecast_renewables(current_data=train_df.tail(24), horizon_hours=48)
    assert len(quick_res) == 48
    assert "solar_kw" in quick_res.columns
    assert "wind_kw" in quick_res.columns
    assert "icing_factor" in quick_res.columns
    print("  [PASS] forecast_renewables(current_data, horizon_hours=48) successfully returned:")
    print(f"  Columns: {list(quick_res.columns)}")
    print(f"  Avg Solar: {quick_res['solar_kw'].mean():.1f} kW | Avg Wind (post-icing): {quick_res['wind_kw'].mean():.1f} kW")

    print("\n" + "=" * 70)
    print(" All Renewable Forecasting & Icing Tests Passed Successfully! ")
    print(" Ready to feed Module C (Microgrid Fuel Optimizer). ")
    print("=" * 70)


if __name__ == "__main__":
    main()
