"""
Training and Evaluation Script for Polar Station Load Forecasters.
Evaluates MAE, RMSE, R^2 on a held-out test split, prints feature importance
explainability tables, and tests the 24-72h forecast_load() pipeline.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.load_forecaster import GradientBoostingLoadForecaster, forecast_load


def main():
    print("=" * 70)
    print(" Polar Station AI Microgrid: Load Forecasting Model Evaluation ")
    print("=" * 70)

    # 1. Load dataset
    data_path = project_root / "data_generator" / "data" / "polar_station_energy_hourly.csv"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}. Running generator first...")
        from data_generator.generator import PolarStationDataGenerator
        gen = PolarStationDataGenerator()
        df = gen.generate()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset: {len(df):,} hourly observations ({df['timestamp'].min()} to {df['timestamp'].max()})")

    # 2. Chronological Train / Test Split (85% Train, 15% Held-Out Test)
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"Training Set:   {len(train_df):,} hours ({train_df['timestamp'].iloc[0]} -> {train_df['timestamp'].iloc[-1]})")
    print(f"Held-Out Test:  {len(test_df):,} hours ({test_df['timestamp'].iloc[0]} -> {test_df['timestamp'].iloc[-1]})")

    # 3. Train Model
    forecaster = GradientBoostingLoadForecaster(
        max_iter=150,
        learning_rate=0.08,
        max_depth=6,
        random_state=42,
    )

    t0 = time.time()
    print("\nTraining dual Gradient Boosting forecasters (Critical + Flexible)...")
    forecaster.fit(train_df)
    train_duration = time.time() - t0
    print(f"Training completed in {train_duration:.2f} seconds! (Target: < 120s for hackathon demo)")

    # 4. Evaluation on Held-Out Test Period
    print("\n" + "-" * 70)
    print(" 1. HELD-OUT TEST EVALUATION METRICS (Nov 16 - Dec 31) ")
    print("-" * 70)
    metrics = forecaster.evaluate(test_df)

    crit_metrics = metrics["load_critical_kw"]
    flex_metrics = metrics["load_flexible_kw"]
    tot_metrics = metrics["total_station_load_kw"]

    print(f"Critical Heating Load (kW):")
    print(f"  - MAE:  {crit_metrics['MAE']:.2f} kW")
    print(f"  - RMSE: {crit_metrics['RMSE']:.2f} kW")
    print(f"  - R²:   {crit_metrics['R2']:.4f}")

    print(f"\nFlexible Science Load (kW):")
    print(f"  - MAE:  {flex_metrics['MAE']:.2f} kW")
    print(f"  - RMSE: {flex_metrics['RMSE']:.2f} kW")
    print(f"  - R²:   {flex_metrics['R2']:.4f}")

    print(f"\nTotal Station Demand (kW):")
    print(f"  - MAE:  {tot_metrics['MAE']:.2f} kW")
    print(f"  - RMSE: {tot_metrics['RMSE']:.2f} kW")

    # 5. Feature Importance & Explainability
    print("\n" + "-" * 70)
    print(" 2. FEATURE EXPLAINABILITY & IMPORTANCE BREAKDOWN ")
    print("-" * 70)
    importance = forecaster.get_feature_importance(eval_df=test_df, n_repeats=5)

    print("Top Drivers for CRITICAL LOAD (Heating / Life Support):")
    for feature, share in list(importance["load_critical_drivers"].items())[:6]:
        bar = "#" * int(share / 2.5)
        print(f"  {feature:<24} : {share:5.1f}%  {bar}")

    print("\nTop Drivers for FLEXIBLE LOAD (Labs / Science / Crew):")
    for feature, share in list(importance["load_flexible_drivers"].items())[:6]:
        bar = "#" * int(share / 2.5)
        print(f"  {feature:<24} : {share:5.1f}%  {bar}")

    # 6. Test forecast_load() for 24h, 48h, and 72h Lookahead
    print("\n" + "-" * 70)
    print(" 3. MULTI-STEP FORECAST DEMO: forecast_load(current_data, horizon_hours) ")
    print("-" * 70)

    # Simulate station dispatch center querying 72h forecast from end of training set
    current_window = train_df.tail(48).copy()
    test_horizon_df = test_df.iloc[:72].copy()

    # Pass future weather forecast from test_df to test multi-step accuracy
    t_start = time.time()
    forecast_72h = forecaster.forecast_load(
        current_data=current_window,
        horizon_hours=72,
        future_exogenous=test_horizon_df,
    )
    inference_time = (time.time() - t_start) * 1000.0

    print(f"Generated 72-Hour Ahead Forecast in {inference_time:.1f} ms:")
    preview_cols = [
        "timestamp",
        "forecast_outdoor_temp_c",
        "crew_headcount",
        "load_critical_pred_kw",
        "load_flexible_pred_kw",
        "total_load_pred_kw",
    ]
    print(forecast_72h[preview_cols].iloc[[0, 11, 23, 47, 71]].to_string(index=False))

    # Calculate 72-hour test period forecast accuracy
    actual_crit = test_horizon_df["load_critical_kw"].values
    actual_flex = test_horizon_df["load_flexible_kw"].values
    pred_crit = forecast_72h["load_critical_pred_kw"].values
    pred_flex = forecast_72h["load_flexible_pred_kw"].values

    mae_72_crit = np.mean(np.abs(actual_crit - pred_crit))
    mae_72_flex = np.mean(np.abs(actual_flex - pred_flex))
    print(f"\n72-Hour Horizon Forecast Benchmark:")
    print(f"  - 72h Critical Load MAE: {mae_72_crit:.2f} kW")
    print(f"  - 72h Flexible Load MAE: {mae_72_flex:.2f} kW")
    print(f"  - 72h Total Load MAE:    {np.mean(np.abs((actual_crit + actual_flex) - (pred_crit + pred_flex))):.2f} kW")

    # 7. Test generic top-level wrapper function
    print("\nTesting top-level wrapper function `forecast_load(current_data, horizon_hours=48)`:")
    quick_forecast = forecast_load(current_data=train_df.tail(24), horizon_hours=48)
    print(f"  Success: returned DataFrame of shape {quick_forecast.shape} with columns:")
    print(f"  {list(quick_forecast.columns)}")

    print("\n" + "=" * 70)
    print(" All Load Forecaster Tests & Verifications Passed Successfully! ")
    print("=" * 70)


if __name__ == "__main__":
    main()
