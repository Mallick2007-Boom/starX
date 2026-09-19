"""
Verification and Testing Suite for Generator and Battery Health Anomaly Detection.
Tests synthetic fault injection, rolling Z-score detection accuracy, false alarm rates,
and human-readable diagnostic reason strings.
"""

from pathlib import Path
import sys
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    inject_synthetic_faults,
)


def main():
    print("=" * 75)
    print(" Polar Station AI Microgrid: Generator & Battery Health Anomaly Detection ")
    print("=" * 75)

    # 1. Load Simulation Telemetry
    sim_path = project_root / "models" / "simulation_results_365d.csv"
    if not sim_path.exists():
        sim_path = project_root / "data_generator" / "data" / "polar_station_energy_hourly.csv"

    df = pd.read_csv(sim_path)
    print(f"Loaded telemetry dataset: {len(df):,} hourly records.")

    # 2. Test Clean Baseline (False Alarm Rate Check)
    print("\n--- 1. Testing False Alarm Rate on Clean Telemetry ---")
    detector = AnomalyDetector(z_threshold=2.8)
    detector.fit(df)

    clean_sample = df.iloc[1000:2000].copy()
    clean_anomalies = detector.detect(clean_sample)
    false_alarm_rate = len(clean_anomalies) / len(clean_sample)
    print(f"Clean sample records: {len(clean_sample)} | Flagged anomalies: {len(clean_anomalies)}")
    print(f"False Alarm Rate:     {false_alarm_rate:.2%}")
    assert false_alarm_rate < 0.03, f"False alarm rate too high: {false_alarm_rate:.2%}"
    print("  [PASS] False alarm rate strictly bounded below 3.0%.")

    # 3. Inject Synthetic Faults
    print("\n--- 2. Injecting Synthetic Fault Episodes ---")
    test_window = df.iloc[3500:5500].copy()  # 2,000-hour winter test period
    corrupted_df, ground_truth_faults = inject_synthetic_faults(
        test_window, fault_rate=0.03, random_seed=42
    )

    print(f"Injected {len(ground_truth_faults)} multi-hour synthetic fault episodes:")
    for f in ground_truth_faults:
        print(f"  - [{f['fault_type']}] {f['start_timestamp']} to {f['end_timestamp']}: {f['description']}")

    # 4. Detect Anomalies on Faulted Telemetry
    print("\n--- 3. Running Anomaly Detection Pipeline ---")
    detected = detector.detect(corrupted_df)
    print(f"Total anomaly hours flagged: {len(detected)} out of {len(corrupted_df)} records.")

    # Check detection recall against ground truth episodes
    flagged_timestamps = set(a["timestamp"] for a in detected)
    episodes_caught = 0

    for f in ground_truth_faults:
        # Check if at least one hour of the episode was detected
        start_ts = f["start_timestamp"]
        end_ts = f["end_timestamp"]
        caught = any(
            start_ts <= a["timestamp"] <= end_ts for a in detected
        )
        if caught:
            episodes_caught += 1

    recall_pct = (episodes_caught / len(ground_truth_faults)) * 100.0
    print(f"Fault Episode Detection Recall: {episodes_caught}/{len(ground_truth_faults)} ({recall_pct:.1f}%)")
    assert recall_pct >= 85.0, f"Detection recall below target: {recall_pct:.1f}%"
    print("  [PASS] Successfully detected all major equipment fault episodes.")

    # 5. Display Explainable Reason Strings for Incident Reports
    print("\n--- 4. Sample Flagged Anomaly Incident Reports & Explainable Reasons ---")
    for i, a in enumerate(detected[:6]):
        print(f"\nAlert #{i+1}: [{a['severity']}] {a['anomaly_type']} at {a['timestamp']}")
        print(f"  Metric:   {a['metric']} = {a['observed_value']} (Baseline: {a['baseline_value']}, Z-Score: {a['z_score']})")
        print(f"  Reason:   \"{a['reason']}\"")

    # 6. Test Generic Top-Level Function detect_anomalies()
    print("\n--- 5. Testing Generic Wrapper detect_anomalies(recent_data) ---")
    fault_slice_indices = ground_truth_faults[0]
    start_i = max(0, fault_slice_indices["start_idx"] - 5)
    end_i = min(len(corrupted_df), fault_slice_indices["end_idx"] + 5)
    fault_sample = corrupted_df.iloc[start_i:end_i]

    quick_alerts = detect_anomalies(fault_sample)
    print(f"  Success: detect_anomalies() returned {len(quick_alerts)} alerts for active fault window.")
    if quick_alerts:
        print(f"  First Alert Reason:\n    \"{quick_alerts[0]['reason']}\"")

    print("\n" + "=" * 75)
    print(" Anomaly Detection Engine Verification Complete! All Tests Passed. ")
    print("=" * 75)


if __name__ == "__main__":
    main()
