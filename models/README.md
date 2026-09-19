# AI Forecasting & Microgrid Optimization Models

This module hosts the predictive intelligence, automated dispatch optimization, and asset health monitoring models for the polar research station smart energy management system.

## Module Structure

```
models/
├── __init__.py                  # Package exports (forecast_load, forecast_renewables, optimize_hour, detect_anomalies)
├── base.py                      # Abstract base class BaseLoadForecaster (LSTM-ready modular architecture)
├── features.py                  # Physics-informed thermal, crew schedule, and cyclical feature engineering
├── load_forecaster.py           # Dual HistGradientBoostingRegressor model for critical & flexible loads
├── renewables_forecaster.py     # Solar & wind forecasting with physics-informed aerodynamic blade icing
├── optimizer.py                 # MPC / LP rolling-horizon microgrid dispatch optimizer & simulation engine
├── anomaly_detector.py          # Generator & battery health anomaly detector with explainable alerts
├── train_evaluate.py            # Train/eval benchmark script for load forecaster
├── train_evaluate_renewables.py # Train/eval benchmark script for renewables & icing heuristic
├── run_optimization_sim.py      # 365-day rolling simulation & constraint verification script
├── test_anomaly_detector.py     # Anomaly detector validation & fault injection benchmark script
├── simulation_results_365d.csv  # 8,760-hour full-year hourly optimization logs
└── README.md                    # Documentation and API reference
```

---

## 1. Load Forecaster (`load_forecaster.py`)

A high-performance forecasting engine predicting station critical life-support heating load and flexible science/operational load for 24- to 72-hour horizons.

### Accuracy Benchmarks (Held-Out Test Split)
| Target | MAE (kW) | RMSE (kW) | $R^2$ Score | Key Physical Drivers |
|---|---|---|---|---|
| **`load_critical_kw`** | **1.84 kW** | **2.32 kW** | **0.9526** | Heating Deficit (53.9%), Ambient Temp (28.8%), Wind Chill (16.1%) |
| **`load_flexible_kw`** | **2.97 kW** | **3.98 kW** | **0.9609** | Solar Availability / Daylight (64.2%), Crew Schedule (19.6%), Hour (11.0%) |
| **`total_load_kw`** | **3.54 kW** | **4.59 kW** | **0.9640** | Combined microgrid demand |

---

## 2. Renewable Generation Forecaster (`renewables_forecaster.py`)

Predicts available solar PV power and effective wind turbine generation (post-aerodynamic icing derating) for 24h to 72h horizons to feed the microgrid fuel optimizer.

- **Polar Night Solar**: Strictly $0.0$ kW during polar night (`daylight_hours == 0` or sun below horizon). Validated across all 744 hours of July.
- **Physics-Informed Icing Heuristic (`icing_loss_factor`)**:
  - Peak rime icing susceptibility occurs between $-20^\circ\text{C}$ and $-1^\circ\text{C}$ with relative humidity $\ge 60\%$ (IEA Wind Task 19 / Ross Island research).
  - Power loss up to $70\% - 85\%$ in freezing fog.
  - Below $-25^\circ\text{C}$ to $-30^\circ\text{C}$, atmospheric dryness causes moisture to exist as dry ice crystals that bounce off blades, preserving aerodynamic efficiency.

---

## 3. Microgrid Optimization Engine (`optimizer.py`)

Model Predictive Control (MPC) and rule-based rolling-horizon optimizer deciding hourly power allocation across diesel generators, battery energy storage (BESS), and renewables.

### Hard Constraints
1. **Life Support Security**: `load_critical_kw` must ALWAYS be met ($100\%$ uncurtailed across all 8,760 hours).
2. **Fuel Depletion Guard**: `diesel_fuel_reserve_liters` must NEVER drop to zero before annual resupply.
3. **Safety Fuel Buffer**: Configurable safety buffer (e.g. minimum $10\%$ safety reserve above remaining-days-needed).
4. **Battery Operating Bounds**: Strict capacity ($300$ kWh), charge/discharge rates ($75$ kW), and sub-zero temperature state-of-charge floor ($20\%$ SoC minimum) to prevent cell degradation in polar cold.

### 365-Day Simulation Benchmarks
- **Simulation Runtime**: **2.21 seconds** for 8,760 hourly timesteps.
- **Critical Load Survival Rate**: **100.0%** (zero hours of critical load curtailment).
- **Annual Fuel Saved**: **18,153 Liters** (9.2% reduction vs baseline).
- **Renewable Energy Penetration**: **49.3%** across the full year.
- **End-of-Year Safety Reserve**: **181,570 Liters** remaining before resupply ship.

---

## 4. Anomaly Detection Engine (`anomaly_detector.py`)

Lightweight asset health monitoring engine for diesel generators and battery storage under harsh Antarctic operating conditions.

### Fault Categories Detected
1. **Generator Efficiency Drift**: Injector fouling or polar fuel wax crystallization causing $+20\%$ to $+40\%$ excessive fuel consumption per kWh.
2. **Battery Cold Degradation**: Sudden capacity drop or failure to accept charge when ambient temperature falls below $-20^\circ\text{C}$ due to BESS thermal enclosure heating failure.
3. **Idle Fuel Leak / Aux Boiler Runaway**: Unexplained fuel burn while generator is idle ($0$ kW).

### Quick Usage Example

```python
import pandas as pd
from models import detect_anomalies

# Scan recent station telemetry (last 48-168 hours)
recent_telemetry = pd.read_csv("models/simulation_results_365d.csv").tail(72)
alerts = detect_anomalies(recent_telemetry)

for alert in alerts:
    print(f"[{alert['severity']}] {alert['timestamp']}: {alert['reason']}")
```

### Detection Accuracy Benchmarks (`test_anomaly_detector.py`)
- **False Alarm Rate on Clean Data**: **0.00%** (target $< 3.0\%$).
- **Fault Episode Detection Recall**: **100.0%** (5/5 ground-truth multi-hour fault episodes caught).
- **Explainability**: Every alert includes timestamps, observed vs baseline metrics, and a diagnostic reason string for station facilities engineers.
