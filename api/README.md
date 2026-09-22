# BOREAS API — REST Microservice Reference

The BOREAS API is a high-performance, asynchronous FastAPI backend microservice engineered for Antarctic microgrid mission control. It provides sub-50ms physical state queries, rolling-horizon MPC dispatch optimization, 72-hour synchronized multi-channel forecasts, real-time mission risk diagnostics, and What-If resilience stress simulations.

---

## Architecture & Conventions

- **Dual-Mount Routing**: Every endpoint is mounted at both `/api/<path>` and `/<path>` for backward compatibility and reverse proxy flexibility.
- **FastAPI Dependency-Clean Serialization**: Query parameters cleanly accept typed query arguments or direct function invocations.

### 🌐 Documentation & Interface Directory

| Interface / Portal | URL Link | Format | Description |
|---|---|---|---|
| **Mission Control Dashboard** | [http://localhost:8000/](http://localhost:8000/) | HTML5 / Web HUD | Full 7-row real-time station telemetry dashboard |
| **Interactive Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI | Live endpoint execution, query parameter testing, & schemas |
| **ReDoc Interface** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | ReDoc Engine | Clean three-column technical reference manual |
| **OpenAPI Specification** | [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json) | JSON (OpenAPI 3.1) | Machine-readable API schema and endpoint definitions |

---

## Complete API Endpoints Catalog

### 1. Antarctic Digital Twin & Physical State

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/digital-twin/state` | `step` (int, 0-8759) | Returns the comprehensive physical Digital Twin snapshot: environmental telemetry, atmospheric metrics, power dispatch, BESS status, fuel reserve, equipment availability, and operating mode (`NORMAL`, `WATCH`, `WARNING`, `SURVIVAL`). |
| `GET` | `/api/telemetry/current` | `step` (int, optional) | Real-time sensor snapshot: outdoor temp, wind speed, icing derating, solar PV, wind, diesel burn, and battery SoC. |
| `GET` | `/api/telemetry/history` | `hours` (int, default 72), `end_step` (int) | Historical time series for multi-channel trend graphing (temperature, wind, icing, generation breakdown). |
| `GET` | `/api/health` | None | Station health check, coordinates (-78.5°S, 166.7°E), elevation (1,200m), and active model roster. |

---

### 2. Predictive Forecasting & Deficit Windows

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/forecast/72h` | `step` (int, default 4716) | Synchronized 72-hour multi-channel forecast: outdoor temperature, wind, solar PV, effective wind with icing, critical vs flexible load, net balance, and marked **Energy Deficit Windows** (start/end hour, duration, peak deficit kW, total energy deficit kWh). |
| `POST` | `/api/predict/load` | `LoadForecastRequest` (horizon_hours, temp_override, etc.) | Runs dual-gradient boosting load forecaster for critical heating vs flexible science operations. |
| `POST` | `/api/predict/renewables` | `RenewableForecastRequest` (horizon_hours, temp_c, wind_mps, rh_pct) | Predicts bifacial solar PV and aerodynamic rime icing derated wind generation under IEA Wind Task 19. |

---

### 3. Mission Risk Engine

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/risk/current` | `step` (int, default 4716) | Evaluates station operational survival risk: risk score (0–100), operating level (`NORMAL`, `WATCH`, `WARNING`, `SURVIVAL`), interpretable physical causes, autonomy days remaining, resupply margin %, and recommended actions. |
| `GET` | `/api/fuel/reserve-projection` | `step` (int, optional) | Estimates fuel exhaustion date, daily burn rates, resupply safety margin %, and operational risk tier. |

---

### 4. What-If Scenario Stress Testing

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/scenarios/catalog` | None | Returns metadata for all 8 preconfigured extreme polar scenarios (blizzard, diesel failure, battery trip, low wind, polar vortex, solar occlusion, high crew, multiple compound failure). |
| `POST` | `/api/scenarios/run` | `ScenarioRunRequest` (`scenario_id`, `custom_duration_hours`, `custom_severity`) | Executes physical simulation of the selected scenario: computes **Before vs After impact**, **Baseline (Reactive) vs BOREAS (Predictive MPC)** comparative metrics (fuel burned, fuel saved, critical load met %, battery min SoC %, load shed kWh). |

---

### 5. Explainable AI (XAI) & Operator Timeline

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/ai/decision` | `step` (int, default 4716) | Emits structured Explainable AI decision card: primary action taken, detailed physical rationale, primary driver, fuel impact (L/h saved), battery impact (%/h), and alternative options considered. |
| `GET` | `/api/ai/timeline` | `step` (int, default 4716) | Chronological operator decision timeline (past actions and planned future dispatches) with severity tagging, category, and audit detail. |

---

### 6. Demand-Side Management & Optimization

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/loads/toggle` | None | Inspects current status of all operator load switches (`drill`, `lidar`, `skidoo`, `laundry`, `sauna`). |
| `POST` | `/api/loads/toggle` | `LoadToggleRequest` (`drill`, `lidar`, `skidoo`, `laundry`, `sauna`) | Updates demand-side switches; triggers immediate MPC recalculation and state update. |
| `POST` | `/api/optimize/dispatch` | `DispatchOptimizationRequest` | Solves single-hour microgrid balance under strict survival constraints (Cold Battery Protection, minimum loading, critical heating priority). |

---

### 7. Performance Benchmarks & Equipment Diagnostics

| Method | Endpoint | Query / Body Params | Description |
|---|---|---|---|
| `GET` | `/api/comparison/baseline` | None | Annual simulation audit under standard reactive rule-based control (no lookahead, standard battery floor). |
| `GET` | `/api/comparison/boreas` | None | Annual simulation audit under BOREAS physics-informed predictive MPC (18,153L fuel saved, 49.3% renewable penetration, 100% critical survival). |
| `GET` | `/api/simulation/summary` | None | High-level summary of the 8,760-hour polar simulation benchmark. |
| `GET` | `/api/anomalies` | `step` (int, optional) | Active multivariate isolation forest anomaly alerts (generator efficiency drift, battery cold capacity collapse, idle fuel leaks). |
