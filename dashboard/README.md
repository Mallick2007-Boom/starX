# Polar Microgrid Operations Dashboard

The dashboard provides real-time situational awareness and operational control for polar research station facilities engineers and station managers.

## Key Views & Telemetry Panels

1. **Polar Survival & Energy HUD**:
   - Outdoor Temperature vs. Critical Life Support Heating Demand.
   - 365-day Diesel Fuel Burn & Remaining Reserve Days Countdown until summer resupply vessel.
   - Live Microgrid Generation Mix (Solar PV, Wind, Battery, Diesel Generators).
2. **Renewable Performance & Icing Monitor**:
   - Wind turbine aerodynamic derating gauge (`wind_icing_factor`) with freezing fog alerts.
   - Solar elevation and diurnal availability tracker.
3. **AI Dispatch & Flexible Load Shedding Panel**:
   - Predicted vs. actual demand curves (24h/48h lookahead).
   - Priority load scheduler (drills, science computers, laundry, skidoo battery chargers).
   - Recommended generator dispatch mode (Auto-AI, Fuel Conservation, Storm Survival).
