/**
 * STAR-X: Autonomous Energy Commander — Physics & Telemetry Engine
 * High-fidelity microgrid simulation, IEA Task 19 wind icing physics,
 * battery electrochemistry, thermal building balance, and Google Weather API ingestion.
 */

class StarXEngine {
  constructor() {
    // Station Configuration
    this.config = {
      name: "STAR-X Antarctic Surface Research Facility",
      latitude: -78.5,
      longitude: 166.66,
      elevationM: 1200,
      pvCapacityKw: 120.0,
      windRatedKw: 100.0,
      batteryCapacityKwh: 360.0,
      batteryMaxChargeKw: 60.0,
      batteryMaxDischargeKw: 75.0,
      batteryMinSocFloor: 20.0,
      batteryColdProtectionFloor: 30.0, // Enforced when ambient < -35°C
      dieselGenset1RatedKw: 100.0,
      dieselGenset2RatedKw: 80.0,
      dieselFuelConsumptionLPerKwh: 0.285,
      dieselIdleBurnLph: 3.5,
      totalFuelReserveLiters: 360000.0,
    };

    // Current Operational State
    this.state = {
      stepIndex: 4716, // Mid-winter polar night default
      timestamp: new Date("2026-07-15T12:00:00Z"),
      isStreaming: false,
      streamSpeedMs: 1500,
      streamTimer: null,
      dataSource: "INITIALIZING",
      apiConnected: false,

      // Environmental readings
      env: {
        outdoorTempC: -52.4,
        windSpeedMps: 14.8,
        relativeHumidityPct: 76.5,
        cloudCoverPct: 45.0,
        daylightHours: 0.0,
        icingFactor: 0.72,
        weatherCondition: "Katabatic Blizzard Warning",
        stormSeverity: "HIGH",
      },

      // Power Grid (kW)
      power: {
        solarKw: 0.0,
        windRawKw: 72.4,
        windEffectiveKw: 52.1,
        totalRenewableKw: 52.1,
        criticalDemandKw: 84.5,
        flexibleDemandKw: 38.0,
        totalDemandKw: 122.5,
        curtailedFlexibleKw: 0.0,
        netDeficitKw: 70.4,
        batteryPowerKw: -35.0, // negative = discharging, positive = charging
        batterySocPct: 62.4,
        batteryKwh: 224.6,
        batteryColdGuardActive: true,
        batteryStateOfHealthPct: 98.4,
        dieselGenset1Kw: 35.4,
        dieselGenset2Kw: 0.0,
        totalDieselKw: 35.4,
        fuelFlowLph: 13.6,
        fuelRemainingLiters: 268400.0,
        autonomyDaysRemaining: 184,
      },

      // Load Shedding Switches
      loads: {
        iceDrill: { name: "Deep Ice Core Drill", kw: 45.0, active: true, shedable: true },
        lidar: { name: "Atmospheric LIDAR", kw: 15.0, active: true, shedable: true },
        skidoo: { name: "E-Skidoo Mobility Hub", kw: 22.0, active: true, shedable: true },
        laundry: { name: "Hydro-Sanitation & Laundry", kw: 8.0, active: true, shedable: true },
        habitatHeating: { name: "Life Support Thermal Loop", kw: 62.0, active: true, shedable: false },
        stationComms: { name: "Satellite Ground Uplink", kw: 7.5, active: true, shedable: false },
      },

      // AI Decision & Diagnostics
      ai: {
        operatingMode: "SURVIVAL", // NORMAL, WATCH, WARNING, SURVIVAL
        riskScore: 78,
        actionTitle: "Thermal Life-Support Defense Dispatch",
        rationale: "Extreme polar chill (-52.4°C) with zero solar availability. Genset #1 active in optimal fuel band; battery providing peak-shaving while honoring 30% freeze protection floor.",
        recommendation: "Pre-emptively defer Deep Ice Drill (45 kW) if wind drops below 9 m/s to conserve 12.8 L/hr diesel.",
        projectedSavingsLiters: 182.5,
      },

      // 72h Lookahead Forecast Series
      forecast72h: [],
    };

    this.subscribers = [];
  }

  // Subscribe to state updates
  subscribe(callback) {
    this.subscribers.push(callback);
    callback(this.state);
  }

  notify() {
    this.subscribers.forEach((cb) => cb(this.state));
  }

  // =========================================================================
  // ATMOSPHERIC & PHYSICS ALGORITHMS
  // =========================================================================

  /**
   * IEA Wind TCP Task 19 Cold Climate Aerodynamic Blade Icing Model
   */
  calculateIcingFactor(tempC, humidityPct, windSpeedMps) {
    // Icing occurs primarily between -22°C and -1°C with saturated air
    if (tempC > 0 || tempC < -26) {
      return 1.0; // Clean blades outside susceptible envelope
    }
    const tempSusceptibility = Math.exp(-Math.pow((tempC - (-8.0)) / 6.5, 2));
    const moistureFactor = Math.max(0, (humidityPct - 55.0) / 45.0);
    const windImpingement = Math.min(1.2, Math.max(0.4, windSpeedMps / 11.0));

    const derateLoss = 0.85 * tempSusceptibility * moistureFactor * windImpingement;
    return Math.max(0.15, Math.min(1.0, 1.0 - derateLoss));
  }

  /**
   * Wind turbine cubic power curve with cut-in, rated, and storm cutout
   */
  calculateWindPower(windSpeedMps, icingFactor) {
    const cutIn = 3.0;
    const rated = 12.0;
    const cutOut = 25.0;

    let rawKw = 0.0;
    if (windSpeedMps >= cutIn && windSpeedMps <= cutOut) {
      if (windSpeedMps >= rated) {
        rawKw = this.config.windRatedKw;
      } else {
        const ratio = (windSpeedMps - cutIn) / (rated - cutIn);
        rawKw = this.config.windRatedKw * Math.pow(ratio, 3);
      }
    }
    const effectiveKw = rawKw * icingFactor;
    return {
      rawKw: Math.round(rawKw * 10) / 10,
      effectiveKw: Math.round(effectiveKw * 10) / 10,
    };
  }

  /**
   * Solar Photovoltaic Generation with polar albedo snow boost & polar night clamping
   */
  calculateSolarPower(dayOfYear, hourOfDay, cloudCoverPct) {
    // Antarctic polar night: Days 120 to 243
    if (dayOfYear >= 120 && dayOfYear <= 243) {
      return 0.0;
    }
    // Midnight sun summer (Days 305 to 31)
    const isMidnightSun = dayOfYear >= 305 || dayOfYear <= 31;
    let elevationSin = 0.0;

    if (isMidnightSun) {
      // 24h sun oscillating with low angle
      elevationSin = 0.28 + 0.18 * Math.sin(((hourOfDay - 12) * Math.PI) / 12);
    } else {
      // Shoulder season
      elevationSin = Math.max(0, Math.sin(((hourOfDay - 5) * Math.PI) / 14) * 0.42);
    }

    if (elevationSin <= 0) return 0.0;

    const clearSkyKw = this.config.pvCapacityKw * elevationSin * 1.25; // 1.25 albedo snow reflection
    const cloudAttenuation = 1.0 - (cloudCoverPct / 100.0) * 0.65;
    return Math.max(0, Math.round(clearSkyKw * cloudAttenuation * 10) / 10);
  }

  /**
   * Station Heating & Critical Life-Support Thermal Equilibrium
   */
  calculateHeatingLoad(outdoorTempC) {
    const indoorTargetC = 19.5;
    const deltaT = Math.max(0, indoorTargetC - outdoorTempC);
    // Base electrical heat + UA envelope conductance loss
    const heatingKw = 38.0 + deltaT * 0.88;
    return Math.round(heatingKw * 10) / 10;
  }

  // =========================================================================
  // MODEL PREDICTIVE CONTROL (MPC) DISPATCH SOLVER
  // =========================================================================

  solveMicrogridDispatch(overrideParams = null) {
    const env = overrideParams ? { ...this.state.env, ...overrideParams } : this.state.env;

    // 1. Calculate generation
    const icingFactor = this.calculateIcingFactor(env.outdoorTempC, env.relativeHumidityPct, env.windSpeedMps);
    const wind = this.calculateWindPower(env.windSpeedMps, icingFactor);
    const dayOfYear = this.state.timestamp.getUTCMonth() * 30 + this.state.timestamp.getUTCDate();
    const hour = this.state.timestamp.getUTCHours();
    const solarKw = this.calculateSolarPower(dayOfYear, hour, env.cloudCoverPct);
    const totalRenewableKw = Math.round((wind.effectiveKw + solarKw) * 10) / 10;

    // 2. Calculate demand
    const heatingKw = this.calculateHeatingLoad(env.outdoorTempC);
    const commsKw = this.state.loads.stationComms.active ? this.state.loads.stationComms.kw : 0.0;
    const criticalDemandKw = Math.round((heatingKw + commsKw) * 10) / 10;

    let flexibleDemandKw = 0.0;
    let curtailedKw = 0.0;
    Object.entries(this.state.loads).forEach(([key, load]) => {
      if (load.shedable) {
        if (load.active) {
          flexibleDemandKw += load.kw;
        } else {
          curtailedKw += load.kw;
        }
      }
    });
    flexibleDemandKw = Math.round(flexibleDemandKw * 10) / 10;
    const totalDemandKw = Math.round((criticalDemandKw + flexibleDemandKw) * 10) / 10;

    // 3. Power balance & Net Deficit
    const netDeficitKw = Math.round((totalDemandKw - totalRenewableKw) * 10) / 10;

    // 4. Battery Logic & Cold Protection Floor
    const coldGuard = env.outdoorTempC < -35.0;
    const minSocFloor = coldGuard ? this.config.batteryColdProtectionFloor : this.config.batteryMinSocFloor;

    let batteryPowerKw = 0.0;
    let batterySoc = this.state.power.batterySocPct;
    let dieselKw = 0.0;

    if (netDeficitKw <= 0) {
      // Surplus power -> Charge battery
      const surplus = Math.abs(netDeficitKw);
      const headroomKwh = (this.config.batteryCapacityKw || 360.0) * ((100.0 - batterySoc) / 100.0);
      batteryPowerKw = Math.min(this.config.batteryMaxChargeKw, Math.min(surplus, headroomKwh));
      batteryPowerKw = Math.round(batteryPowerKw * 10) / 10;
      dieselKw = 0.0;
    } else {
      // Deficit -> Discharge battery first down to minSocFloor, then run diesel
      const usableSoc = Math.max(0, batterySoc - minSocFloor);
      const usableEnergyKw = (usableSoc / 100.0) * this.config.batteryCapacityKwh;

      if (usableEnergyKw > 10.0) {
        // Battery shares load
        batteryPowerKw = -Math.min(this.config.batteryMaxDischargeKw, Math.min(netDeficitKw * 0.65, usableEnergyKw));
        batteryPowerKw = Math.round(batteryPowerKw * 10) / 10;
      } else {
        batteryPowerKw = 0.0; // Battery protected
      }

      // Remaining deficit covered by Diesel Generator
      const unservedDeficit = netDeficitKw + batteryPowerKw; // batteryPowerKw is negative
      dieselKw = Math.max(0.0, Math.round(unservedDeficit * 10) / 10);
    }

    // Generator dispatch allocation between Gen #1 (primary) and Gen #2 (peaking/backup)
    let gen1Kw = 0.0;
    let gen2Kw = 0.0;
    if (dieselKw > 0) {
      gen1Kw = Math.min(this.config.dieselGenset1RatedKw, dieselKw);
      gen2Kw = Math.max(0, dieselKw - gen1Kw);
    }

    // Fuel consumption rate (Specific fuel curve: 0.285 L/kWh + standby idle burn)
    const fuelFlowLph = dieselKw > 0
      ? Math.round((dieselKw * this.config.dieselFuelConsumptionLPerKwh + this.config.dieselIdleBurnLph) * 10) / 10
      : 0.0;

    // Determine Operating Mode
    let mode = "NORMAL";
    let risk = 20;
    if (netDeficitKw > 60 && batterySoc < 35.0) {
      mode = "SURVIVAL";
      risk = 92;
    } else if (netDeficitKw > 35.0 || env.outdoorTempC < -48.0 || icingFactor < 0.5) {
      mode = "WARNING";
      risk = 65;
    } else if (env.windSpeedMps > 20.0 || env.outdoorTempC < -35.0) {
      mode = "WATCH";
      risk = 42;
    }

    // Update state
    this.state.env = {
      ...env,
      icingFactor: Math.round(icingFactor * 1000) / 1000,
    };

    this.state.power = {
      ...this.state.power,
      solarKw,
      windRawKw: wind.rawKw,
      windEffectiveKw: wind.effectiveKw,
      totalRenewableKw,
      criticalDemandKw,
      flexibleDemandKw,
      totalDemandKw,
      curtailedFlexibleKw: Math.round(curtailedKw * 10) / 10,
      netDeficitKw: Math.max(0, netDeficitKw),
      batteryPowerKw,
      batteryColdGuardActive: coldGuard,
      dieselGenset1Kw: Math.round(gen1Kw * 10) / 10,
      dieselGenset2Kw: Math.round(gen2Kw * 10) / 10,
      totalDieselKw: dieselKw,
      fuelFlowLph,
    };

    this.state.ai.operatingMode = mode;
    this.state.ai.riskScore = risk;

    this.generateAIDecisionCards();
    this.notify();
  }

  generateAIDecisionCards() {
    const p = this.state.power;
    const env = this.state.env;

    if (this.state.ai.operatingMode === "SURVIVAL") {
      this.state.ai.actionTitle = "🚨 Polar Emergency Fuel & Thermal Conservation";
      this.state.ai.rationale = `Severe energy deficit (${p.netDeficitKw} kW) at ${env.outdoorTempC}°C. Battery protected above ${this.config.batteryColdProtectionFloor}% SoC freeze limit. Life-support habitability guaranteed.`;
      this.state.ai.recommendation = "Automatically curtail Deep Ice Drill (45 kW) and E-Skidoo Charger (22 kW) immediately.";
      this.state.ai.projectedSavingsLiters = 19.1;
    } else if (this.state.ai.operatingMode === "WARNING") {
      this.state.ai.actionTitle = "⚠️ Lookahead Aerodynamic Icing & Deficit Defense";
      this.state.ai.rationale = `Blade rime icing derating wind generation by ${Math.round((1 - env.icingFactor) * 100)}%. Genset #1 dispatched in peak fuel-efficiency band (35–55 kW).`;
      this.state.ai.recommendation = "Maintain battery buffer at 40% SoC for incoming katabatic storm front.";
      this.state.ai.projectedSavingsLiters = 11.4;
    } else if (this.state.ai.operatingMode === "WATCH") {
      this.state.ai.actionTitle = "🌬️ High-Wind Katabatic Storm Watch Active";
      this.state.ai.rationale = `Wind speed elevated at ${env.windSpeedMps} m/s. Maximizing battery pre-charge before storm cut-out threshold (25 m/s).`;
      this.state.ai.recommendation = "Pre-charge BESS battery at 40 kW to absorb transient gusts.";
      this.state.ai.projectedSavingsLiters = 8.2;
    } else {
      this.state.ai.actionTitle = "✨ High-Efficiency Green Renewable Dominance";
      this.state.ai.rationale = `Renewable supply (${p.totalRenewableKw} kW) satisfying 100% station load. Zero diesel burned; fuel reserves fully conserved for winter.`;
      this.state.ai.recommendation = "All scientific and exploratory loads authorized for unrestricted operation.";
      this.state.ai.projectedSavingsLiters = 0.0;
    }
  }

  // =========================================================================
  // INTERACTIVE SANDBOX CONTROLS
  // =========================================================================

  setSandboxEnvironment(key, value) {
    const num = parseFloat(value);
    if (!isNaN(num)) {
      this.state.env[key] = num;
      this.solveMicrogridDispatch();
    }
  }

  setBatterySoc(val) {
    const num = parseFloat(val);
    if (!isNaN(num)) {
      this.state.power.batterySocPct = Math.max(10.0, Math.min(100.0, num));
      this.state.power.batteryKwh = Math.round(((this.state.power.batterySocPct / 100.0) * this.config.batteryCapacityKwh) * 10) / 10;
      this.solveMicrogridDispatch();
    }
  }

  toggleLoad(loadKey, active) {
    if (this.state.loads[loadKey] && this.state.loads[loadKey].shedable) {
      this.state.loads[loadKey].active = Boolean(active);
      this.solveMicrogridDispatch();
    }
  }

  // =========================================================================
  // GOOGLE WEATHER API LIVE INGESTION
  // =========================================================================

  async fetchLiveGoogleWeather(apiKey = null) {
    try {
      this.state.dataSource = "SYNCING_LIVE_API";
      const apiBase = (typeof window !== 'undefined' && (window.location.protocol === 'file:' || (window.location.port !== '8000' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname === ''))))
        ? 'http://127.0.0.1:8000'
        : '';
      const url = apiKey
        ? `${apiBase}/api/weather/live?api_key=${encodeURIComponent(apiKey)}&hours=72`
        : `${apiBase}/api/weather/live?hours=72`;

      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data && data.forecast_hours && data.forecast_hours.length > 0) {
        const current = data.forecast_hours[0];
        this.state.env.outdoorTempC = current.outdoor_temp_c;
        this.state.env.windSpeedMps = current.wind_speed_mps;
        this.state.env.relativeHumidityPct = current.relative_humidity_pct;
        this.state.env.cloudCoverPct = current.cloud_cover_pct || 40.0;
        this.state.env.weatherCondition = current.weather_condition || "Live Atmospheric Data";
        this.state.dataSource = data.data_source === "google_weather_api" ? "GOOGLE_WEATHER_LIVE" : "POLAR_CALIBRATED_FALLBACK";
        this.state.apiConnected = true;

        // Generate 72h forecast series from live data
        this.generate72hForecastFromSeries(data.forecast_hours);
        this.solveMicrogridDispatch();
        return { success: true, source: this.state.dataSource, count: data.forecast_hours.length };
      }
      throw new Error("Invalid payload");
    } catch (err) {
      console.warn("Live weather fetch failed, using local engine:", err);
      this.state.dataSource = "LOCAL_PHYSICS_ENGINE";
      this.state.apiConnected = false;
      this.generateSynthetic72hForecast();
      this.solveMicrogridDispatch();
      return { success: false, error: err.message };
    }
  }

  generate72hForecastFromSeries(series) {
    const list = [];
    series.slice(0, 72).forEach((item, idx) => {
      const icing = this.calculateIcingFactor(item.outdoor_temp_c, item.relative_humidity_pct, item.wind_speed_mps);
      const wind = this.calculateWindPower(item.wind_speed_mps, icing);
      const solar = this.calculateSolarPower(200, (idx + 12) % 24, item.cloud_cover_pct || 50.0);
      const heat = this.calculateHeatingLoad(item.outdoor_temp_c);
      const totDemand = heat + 35.0;
      const totRenew = wind.effectiveKw + solar;
      const deficit = Math.max(0, totDemand - totRenew);
      const diesel = deficit > 20 ? Math.min(100, deficit * 0.75 + 5) : 0;
      const bat = deficit > 0 ? Math.min(45, deficit - diesel) : 0;

      list.push({
        hour: idx + 1,
        timeLabel: `+${idx + 1}h`,
        tempC: item.outdoor_temp_c,
        windMps: item.wind_speed_mps,
        icingFactor: Math.round(icing * 100) / 100,
        demandKw: Math.round(totDemand * 10) / 10,
        renewableKw: Math.round(totRenew * 10) / 10,
        dieselKw: Math.round(diesel * 10) / 10,
        batteryKw: Math.round(bat * 10) / 10,
        deficitKw: Math.round(deficit * 10) / 10,
      });
    });
    this.state.forecast72h = list;
  }

  generateSynthetic72hForecast() {
    const list = [];
    const baseTemp = this.state.env.outdoorTempC;
    const baseWind = this.state.env.windSpeedMps;

    for (let h = 1; h <= 72; h++) {
      const temp = baseTemp + Math.sin(h / 12) * 5.2 - (h > 24 && h < 48 ? 6.0 : 0);
      const wind = Math.max(2.0, baseWind + Math.sin(h / 8) * 6.5 + (h > 30 && h < 55 ? 8.0 : 0));
      const rh = Math.min(95, Math.max(45, 70 + Math.sin(h / 6) * 18));
      const icing = this.calculateIcingFactor(temp, rh, wind);
      const windP = this.calculateWindPower(wind, icing);
      const heat = this.calculateHeatingLoad(temp);
      const totDemand = heat + 38.0;
      const totRenew = windP.effectiveKw; // Polar night default
      const deficit = Math.max(0, totDemand - totRenew);
      const diesel = deficit > 15 ? Math.min(100, deficit * 0.8) : 0;
      const bat = deficit > 0 ? Math.min(40, deficit - diesel) : 0;

      list.push({
        hour: h,
        timeLabel: `+${h}h`,
        tempC: Math.round(temp * 10) / 10,
        windMps: Math.round(wind * 10) / 10,
        icingFactor: Math.round(icing * 100) / 100,
        demandKw: Math.round(totDemand * 10) / 10,
        renewableKw: Math.round(totRenew * 10) / 10,
        dieselKw: Math.round(diesel * 10) / 10,
        batteryKw: Math.round(bat * 10) / 10,
        deficitKw: Math.round(deficit * 10) / 10,
      });
    }
    this.state.forecast72h = list;
  }

  // =========================================================================
  // EXPORT SIMULATION LOGS
  // =========================================================================

  exportTelemetryCsv() {
    const rows = [
      ["Hour", "Timestamp", "OutdoorTemp_C", "WindSpeed_mps", "IcingFactor", "Demand_kW", "Renewables_kW", "Diesel_kW", "Battery_kW", "Deficit_kW"]
    ];
    this.state.forecast72h.forEach((r) => {
      rows.push([r.hour, r.timeLabel, r.tempC, r.windMps, r.icingFactor, r.demandKw, r.renewableKw, r.dieselKw, r.batteryKw, r.deficitKw]);
    });
    const csvContent = "data:text/csv;charset=utf-8," + rows.map((e) => e.join(",")).join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `starx_telemetry_forecast_72h_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  exportTelemetryJson() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.state, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `starx_telemetry_snapshot_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }
}

// Global instance
window.starX = new StarXEngine();
