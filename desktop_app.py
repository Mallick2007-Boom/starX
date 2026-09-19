"""
Boreas Polar Research Station — Desktop Mission Control Prototype
A native Windows desktop GUI application that connects to the AI Microgrid REST API.
Provides real-time telemetry HUD, season jumping, interactive load shedding, and explainable AI insights.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import urllib.request
import json
import threading
import time

API_BASE = "http://127.0.0.1:8000"

# Color Palette (Antarctic Mission Control Theme)
BG_BASE = "#060b14"
BG_CARD = "#0b1424"
BG_CARD_LIGHT = "#121d33"
TEXT_PRIMARY = "#f0f6fc"
TEXT_MUTED = "#8b9bb4"
CYAN_ACCENT = "#00f0ff"
GREEN_NEON = "#00e676"
AMBER_WARN = "#ffab00"
RED_ALERT = "#ff1744"
PURPLE_ACCENT = "#b388ff"


class PolarStationDesktopApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Boreas Polar Station — AI Microgrid Desktop Mission Control")
        self.geometry("980x740")
        self.minsize(880, 680)
        self.configure(bg=BG_BASE)

        self.current_step = 4716  # Default mid-winter
        self.is_streaming = False
        self.loads_active = {
            "drill": tk.BooleanVar(value=True),
            "lidar": tk.BooleanVar(value=True),
            "skidoo": tk.BooleanVar(value=True),
            "laundry": tk.BooleanVar(value=True),
        }

        self.build_ui()
        self.refresh_telemetry()

    def build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self, bg=BG_CARD, padx=20, pady=12, highlightbackground="#00e5ff", highlightthickness=1)
        header.pack(fill=tk.X, padx=15, pady=(12, 8))

        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_frame,
            text="❄️ BOREAS POLAR RESEARCH STATION",
            font=("Segoe UI", 16, "bold"),
            fg=CYAN_ACCENT,
            bg=BG_CARD,
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            title_frame,
            text="AI Microgrid MPC Controller • Ross Ice Shelf, Antarctica (-78.5°S, 166.7°E)",
            font=("Segoe UI", 9),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        sub_lbl.pack(anchor="w")

        # Live Status Chip
        self.status_chip = tk.Label(
            header,
            text="● SYSTEM ONLINE",
            font=("Segoe UI", 10, "bold"),
            fg=GREEN_NEON,
            bg="#0f261c",
            padx=12,
            pady=6,
        )
        self.status_chip.pack(side=tk.RIGHT)

        # 2. Season Scrubber & Fast Controls
        ctrl_frame = tk.Frame(self, bg=BG_CARD, padx=15, pady=10)
        ctrl_frame.pack(fill=tk.X, padx=15, pady=4)

        tk.Label(ctrl_frame, text="Season Timeline:", font=("Segoe UI", 10, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 10))

        btn_summer = tk.Button(ctrl_frame, text="☀️ Summer Peak", font=("Segoe UI", 9, "bold"), bg=BG_CARD_LIGHT, fg=AMBER_WARN, command=lambda: self.jump_season(360), relief=tk.FLAT, padx=8, pady=3)
        btn_summer.pack(side=tk.LEFT, padx=4)

        btn_autumn = tk.Button(ctrl_frame, text="🍂 Autumn Freeze", font=("Segoe UI", 9, "bold"), bg=BG_CARD_LIGHT, fg=TEXT_PRIMARY, command=lambda: self.jump_season(2520), relief=tk.FLAT, padx=8, pady=3)
        btn_autumn.pack(side=tk.LEFT, padx=4)

        btn_winter = tk.Button(ctrl_frame, text="❄️ Mid-Winter Blizzard", font=("Segoe UI", 9, "bold"), bg=BG_CARD_LIGHT, fg=CYAN_ACCENT, command=lambda: self.jump_season(4716), relief=tk.FLAT, padx=8, pady=3)
        btn_winter.pack(side=tk.LEFT, padx=4)

        btn_spring = tk.Button(ctrl_frame, text="🌅 Spring Dawn", font=("Segoe UI", 9, "bold"), bg=BG_CARD_LIGHT, fg=PURPLE_ACCENT, command=lambda: self.jump_season(7200), relief=tk.FLAT, padx=8, pady=3)
        btn_spring.pack(side=tk.LEFT, padx=4)

        self.btn_stream = tk.Button(ctrl_frame, text="▶ Stream Live", font=("Segoe UI", 9, "bold"), bg="#0051ff", fg="#ffffff", command=self.toggle_stream, relief=tk.FLAT, padx=12, pady=3)
        self.btn_stream.pack(side=tk.RIGHT, padx=5)

        # 3. Four Key Metric HUD Cards
        cards_frame = tk.Frame(self, bg=BG_BASE)
        cards_frame.pack(fill=tk.X, padx=15, pady=8)

        # Card 1: Outdoor Temp
        self.card_temp = self.create_hud_card(cards_frame, "OUTDOOR TEMP & HEATING", "-55.4 °C", "Critical Heating: 125.0 kW", CYAN_ACCENT)
        self.card_temp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Card 2: Diesel Reserve
        self.card_fuel = self.create_hud_card(cards_frame, "DIESEL FUEL RESERVE", "253,809 L", "Days Remaining: 250.6 Days", AMBER_WARN)
        self.card_fuel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Card 3: Battery BESS
        self.card_battery = self.create_hud_card(cards_frame, "BATTERY BESS (20% FLOOR)", "65.0 %", "Charge/Discharge: IDLE", GREEN_NEON)
        self.card_battery.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)

        # Card 4: Total Demand
        self.card_demand = self.create_hud_card(cards_frame, "TOTAL STATION POWER", "145.2 kW", "Wind: 45 kW • Diesel: 60 kW", PURPLE_ACCENT)
        self.card_demand.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # 4. Middle Section: Power Allocation Grid
        alloc_frame = tk.LabelFrame(self, text=" Live Microgrid Generation Mix ", font=("Segoe UI", 10, "bold"), fg=CYAN_ACCENT, bg=BG_CARD, padx=15, pady=10)
        alloc_frame.pack(fill=tk.X, padx=15, pady=6)

        grid_frame = tk.Frame(alloc_frame, bg=BG_CARD)
        grid_frame.pack(fill=tk.X)

        self.lbl_solar = tk.Label(grid_frame, text="☀️ Solar PV: 0.0 kW", font=("Consolas", 12, "bold"), fg=AMBER_WARN, bg=BG_CARD_LIGHT, padx=12, pady=8)
        self.lbl_solar.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_wind = tk.Label(grid_frame, text="💨 Wind: 45.2 kW (Icing: 0.95)", font=("Consolas", 12, "bold"), fg=CYAN_ACCENT, bg=BG_CARD_LIGHT, padx=12, pady=8)
        self.lbl_wind.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_diesel = tk.Label(grid_frame, text="⛽ Diesel Genset: 65.0 kW", font=("Consolas", 12, "bold"), fg=RED_ALERT, bg=BG_CARD_LIGHT, padx=12, pady=8)
        self.lbl_diesel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        self.lbl_bat = tk.Label(grid_frame, text="🔋 Battery: 0.0 kW", font=("Consolas", 12, "bold"), fg=GREEN_NEON, bg=BG_CARD_LIGHT, padx=12, pady=8)
        self.lbl_bat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        # 5. Explainable AI Decision Box
        ai_box = tk.LabelFrame(self, text=" 🧠 AI Model Predictive Control (MPC) Reason String ", font=("Segoe UI", 10, "bold"), fg=GREEN_NEON, bg=BG_CARD, padx=15, pady=10)
        ai_box.pack(fill=tk.X, padx=15, pady=6)

        self.ai_reason_lbl = tk.Label(
            ai_box,
            text="Calculating initial optimal dispatch setpoints...",
            font=("Segoe UI", 10, "italic"),
            fg="#e1ecfa",
            bg=BG_CARD,
            wraplength=920,
            justify=tk.LEFT,
        )
        self.ai_reason_lbl.pack(anchor="w")

        # 6. Interactive Flexible Load Shedder (Switches)
        shed_frame = tk.LabelFrame(self, text=" 🎛️ Demand-Side Management: Priority Load Shedder ", font=("Segoe UI", 10, "bold"), fg=AMBER_WARN, bg=BG_CARD, padx=15, pady=10)
        shed_frame.pack(fill=tk.X, padx=15, pady=6)

        switches_box = tk.Frame(shed_frame, bg=BG_CARD)
        switches_box.pack(fill=tk.X)

        tk.Checkbutton(
            switches_box,
            text="Deep Ice Drill (45 kW)",
            variable=self.loads_active["drill"],
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
            selectcolor=BG_CARD_LIGHT,
            activebackground=BG_CARD,
            activeforeground=CYAN_ACCENT,
            command=self.on_load_toggle,
        ).pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(
            switches_box,
            text="Atmospheric Lidar (20 kW)",
            variable=self.loads_active["lidar"],
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
            selectcolor=BG_CARD_LIGHT,
            activebackground=BG_CARD,
            activeforeground=CYAN_ACCENT,
            command=self.on_load_toggle,
        ).pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(
            switches_box,
            text="Skidoo Fast Chargers (15 kW)",
            variable=self.loads_active["skidoo"],
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
            selectcolor=BG_CARD_LIGHT,
            activebackground=BG_CARD,
            activeforeground=CYAN_ACCENT,
            command=self.on_load_toggle,
        ).pack(side=tk.LEFT, padx=12)

        tk.Checkbutton(
            switches_box,
            text="Laundry & Sauna (10 kW)",
            variable=self.loads_active["laundry"],
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
            selectcolor=BG_CARD_LIGHT,
            activebackground=BG_CARD,
            activeforeground=CYAN_ACCENT,
            command=self.on_load_toggle,
        ).pack(side=tk.LEFT, padx=12)

        # 7. Bottom Status Bar
        footer = tk.Frame(self, bg="#04080e", padx=15, pady=6)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        self.footer_lbl = tk.Label(footer, text=f"Backend API: {API_BASE} • Hour {self.current_step}/8759 • 100% Critical Load Served", font=("Consolas", 8), fg=TEXT_MUTED, bg="#04080e")
        self.footer_lbl.pack(side=tk.LEFT)

    def create_hud_card(self, parent, title, main_val, sub_val, accent_color):
        frame = tk.Frame(parent, bg=BG_CARD, padx=12, pady=10, highlightbackground=accent_color, highlightthickness=1)
        lbl_title = tk.Label(frame, text=title, font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_title.pack(anchor="w")

        lbl_main = tk.Label(frame, text=main_val, font=("Segoe UI", 16, "bold"), fg="#ffffff", bg=BG_CARD)
        lbl_main.pack(anchor="w", pady=(2, 0))

        lbl_sub = tk.Label(frame, text=sub_val, font=("Segoe UI", 8), fg=accent_color, bg=BG_CARD)
        lbl_sub.pack(anchor="w")

        # Save references
        frame.lbl_main = lbl_main
        frame.lbl_sub = lbl_sub
        return frame

    def jump_season(self, step: int):
        self.current_step = step
        self.refresh_telemetry()

    def toggle_stream(self):
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self.btn_stream.config(text="⏸ Pause Stream", bg="#ff1744")
            threading.Thread(target=self.stream_worker, daemon=True).start()
        else:
            self.btn_stream.config(text="▶ Stream Live", bg="#0051ff")

    def stream_worker(self):
        while self.is_streaming:
            self.current_step = (self.current_step + 1) % 8760
            self.after(0, self.refresh_telemetry)
            time.sleep(1.2)

    def on_load_toggle(self):
        self.refresh_telemetry()

    def calculate_flexible_offset(self):
        # Subtract loads that are toggled OFF
        offset = 0.0
        if not self.loads_active["drill"].get():
            offset += 45.0
        if not self.loads_active["lidar"].get():
            offset += 20.0
        if not self.loads_active["skidoo"].get():
            offset += 15.0
        if not self.loads_active["laundry"].get():
            offset += 10.0
        return offset

    def refresh_telemetry(self):
        try:
            # First try Digital Twin state endpoint
            url = f"{API_BASE}/api/digital-twin/state?step_index={self.current_step}"
            try:
                req = urllib.request.urlopen(url, timeout=3)
                dt_data = json.loads(req.read().decode())
                env = dt_data.get("environment", {})
                eng = dt_data.get("energy", {})
                stn = dt_data.get("station", {})
                rsk = dt_data.get("risk", {})

                temp = env.get("outdoor_temp_c", -30.0)
                crit = eng.get("critical_demand_kw", 65.0)
                fuel = eng.get("diesel_fuel_reserve_liters", 240000.0)
                days = rsk.get("autonomy_days", 200.0)
                soc = eng.get("battery_soc_pct", 65.0)
                solar = eng.get("solar_generation_kw", 0.0)
                wind = eng.get("wind_effective_kw", 0.0)
                icing = env.get("wind_icing_factor", 1.0)
                diesel = eng.get("diesel_generation_kw", 50.0)
                bat = eng.get("battery_power_kw", 0.0)
                mode_str = stn.get("operating_mode", "NORMAL")
                reason = dt_data.get("decision", {}).get("reason", "Nominal microgrid dispatch.")
                ts = dt_data.get("timestamp", "2026-07-15")
                demand = eng.get("total_station_demand_kw", 110.0)
                bat_mode = eng.get("battery_mode", "IDLE")
            except Exception:
                # Fallback to current telemetry endpoint
                url = f"{API_BASE}/api/telemetry/current?step_index={self.current_step}"
                req = urllib.request.urlopen(url, timeout=3)
                data = json.loads(req.read().decode())
                temp = data.get("outdoor_temp_c", -30.0)
                crit = data.get("load_critical_kw", 65.0)
                fuel = data.get("fuel_reserve_liters", 250000.0)
                days = data.get("days_of_fuel_remaining", 200.0)
                soc = data.get("battery_soc_pct", 65.0)
                solar = data.get("solar_kw", 0.0)
                wind = data.get("wind_kw", 0.0)
                icing = data.get("icing_factor", 1.0)
                diesel = data.get("kw_from_diesel", 50.0)
                bat = data.get("kw_from_battery", 0.0)
                mode_str = "NORMAL"
                reason = data.get("reason", "Nominal microgrid dispatch.")
                ts = data.get("timestamp", "2026-07-15")
                demand = data.get("total_demand_kw", 110.0)
                bat_mode = data.get("battery_charge_or_discharge", "IDLE")

            # Apply demand shedding offset
            shedded = self.calculate_flexible_offset()
            effective_demand = max(crit, demand - shedded)

            # Update status chip based on operating mode
            if mode_str == "SURVIVAL":
                self.status_chip.config(text="⚠️ SURVIVAL MODE", fg=RED_ALERT, bg="#3b0a0a")
            elif mode_str == "WARNING":
                self.status_chip.config(text="⚠️ ENERGY WARNING", fg=AMBER_WARN, bg="#332405")
            elif mode_str == "STORM_WATCH":
                self.status_chip.config(text="🌬️ STORM WATCH", fg=CYAN_ACCENT, bg="#08212e")
            else:
                self.status_chip.config(text="● SYSTEM ONLINE", fg=GREEN_NEON, bg="#0f261c")

            # Update UI elements
            self.card_temp.lbl_main.config(text=f"{temp:.1f} °C")
            self.card_temp.lbl_sub.config(text=f"Critical Heating: {crit:.1f} kW")

            self.card_fuel.lbl_main.config(text=f"{fuel:,.0f} L")
            self.card_fuel.lbl_sub.config(text=f"Autonomy: {days:.1f} Days")

            self.card_battery.lbl_main.config(text=f"{soc:.1f} %")
            self.card_battery.lbl_sub.config(text=f"Mode: {bat_mode}")

            self.card_demand.lbl_main.config(text=f"{effective_demand:.1f} kW")
            shed_text = f"Shedded: {shedded:.0f} kW" if shedded > 0 else "Nominal Science Demand"
            self.card_demand.lbl_sub.config(text=shed_text)

            self.lbl_solar.config(text=f"☀️ Solar: {solar:.1f} kW")
            self.lbl_wind.config(text=f"💨 Wind: {wind:.1f} kW (Icing: {icing:.2f})")
            self.lbl_diesel.config(text=f"⛽ Diesel: {diesel:.1f} kW")
            self.lbl_bat.config(text=f"🔋 Battery: {bat:.1f} kW")

            if shedded > 0:
                reason = f"[LOAD SHEDDING ACTIVE: -{shedded:.0f} kW] " + reason
            self.ai_reason_lbl.config(text=f'"{reason}"')

            self.footer_lbl.config(text=f"Backend API: {API_BASE} • Hour {self.current_step}/8759 • Date: {ts} • 100% Critical Load Met")

        except Exception as e:
            self.status_chip.config(text="● API OFFLINE", fg=RED_ALERT, bg="#2b0f0f")
            self.ai_reason_lbl.config(text=f"Failed to fetch telemetry from {API_BASE}: {e}")


if __name__ == "__main__":
    app = PolarStationDesktopApp()
    app.mainloop()
