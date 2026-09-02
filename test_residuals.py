from digital_twin.twin_engine import DigitalTwinEngine
from digital_twin.residuals import ResidualAnalyzer


# ============================================================
# 1. CREATE DIGITAL TWIN
# ============================================================

twin = DigitalTwinEngine()


# ============================================================
# 2. OPERATING CONDITIONS
# ============================================================

throttle = 70.0
load = 65.0
altitude = 5000.0
ambient_temperature = 25.0
air_density = 0.736


# ============================================================
# 3. GET EXPECTED HEALTHY VALUES
# ============================================================

expected = twin.estimate(

    throttle_pct=throttle,

    engine_load_pct=load,

    altitude_ft=altitude,

    ambient_temperature_c=ambient_temperature,

    air_density_kg_m3=air_density
)


# ============================================================
# 4. SIMULATED ACTUAL VALUES
# ============================================================

actual = {

    "rpm": 5600.0,

    "cht_c": 87.0,

    "egt_c": 635.0,

    "oil_pressure_bar": 4.2,

    "oil_temperature_c": 70.0,

    "fuel_flow_lph": 20.0,

    "injection_timing_deg": 25.4,

    "vibration_mm_s": 1.6,

    "battery_voltage_v": 24.6,

    "alternator_current_a": 15.5
}


# ============================================================
# 5. CALCULATE RESIDUALS
# ============================================================

analyzer = ResidualAnalyzer()

residuals = analyzer.calculate(
    actual,
    expected
)


# ============================================================
# 6. DISPLAY
# ============================================================

print()
print("=" * 70)
print(" DIGITAL TWIN RESIDUAL ANALYSIS")
print("=" * 70)
print()

for name, value in residuals.items():

    print(
        f"{name:<35} : {value:+.2f}"
    )

print()