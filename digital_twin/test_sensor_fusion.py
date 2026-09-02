from sensor_fusion import SensorFusion


fusion = SensorFusion()

residuals = {
    "rpm_residual": 100.0,
    "cht_c_residual": 10.0,
    "egt_c_residual": 20.0,
    "oil_pressure_bar_residual": 0.2,
    "oil_temperature_c_residual": 7.0,
    "fuel_flow_lph_residual": 1.0,
    "injection_timing_deg_residual": 0.5,
    "vibration_mm_s_residual": 1.0,
    "battery_voltage_v_residual": 0.2,
    "alternator_current_a_residual": 1.0,
}

result = fusion.calculate(residuals)

print("\nSENSOR FUSION RESULT")
print("--------------------")

for key, value in result.items():
    print(f"{key:30s}: {value:.3f}")