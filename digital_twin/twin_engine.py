class DigitalTwinEngine:

    """
    Healthy reference model for the Aero-Piston Engine.

    The Digital Twin does NOT receive fault information.

    It estimates what a healthy engine should produce from:
        - throttle
        - engine load
        - altitude
        - ambient temperature
        - air density

    The result is later compared with actual telemetry.
    """

    def __init__(self):

        # Healthy engine operating limits
        self.idle_rpm = 2500.0
        self.max_rpm = 7500.0

    # =========================================================
    # MAIN ESTIMATION
    # =========================================================

    def estimate(
        self,
        throttle_pct,
        engine_load_pct,
        altitude_ft,
        ambient_temperature_c,
        air_density_kg_m3
    ):

        # -----------------------------------------------------
        # Normalize inputs
        # -----------------------------------------------------

        throttle = max(
            0.0,
            min(throttle_pct / 100.0, 1.0)
        )

        load = max(
            0.0,
            min(engine_load_pct / 100.0, 1.0)
        )

        # -----------------------------------------------------
        # Air density ratio
        # -----------------------------------------------------

        sea_level_density = 1.225

        density_ratio = (
            air_density_kg_m3
            / sea_level_density
        )

        density_ratio = max(
            0.55,
            min(1.10, density_ratio)
        )

        # =====================================================
        # 1. EXPECTED RPM
        # =====================================================

        expected_rpm = (
            self.idle_rpm
            + throttle
            * (
                self.max_rpm
                - self.idle_rpm
            )
        )

        # Load reduces achievable RPM
        expected_rpm *= (
            1.0
            - 0.05 * load
        )

        # Reduced air density reduces available power
        expected_rpm *= (
            0.92
            + 0.08 * density_ratio
        )

        # Keep inside engine limits
        expected_rpm = max(
            self.idle_rpm,
            min(
                expected_rpm,
                self.max_rpm
            )
        )

        # =====================================================
        # 2. EXPECTED FUEL FLOW
        # =====================================================

        expected_fuel_flow = (
            3.0
            + 18.0 * throttle
            + 4.0 * load
            + 2.0 * (1.0 - density_ratio)
        )

        # =====================================================
        # 3. EXPECTED INJECTION TIMING
        # =====================================================

        expected_injection_timing = (
            24.0
            + 2.0 * throttle
        )

        # =====================================================
        # 4. EXPECTED EGT
        # =====================================================

        expected_egt = (
            400.0
            + 280.0 * load
            + 50.0 * throttle
            + 25.0 * (1.0 - density_ratio)
        )

        # =====================================================
        # 5. EXPECTED CHT
        # =====================================================

        # Healthy CHT reference calibrated from the
        # normal mission envelope.

        expected_cht = (
            89.11
            - 4.14 * throttle
            + 6.59 * load
            - 18.20 * density_ratio
            + 0.26 * ambient_temperature_c
        )

        # Physical lower limit
        expected_cht = max(
            ambient_temperature_c + 20.0,
            expected_cht
        )

        # Physical upper limit
        expected_cht = min(
            expected_cht,
            120.0
        )

        # =====================================================
        # 6. EXPECTED OIL TEMPERATURE
        # =====================================================

        # Healthy reference calibrated against the normal
        # mission data.

        expected_oil_temperature = (
            52.67
            + 2.06 * throttle
            - 3.36 * load
            + 0.23 * density_ratio
            + 0.56 * ambient_temperature_c
        )

        expected_oil_temperature = max(
            ambient_temperature_c + 20.0,
            expected_oil_temperature
        )

        # =====================================================
        # 7. EXPECTED OIL PRESSURE
        # =====================================================

        rpm_factor = (
            expected_rpm
            / 7000.0
        )

        temperature_factor = max(
            0.70,
            1.0
            - 0.002
            * (
                expected_oil_temperature
                - 80.0
            )
        )

        expected_oil_pressure = (
            2.0
            + 3.5
            * rpm_factor
            * temperature_factor
        )

        # =====================================================
        # 8. EXPECTED VIBRATION
        # =====================================================

        expected_vibration = (
            0.5
            + 0.00010 * expected_rpm
            + 0.8 * load
        )

        # =====================================================
        # 9. EXPECTED BATTERY VOLTAGE
        # =====================================================

        expected_battery_voltage = (
            24.0
            + 0.6
            * min(
                expected_rpm / 5000.0,
                1.0
            )
        )

        # =====================================================
        # 10. EXPECTED ALTERNATOR CURRENT
        # =====================================================

        alternator_efficiency = (
            0.80
            + 0.20
            * min(
                expected_rpm / 5000.0,
                1.0
            )
        )

        expected_alternator_current = (
            5.0
            + 15.0
            * throttle
            * alternator_efficiency
        )

        # =====================================================
        # RETURN HEALTHY EXPECTED STATE
        # =====================================================

        return {

            "expected_rpm":
                expected_rpm,

            "expected_cht_c":
                expected_cht,

            "expected_egt_c":
                expected_egt,

            "expected_oil_pressure_bar":
                expected_oil_pressure,

            "expected_oil_temperature_c":
                expected_oil_temperature,

            "expected_fuel_flow_lph":
                expected_fuel_flow,

            "expected_injection_timing_deg":
                expected_injection_timing,

            "expected_vibration_mm_s":
                expected_vibration,

            "expected_battery_voltage_v":
                expected_battery_voltage,

            "expected_alternator_current_a":
                expected_alternator_current
        }