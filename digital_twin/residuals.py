class ResidualAnalyzer:

    def calculate(self, actual, expected):

        residuals = {}

        # -----------------------------------------------------
        # Calculate Actual - Expected
        # -----------------------------------------------------

        parameters = [
            "rpm",
            "cht_c",
            "egt_c",
            "oil_pressure_bar",
            "oil_temperature_c",
            "fuel_flow_lph",
            "injection_timing_deg",
            "vibration_mm_s",
            "battery_voltage_v",
            "alternator_current_a"
        ]

        for parameter in parameters:

            actual_key = parameter
            expected_key = "expected_" + parameter

            if (
                actual_key in actual
                and expected_key in expected
            ):

                residuals[
                    parameter + "_residual"
                ] = (
                    actual[actual_key]
                    - expected[expected_key]
                )

        return residuals