class DegradationAnalyzer:
    """
    Calculates subsystem-specific degradation trends.

    A positive trend means the corresponding residual is increasing
    over time.

    This is a prototype trend-analysis module, not a certified
    remaining-life model.
    """

    def __init__(self, window_size=100):
        # 100 samples at 10 Hz = approximately 10 seconds
        self.window_size = window_size

    def calculate_slope(self, values, timestamps):
        """
        Calculate linear regression slope per second.
        """

        if len(values) < 2:
            return 0.0

        n = len(values)

        mean_x = sum(timestamps) / n
        mean_y = sum(values) / n

        numerator = sum(
            (timestamps[i] - mean_x)
            * (values[i] - mean_y)
            for i in range(n)
        )

        denominator = sum(
            (timestamps[i] - mean_x) ** 2
            for i in range(n)
        )

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def analyze(self, timestamps, residual_history):
        """
        Calculate parameter and subsystem-specific trends.

        Expected residual_history:

        {
            "cht_c_residual": [...],
            "oil_temperature_c_residual": [...],
            "egt_c_residual": [...],
            "vibration_mm_s_residual": [...],
            "rpm_residual": [...],
            "oil_pressure_bar_residual": [...]
        }
        """

        timestamps = timestamps[-self.window_size:]

        parameter_trends = {}

        # ---------------------------------------------------------
        # Individual parameter trends
        # ---------------------------------------------------------

        for parameter, values in residual_history.items():

            values = values[-self.window_size:]

            count = min(
                len(timestamps),
                len(values)
            )

            if count < 2:
                slope = 0.0
            else:
                slope = self.calculate_slope(
                    values[-count:],
                    timestamps[-count:]
                )

            parameter_trends[
                parameter + "_trend"
            ] = slope

        # ---------------------------------------------------------
        # THERMAL SUBSYSTEM
        # ---------------------------------------------------------

        cht_trend = parameter_trends.get(
            "cht_c_residual_trend",
            0.0
        )

        oil_temp_trend = parameter_trends.get(
            "oil_temperature_c_residual_trend",
            0.0
        )

        egt_trend = parameter_trends.get(
            "egt_c_residual_trend",
            0.0
        )

        thermal_trend = (
            0.60 * cht_trend
            + 0.30 * oil_temp_trend
            + 0.10 * egt_trend
        )

        # ---------------------------------------------------------
        # LUBRICATION SUBSYSTEM
        # ---------------------------------------------------------

        oil_pressure_trend = parameter_trends.get(
            "oil_pressure_bar_residual_trend",
            0.0
        )

        lubrication_trend = (
            0.60 * oil_pressure_trend
            + 0.40 * oil_temp_trend
        )

        # ---------------------------------------------------------
        # MECHANICAL SUBSYSTEM
        # ---------------------------------------------------------

        rpm_trend = parameter_trends.get(
            "rpm_residual_trend",
            0.0
        )

        vibration_trend = parameter_trends.get(
            "vibration_mm_s_residual_trend",
            0.0
        )

        mechanical_trend = (
            0.25 * rpm_trend
            + 0.75 * vibration_trend
        )

        # ---------------------------------------------------------
        # COMBUSTION SUBSYSTEM
        # ---------------------------------------------------------

        fuel_flow_trend = parameter_trends.get(
            "fuel_flow_lph_residual_trend",
            0.0
        )

        injection_timing_trend = parameter_trends.get(
            "injection_timing_deg_residual_trend",
            0.0
        )

        combustion_trend = (
            0.40 * fuel_flow_trend
            + 0.25 * injection_timing_trend
            + 0.35 * egt_trend
        )

        # ---------------------------------------------------------
        # ELECTRICAL SUBSYSTEM
        # ---------------------------------------------------------

        battery_trend = parameter_trends.get(
            "battery_voltage_v_residual_trend",
            0.0
        )

        alternator_trend = parameter_trends.get(
            "alternator_current_a_residual_trend",
            0.0
        )

        electrical_trend = (
            0.50 * battery_trend
            + 0.50 * alternator_trend
        )

        return {
            **parameter_trends,

            "thermal_degradation_trend":
                thermal_trend,

            "lubrication_degradation_trend":
                lubrication_trend,

            "mechanical_degradation_trend":
                mechanical_trend,

            "combustion_degradation_trend":
                combustion_trend,

            "electrical_degradation_trend":
                electrical_trend,
        }