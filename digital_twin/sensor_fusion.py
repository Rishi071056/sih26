from baseline import BASELINE
class SensorFusion:
    """
    Sensor Fusion module for the Aero-Piston Engine Digital Twin.

    Purpose:
        1. Normalize Digital Twin residuals.
        2. Preserve residual direction.
        3. Combine related sensors into subsystem indicators.
        4. Detect persistent abnormal behavior using a rolling history.

    Important:
        This module does NOT use:
            - fault_type
            - fault_severity
            - cooling_health
            - injector_health
            - lubrication_health
            - mechanical_health

        These are ground-truth variables from the simulator and must
        remain hidden from the diagnostic system.
    """

    def __init__(self, history_length=50):

        # Number of recent samples used for temporal persistence.
        # At 10 Hz, 50 samples = approximately 5 seconds.
        self.history_length = history_length

        self.history = []

        # Approximate prototype scales for residual normalization.
        #
        # These are NOT certified engine limits.
        # They are sensitivity/reference scales for our prototype.
        self.scales = {
            "rpm_residual": 500.0,
            "cht_c_residual": 20.0,
            "egt_c_residual": 50.0,
            "oil_pressure_bar_residual": 0.8,
            "oil_temperature_c_residual": 15.0,
            "fuel_flow_lph_residual": 3.0,
            "injection_timing_deg_residual": 2.0,
            "vibration_mm_s_residual": 2.0,
            "battery_voltage_v_residual": 1.0,
            "alternator_current_a_residual": 3.0,
        }

    def normalize(self, residual_name, value, mission_phase):
        """
        Convert a residual into a phase-aware normalized deviation.

        Uses healthy mean and standard deviation for the current
        mission phase.

        This is effectively a prototype Z-score.
        """

        phase_baseline = BASELINE.get(mission_phase)

        if phase_baseline is None:
            return 0.0

        parameter_baseline = phase_baseline.get(residual_name)

        if parameter_baseline is None:
            return 0.0

        mean = parameter_baseline["mean"]
        std = parameter_baseline["std"]

        # Protect against division by zero.
        if std < 1e-6:
            std = 1e-6

        z_score = (value - mean) / std

        # Limit extreme values so one sensor cannot dominate fusion.
        z_score = max(-3.0, min(3.0, z_score))

        return z_score

    def calculate_current(self, residuals,mission_phase):
        """
        Calculate the current fused subsystem scores.

        Returns signed subsystem deviations and magnitude scores.
        """

        normalized = {}

        for name, value in residuals.items():
            normalized[name] = self.normalize(name, value, mission_phase)

        # ---------------------------------------------------------
        # THERMAL SUBSYSTEM
        # ---------------------------------------------------------

        thermal = (
            0.45 * normalized.get("cht_c_residual", 0.0)
            + 0.35 * normalized.get("oil_temperature_c_residual", 0.0)
            + 0.20 * normalized.get("egt_c_residual", 0.0)
        )

        # ---------------------------------------------------------
        # LUBRICATION SUBSYSTEM
        # ---------------------------------------------------------

        lubrication = (
            0.60 * normalized.get("oil_pressure_bar_residual", 0.0)
            + 0.40 * normalized.get("oil_temperature_c_residual", 0.0)
        )

        # ---------------------------------------------------------
        # COMBUSTION / FUEL SUBSYSTEM
        # ---------------------------------------------------------

        combustion = (
            0.40 * normalized.get("fuel_flow_lph_residual", 0.0)
            + 0.25 * normalized.get("injection_timing_deg_residual", 0.0)
            + 0.35 * normalized.get("egt_c_residual", 0.0)
        )

        # ---------------------------------------------------------
        # MECHANICAL SUBSYSTEM
        # ---------------------------------------------------------

        mechanical = (
            0.50 * normalized.get("rpm_residual", 0.0)
            + 0.50 * normalized.get("vibration_mm_s_residual", 0.0)
        )

        # ---------------------------------------------------------
        # ELECTRICAL SUBSYSTEM
        # ---------------------------------------------------------

        electrical = (
            0.50 * normalized.get("battery_voltage_v_residual", 0.0)
            + 0.50 * normalized.get("alternator_current_a_residual", 0.0)
        )

        # Store signed subsystem values.
        subsystem_signed = {
            "thermal": thermal,
            "lubrication": lubrication,
            "combustion": combustion,
            "mechanical": mechanical,
            "electrical": electrical,
        }

        # Magnitude of each subsystem deviation.
        subsystem_magnitude = {
            name + "_deviation": min(abs(value), 3.0)
            for name, value in subsystem_signed.items()
        }

        # ---------------------------------------------------------
        # OVERALL CURRENT ANOMALY
        # ---------------------------------------------------------

        overall_current = (
            0.30 * abs(thermal)
            + 0.20 * abs(lubrication)
            + 0.20 * abs(combustion)
            + 0.20 * abs(mechanical)
            + 0.10 * abs(electrical)
        )

        return {
            "thermal_signed": thermal,
            "lubrication_signed": lubrication,
            "combustion_signed": combustion,
            "mechanical_signed": mechanical,
            "electrical_signed": electrical,

            **subsystem_magnitude,

            "current_anomaly_score": min(overall_current, 3.0),
        }

    def calculate(self, residuals, mission_phase):
        """
        Main Sensor Fusion function.

        Separates:
            - current anomaly
            - persistence
            - degradation trend

        This prevents a single transient sample from dominating
        the final diagnostic signal.
        """

        current = self.calculate_current(
            residuals,
            mission_phase
        )

        current_score = current["current_anomaly_score"]

        # Store current score
        self.history.append(current_score)

        # Keep only recent history
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # ---------------------------------------------------------
        # PERSISTENCE
        # ---------------------------------------------------------

        persistence_score = sum(self.history) / len(self.history)

        # ---------------------------------------------------------
        # RECENT TREND
        # ---------------------------------------------------------

        trend_score = 0.0

        if len(self.history) >= 10:

            recent = self.history[-5:]
            previous = self.history[-10:-5]

            recent_average = sum(recent) / len(recent)
            previous_average = sum(previous) / len(previous)

            trend_score = recent_average - previous_average

        # ---------------------------------------------------------
        # FINAL SCORE
        # ---------------------------------------------------------
        #
        # Current behavior is the strongest component.
        # Persistence confirms that it is not just a transient.
        # Positive trend provides additional evidence of degradation.
        #

        trend_contribution = max(trend_score, 0.0)

        overall_anomaly_score = (
            0.50 * current_score
            + 0.35 * persistence_score
            + 0.15 * trend_contribution
        )

        overall_anomaly_score = min(
            overall_anomaly_score,
            3.0
        )

        return {
            **current,

            "persistence_score": persistence_score,

            "trend_score": trend_score,

            "overall_anomaly_score": overall_anomaly_score,
        }