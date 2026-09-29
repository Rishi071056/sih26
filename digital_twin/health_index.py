class HealthIndex:
    """
    Engine Health Index: 0 to 100.

    100 = healthy
    85-100 = HEALTHY
    70-84 = DEGRADED
    40-69 = WARNING
    0-39 = CRITICAL

    Combines:
    - Digital Twin residual deviation
    - ML fault classification confidence

    Simulator fault truth is NOT used in the operational score.
    """

    def __init__(self):
        self.previous_health = 100.0

        self.deadbands = {
            "rpm_residual": 100.0,
            "cht_c_residual": 3.0,
            "egt_c_residual": 10.0,
            "oil_pressure_bar_residual": 0.20,
            "oil_temperature_c_residual": 3.0,
            "fuel_flow_lph_residual": 0.75,
            "injection_timing_deg_residual": 0.50,
            "vibration_mm_s_residual": 0.20,
            "battery_voltage_v_residual": 0.30,
            "alternator_current_a_residual": 0.75,
        }

        self.limits = {
            "rpm_residual": 500.0,
            "cht_c_residual": 20.0,
            "egt_c_residual": 50.0,
            "oil_pressure_bar_residual": 1.0,
            "oil_temperature_c_residual": 20.0,
            "fuel_flow_lph_residual": 5.0,
            "injection_timing_deg_residual": 5.0,
            "vibration_mm_s_residual": 1.5,
            "battery_voltage_v_residual": 2.0,
            "alternator_current_a_residual": 5.0,
        }

        self.weights = {
            "rpm_residual": 0.05,
            "cht_c_residual": 0.20,
            "egt_c_residual": 0.15,
            "oil_pressure_bar_residual": 0.15,
            "oil_temperature_c_residual": 0.15,
            "fuel_flow_lph_residual": 0.05,
            "injection_timing_deg_residual": 0.10,
            "vibration_mm_s_residual": 0.05,
            "battery_voltage_v_residual": 0.05,
            "alternator_current_a_residual": 0.05,
        }

    def _calculate_residual_score(self, residuals):
        score = 0.0

        for parameter, weight in self.weights.items():
            value = abs(residuals.get(parameter, 0.0))

            deadband = self.deadbands[parameter]
            limit = self.limits[parameter]

            excess = max(0.0, value - deadband)
            available_range = max(0.001, limit - deadband)

            normalized = min(
                excess / available_range,
                1.0
            )

            score += normalized * weight

        return min(score, 1.0)

    def calculate(
        self,
        residuals,
        fault_confidence=0.0,
        fault_type="NORMAL"
    ):
        residual_score = self._calculate_residual_score(
            residuals
        )

        if fault_type == "NORMAL":
            ml_score = 0.0
        else:
            ml_score = max(
                0.0,
                min(float(fault_confidence), 1.0)
            )

        twin_health = 100.0 * (1.0 - residual_score)
        ml_health = 100.0 * (1.0 - ml_score)

        if fault_type == "NORMAL":
            raw_health = (
                0.85 * twin_health
                + 0.15 * ml_health
            )
        else:
            raw_health = (
                0.45 * twin_health
                + 0.55 * ml_health
            )

        # Smooth the health index to avoid sudden jumps.
        health = (
            0.70 * self.previous_health
            + 0.30 * raw_health
        )

        health = max(
            0.0,
            min(100.0, health)
        )

        self.previous_health = health

        if health >= 85:
            status = "HEALTHY"
        elif health >= 70:
            status = "DEGRADED"
        elif health >= 40:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return {
            "health_index": round(health, 1),
            "status": status,
            "fault_type": fault_type,
            "residual_score": round(residual_score, 3),
            "twin_health": round(twin_health, 1),
            "ml_health": round(ml_health, 1),
        }