class HealthIndex:
    """
    Converts Sensor Fusion anomaly information into
    an engine Health Index from 0 to 100.

    100 = healthy
      0 = severely abnormal

    This is a prototype diagnostic index, not a certified
    engine-health limit.
    """

    def __init__(self):
        pass

    def calculate(self, fusion_result):
        """
        Calculate overall engine Health Index.
        """

        anomaly = fusion_result.get(
            "overall_anomaly_score",
            0.0
        )

        # Convert anomaly score to a 0-100 health scale.
        #
        # anomaly = 0  -> health = 100
        # anomaly = 1  -> health = 70
        # anomaly = 2  -> health = 40
        # anomaly >= 3 -> health = 10
        #
        # This mapping is intentionally simple for the prototype.

        health = 100.0 - (30.0 * anomaly)

        health = max(0.0, min(100.0, health))

        # ---------------------------------------------------------
        # Determine health status
        # ---------------------------------------------------------

        if health >= 85:
            status = "HEALTHY"

        elif health >= 70:
            status = "NORMAL"

        elif health >= 50:
            status = "DEGRADED"

        elif health >= 30:
            status = "WARNING"

        else:
            status = "CRITICAL"

        # ---------------------------------------------------------
        # Identify dominant subsystem
        # ---------------------------------------------------------

        subsystems = {
            "THERMAL": fusion_result.get(
                "thermal_deviation", 0.0
            ),

            "LUBRICATION": fusion_result.get(
                "lubrication_deviation", 0.0
            ),

            "COMBUSTION": fusion_result.get(
                "combustion_deviation", 0.0
            ),

            "MECHANICAL": fusion_result.get(
                "mechanical_deviation", 0.0
            ),

            "ELECTRICAL": fusion_result.get(
                "electrical_deviation", 0.0
            ),
        }

        dominant_subsystem = max(
            subsystems,
            key=subsystems.get
        )

        return {
            "health_index": health,
            "health_status": status,
            "dominant_subsystem": dominant_subsystem,
        }