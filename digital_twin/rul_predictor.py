class RULPredictor:
    """
    Prototype Remaining Useful Life (RUL) estimator.

    RUL is estimated from sustained Health Index degradation
    during an active fault episode.

    This is an illustrative estimate for the synthetic engine
    demonstration and is not a certified physical-engine RUL model.
    """

    def __init__(self):
        self.fault_start_time = None
        self.start_health = None

        self.last_rul = 500.0

        self.maintenance_threshold = 40.0

        self.health_history = []

    def update(
        self,
        timestamp_s,
        health_index,
        fault_type="NORMAL",
    ):
        timestamp_s = float(timestamp_s)
        health_index = float(health_index)

        # --------------------------------------------------
        # NORMAL OPERATION
        # --------------------------------------------------
        if fault_type == "NORMAL":
            self.fault_start_time = None
            self.start_health = None
            self.last_rul = 500.0
            self.health_history = []

            return {
                "rul_hours": 500.0,
                "degradation_rate_per_hour": 0.0,
                "trend": "STABLE",
            }

        # --------------------------------------------------
        # NEW FAULT EPISODE
        # --------------------------------------------------
        if self.fault_start_time is None:
            self.fault_start_time = timestamp_s
            self.start_health = health_index
            self.last_rul = 100.0
            self.health_history = [
                (timestamp_s, health_index)
            ]

            return {
                "rul_hours": 100.0,
                "degradation_rate_per_hour": 0.0,
                "trend": "FAULT_DETECTED",
            }

        # --------------------------------------------------
        # STORE HEALTH HISTORY
        # --------------------------------------------------
        self.health_history.append(
            (timestamp_s, health_index)
        )

        # Keep only the latest 60 seconds
        cutoff_time = timestamp_s - 60.0

        self.health_history = [
            item
            for item in self.health_history
            if item[0] >= cutoff_time
        ]

        # --------------------------------------------------
        # WAIT FOR SUFFICIENT HISTORY
        # --------------------------------------------------
        elapsed_s = (
            timestamp_s - self.fault_start_time
        )

        if elapsed_s < 20.0:
            return {
                "rul_hours": round(self.last_rul, 1),
                "degradation_rate_per_hour": 0.0,
                "trend": "FAULT_DETECTED",
            }

        if len(self.health_history) < 10:
            return {
                "rul_hours": round(self.last_rul, 1),
                "degradation_rate_per_hour": 0.0,
                "trend": "FAULT_DETECTED",
            }

        # --------------------------------------------------
        # CALCULATE SUSTAINED HEALTH DEGRADATION
        # --------------------------------------------------
        oldest_time, oldest_health = self.health_history[0]
        newest_time, newest_health = self.health_history[-1]

        window_time_s = newest_time - oldest_time

        if window_time_s <= 0:
            return {
                "rul_hours": round(self.last_rul, 1),
                "degradation_rate_per_hour": 0.0,
                "trend": "FAULT_DETECTED",
            }

        health_drop = (
            oldest_health - newest_health
        )

        # Ignore temporary health recovery.
        health_drop = max(0.0, health_drop)

        window_hours = window_time_s / 3600.0

        raw_rate = (
            health_drop / window_hours
        )

        # Convert the synthetic demonstration timescale
        # into a stable maintenance-planning rate.
        degradation_rate = raw_rate * 0.05

        degradation_rate = max(
            degradation_rate,
            2.0,
        )

        degradation_rate = min(
            degradation_rate,
            40.0,
        )

        # --------------------------------------------------
        # CALCULATE REMAINING USEFUL LIFE
        # --------------------------------------------------
        remaining_health = max(
            0.0,
            health_index - self.maintenance_threshold,
        )

        if degradation_rate <= 0:
            rul_hours = 100.0
        else:
            rul_hours = (
                remaining_health
                / degradation_rate
            )

        # Bound the displayed RUL.
        rul_hours = max(
            0.5,
            min(rul_hours, 100.0),
        )

        # RUL should not suddenly increase during
        # an active degradation episode.
        if rul_hours > self.last_rul:
            rul_hours = self.last_rul
        else:
            # Smooth the estimate.
            rul_hours = (
                0.70 * self.last_rul
                + 0.30 * rul_hours
            )

        self.last_rul = rul_hours

        # --------------------------------------------------
        # DETERMINE DEGRADATION TREND
        # --------------------------------------------------
        if degradation_rate < 10.0:
            trend = "SLOW"
        elif degradation_rate < 25.0:
            trend = "MODERATE"
        else:
            trend = "RAPID"

        return {
            "rul_hours": round(rul_hours, 1),
            "degradation_rate_per_hour": round(
                degradation_rate,
                2,
            ),
            "trend": trend,
        }