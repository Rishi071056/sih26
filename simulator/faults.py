from dataclasses import dataclass



@dataclass
class FaultState:

    fault_type: str = "NORMAL"

    severity: float = 0.0

    cooling_health: float = 1.0
    injector_health: float = 1.0
    lubrication_health: float = 1.0
    mechanical_health: float = 1.0

    misfire_severity: float = 0.0

    sensor_drift_cht_c: float = 0.0
    sensor_drift_egt_c: float = 0.0
    sensor_drift_oil_pressure_bar: float = 0.0


class FaultGenerator:

    def __init__(self):

        self.fault_type = "NORMAL"

        self.start_time = 300.0

        self.ramp_duration = 300.0

        self.max_severity = 1.0

    def set_fault(
        self,
        fault_type: str,
        start_time: float = 300.0,
        ramp_duration: float = 300.0
    ):

        self.fault_type = fault_type

        self.start_time = start_time

        self.ramp_duration = max(
            1.0,
            ramp_duration
        )

    def calculate_severity(
        self,
        time_s: float
    ) -> float:

        if self.fault_type == "NORMAL":
            return 0.0

        if time_s < self.start_time:
            return 0.0

        elapsed = (
            time_s
            - self.start_time
        )

        severity = (
            elapsed
            / self.ramp_duration
        )

        return max(
            0.0,
            min(
                self.max_severity,
                severity
            )
        )

    def update(
        self,
        time_s: float
    ) -> FaultState:

        severity = self.calculate_severity(
            time_s
        )

        state = FaultState(
            fault_type=self.fault_type,
            severity=severity
        )

        # -----------------------------------------
        # COOLING DEGRADATION
        # -----------------------------------------

        if self.fault_type == "COOLING_DEGRADATION":

            state.cooling_health = (
                1.0
                - 0.60 * severity
            )

        # -----------------------------------------
        # INJECTOR DEGRADATION
        # -----------------------------------------

        elif self.fault_type == "INJECTOR_DEGRADATION":

            state.injector_health = (
                1.0
                - 0.60 * severity
            )

        # -----------------------------------------
        # LUBRICATION DEGRADATION
        # -----------------------------------------

        elif self.fault_type == "LUBRICATION_DEGRADATION":

            state.lubrication_health = (
                1.0
                - 0.70 * severity
            )

        # -----------------------------------------
        # MISFIRE
        # -----------------------------------------

        elif self.fault_type == "MISFIRE":

            state.misfire_severity = (
                severity
            )

        # -----------------------------------------
        # MECHANICAL DEGRADATION
        # -----------------------------------------

        elif self.fault_type == "MECHANICAL_DEGRADATION":

            state.mechanical_health = (
                1.0
                - 0.70 * severity
            )

        # -----------------------------------------
        # SENSOR DRIFT
        # -----------------------------------------

        elif self.fault_type == "SENSOR_DRIFT":

            state.sensor_drift_cht_c = (
                20.0 * severity
            )

            state.sensor_drift_egt_c = (
                30.0 * severity
            )

            state.sensor_drift_oil_pressure_bar = (
                -0.5 * severity
            )

        return state