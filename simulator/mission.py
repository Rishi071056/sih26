from dataclasses import dataclass


@dataclass
class MissionState:
    time_s: float
    phase: str
    throttle_pct: float
    altitude_ft: float
    ambient_temperature_c: float
    engine_load_pct: float


def get_mission_state(time_s: float) -> MissionState:

    # ---------------------------------------------------------
    # TAKEOFF
    # ---------------------------------------------------------

    if time_s < 60:

        progress = time_s / 60.0

        throttle = 85.0 + 10.0 * progress

        altitude = 0.0 + 1500.0 * progress

        phase = "TAKEOFF"

    # ---------------------------------------------------------
    # CLIMB
    # ---------------------------------------------------------

    elif time_s < 180:

        progress = (time_s - 60.0) / 120.0

        throttle = 85.0 - 10.0 * progress

        altitude = 1500.0 + 8500.0 * progress

        phase = "CLIMB"

    # ---------------------------------------------------------
    # CRUISE
    # ---------------------------------------------------------

    elif time_s < 600:

        throttle = 65.0

        altitude = 10000.0

        phase = "CRUISE"

    # ---------------------------------------------------------
    # LOITER
    # ---------------------------------------------------------

    elif time_s < 900:

        throttle = 55.0

        altitude = 12000.0

        phase = "LOITER"

    # ---------------------------------------------------------
    # DESCENT
    # ---------------------------------------------------------

    elif time_s < 960:

        progress = (time_s - 900.0) / 60.0

        throttle = 55.0 - 15.0 * progress

        altitude = 12000.0 - 9000.0 * progress

        phase = "DESCENT"

    # ---------------------------------------------------------
    # LANDING
    # ---------------------------------------------------------

    else:

        progress = min(
            (time_s - 960.0) / 60.0,
            1.0
        )

        throttle = 40.0 - 10.0 * progress

        altitude = 3000.0 - 3000.0 * progress

        phase = "LANDING"

    # ---------------------------------------------------------
    # Ambient temperature
    # ---------------------------------------------------------

    # Simple altitude-dependent temperature.
    ambient_temperature = 30.0 - (
        altitude / 1000.0 * 2.0
    )

    # Keep it reasonable.
    ambient_temperature = max(
        -20.0,
        min(40.0, ambient_temperature)
    )

    # ---------------------------------------------------------
    # Engine load
    # ---------------------------------------------------------

    load = (
        0.65 * throttle
        + 5.0
    )

    load = max(
        0.0,
        min(100.0, load)
    )

    return MissionState(
        time_s=time_s,
        phase=phase,
        throttle_pct=throttle,
        altitude_ft=altitude,
        ambient_temperature_c=ambient_temperature,
        engine_load_pct=load
    )