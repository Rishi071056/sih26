import os
import random

from simulator.faults import FaultGenerator
from simulator.environment import create_environment
from simulator.engine import AeroPistonEngine, EngineInputs
from simulator.mission import get_mission_state
from simulator.sensors import SensorModel
from simulator.telemetry import TelemetryLogger


# ============================================================
# CONFIGURATION
# ============================================================

DT = 0.1
SIMULATION_DURATION = 1020.0

OUTPUT_DIR = "data/missions/ml"

MISSIONS_PER_FAULT = 5

FAULT_TYPES = [
    "NORMAL",
    "COOLING_DEGRADATION",
    "INJECTOR_DEGRADATION",
    "LUBRICATION_DEGRADATION",
    "MISFIRE",
    "MECHANICAL_DEGRADATION",
    "SENSOR_DRIFT",
]


# ============================================================
# CREATE ONE MISSION VARIATION
# ============================================================

def create_mission_variation():

    return {
        # Scale the baseline altitude rather than adding
        # a fixed offset.
        #
        # This keeps takeoff at 0 ft while allowing
        # different cruise/loiter altitudes.

        "altitude_scale": random.uniform(
            0.90,
            1.10
        ),

        # Ambient temperature variation for the
        # entire mission.

        "temperature_offset": random.uniform(
            -5.0,
            5.0
        ),

        # Throttle variation for the entire mission.

        "throttle_offset": random.uniform(
            -5.0,
            5.0
        )
    }


# ============================================================
# GET VARIED MISSION CONDITIONS
# ============================================================

def get_varied_mission(
    time_s,
    variation
):

    # Get the original mission profile.

    mission = get_mission_state(
        time_s
    )

    # --------------------------------------------------------
    # ALTITUDE
    # --------------------------------------------------------

    altitude = (
        mission.altitude_ft
        * variation["altitude_scale"]
    )

    altitude = max(
        0.0,
        altitude
    )

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temperature = (
        mission.ambient_temperature_c
        + variation["temperature_offset"]
    )

    temperature = max(
        -20.0,
        min(
            40.0,
            temperature
        )
    )

    # --------------------------------------------------------
    # THROTTLE
    # --------------------------------------------------------

    throttle = (
        mission.throttle_pct
        + variation["throttle_offset"]
    )

    throttle = max(
        0.0,
        min(
            100.0,
            throttle
        )
    )

    # --------------------------------------------------------
    # ENGINE LOAD
    # --------------------------------------------------------

    load = (
        0.65 * throttle
        + 5.0
    )

    load = max(
        0.0,
        min(
            100.0,
            load
        )
    )

    return (
        mission.phase,
        throttle,
        altitude,
        temperature,
        load
    )


# ============================================================
# RUN ONE SIMULATION
# ============================================================

def run_simulation(
    output_file,
    fault_type,
    fault_start,
    fault_ramp,
    variation
):

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = AeroPistonEngine()

    # --------------------------------------------------------
    # FAULT GENERATOR
    # --------------------------------------------------------

    fault_generator = FaultGenerator()

    fault_generator.set_fault(
        fault_type,
        start_time=fault_start,
        ramp_duration=fault_ramp
    )

    # --------------------------------------------------------
    # SENSOR MODEL
    # --------------------------------------------------------

    sensors = SensorModel()

    # --------------------------------------------------------
    # TELEMETRY LOGGER
    # --------------------------------------------------------

    logger = TelemetryLogger(
        output_file
    )

    simulation_time = 0.0

    try:

        while simulation_time <= SIMULATION_DURATION:

            # =================================================
            # 1. MISSION CONDITIONS
            # =================================================

            (
                phase,
                throttle,
                altitude,
                ambient_temperature,
                engine_load
            ) = get_varied_mission(
                simulation_time,
                variation
            )

            # =================================================
            # 2. FAULT STATE
            # =================================================

            fault = fault_generator.update(
                simulation_time
            )

            # =================================================
            # 3. ENVIRONMENT
            # =================================================

            environment = create_environment(
                altitude,
                ambient_temperature
            )

            # =================================================
            # 4. ENGINE INPUTS
            # =================================================

            engine_inputs = EngineInputs(

                throttle_pct=throttle,

                engine_load_pct=engine_load,

                environment=environment,

                cooling_health=(
                    fault.cooling_health
                ),

                injector_health=(
                    fault.injector_health
                ),

                lubrication_health=(
                    fault.lubrication_health
                ),

                mechanical_health=(
                    fault.mechanical_health
                ),

                misfire_severity=(
                    fault.misfire_severity
                )
            )

            # =================================================
            # 5. ENGINE UPDATE
            # =================================================

            engine_state = engine.update(
                engine_inputs,
                DT
            )

            # =================================================
            # 6. SENSOR MODEL
            # =================================================

            sensor = sensors.read(

                engine_state,

                cht_bias=(
                    fault.sensor_drift_cht_c
                ),

                egt_bias=(
                    fault.sensor_drift_egt_c
                ),

                oil_pressure_bias=(
                    fault.sensor_drift_oil_pressure_bar
                )
            )

            # =================================================
            # 7. TELEMETRY PACKET
            # =================================================

            telemetry = {

                "timestamp_s": round(
                    simulation_time,
                    2
                ),

                "mission_phase": phase,

                # ------------------------------------------------
                # Mission / Environment
                # ------------------------------------------------

                "throttle_pct": round(
                    throttle,
                    2
                ),

                "altitude_ft": round(
                    altitude,
                    2
                ),

                "ambient_temperature_c": round(
                    ambient_temperature,
                    2
                ),

                "ambient_pressure_kpa": round(
                    environment.ambient_pressure_kpa,
                    3
                ),

                "air_density_kg_m3": round(
                    environment.air_density_kg_m3,
                    4
                ),

                "engine_load_pct": round(
                    engine_load,
                    2
                ),

                # ------------------------------------------------
                # Engine telemetry
                # ------------------------------------------------

                "rpm": round(
                    sensor.rpm,
                    2
                ),

                "cht_c": round(
                    sensor.cht_c,
                    2
                ),

                "egt_c": round(
                    sensor.egt_c,
                    2
                ),

                "oil_pressure_bar": round(
                    sensor.oil_pressure_bar,
                    3
                ),

                "oil_temperature_c": round(
                    sensor.oil_temperature_c,
                    2
                ),

                "fuel_flow_lph": round(
                    sensor.fuel_flow_lph,
                    3
                ),

                "injection_timing_deg": round(
                    sensor.injection_timing_deg,
                    3
                ),

                "vibration_mm_s": round(
                    sensor.vibration_mm_s,
                    3
                ),

                "battery_voltage_v": round(
                    sensor.battery_voltage_v,
                    3
                ),

                "alternator_current_a": round(
                    sensor.alternator_current_a,
                    3
                ),

                # ------------------------------------------------
                # Ground-truth fault information
                #
                # These are labels for ML training.
                # They must NOT be used as ML input features.
                # ------------------------------------------------

                "fault_type": fault.fault_type,

                "fault_severity": round(
                    fault.severity,
                    3
                ),

                "cooling_health": round(
                    fault.cooling_health,
                    3
                ),

                "injector_health": round(
                    fault.injector_health,
                    3
                ),

                "lubrication_health": round(
                    fault.lubrication_health,
                    3
                ),

                "mechanical_health": round(
                    fault.mechanical_health,
                    3
                ),

                "misfire_severity": round(
                    fault.misfire_severity,
                    3
                )
            }

            # =================================================
            # 8. SAVE TELEMETRY
            # =================================================

            logger.write(
                telemetry
            )

            # =================================================
            # 9. ADVANCE SIMULATION
            # =================================================

            simulation_time += DT

    finally:

        logger.close()


# ============================================================
# MAIN DATASET GENERATOR
# ============================================================

def main():

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(" ML MISSION DATASET GENERATOR")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Fixed seed
    #
    # This makes the dataset reproducible.
    # --------------------------------------------------------

    random.seed(42)

    mission_number = 1

    # ========================================================
    # GENERATE EACH FAULT CLASS
    # ========================================================

    for fault_type in FAULT_TYPES:

        print()
        print(
            f"Generating {fault_type} missions..."
        )

        for _ in range(
            MISSIONS_PER_FAULT
        ):

            # ------------------------------------------------
            # Mission variation
            #
            # Created ONCE per mission.
            # ------------------------------------------------

            variation = (
                create_mission_variation()
            )

            # ------------------------------------------------
            # Fault timing
            # ------------------------------------------------

            if fault_type == "NORMAL":

                # No fault during a normal mission.

                fault_start = 99999.0
                fault_ramp = 1.0

            else:

                # Fault begins at a different time
                # for each mission.

                fault_start = random.uniform(
                    100.0,
                    700.0
                )

                fault_ramp = random.uniform(
                    60.0,
                    300.0
                )

            # ------------------------------------------------
            # Filename
            # ------------------------------------------------

            filename = (
                f"{fault_type.lower()}_"
                f"{mission_number:03d}.csv"
            )

            output_file = os.path.join(
                OUTPUT_DIR,
                filename
            )

            # ------------------------------------------------
            # Run simulation
            # ------------------------------------------------

            run_simulation(
                output_file,
                fault_type,
                fault_start,
                fault_ramp,
                variation
            )

            # ------------------------------------------------
            # Print information
            # ------------------------------------------------

            print(
                f"  Created: {filename}"
            )

            print(
                f"    Altitude scale: "
                f"{variation['altitude_scale']:.3f}x"
            )

            print(
                f"    Temperature offset: "
                f"{variation['temperature_offset']:.1f} C"
            )

            print(
                f"    Throttle offset: "
                f"{variation['throttle_offset']:.1f} %"
            )

            if fault_type != "NORMAL":

                print(
                    f"    Fault start: "
                    f"{fault_start:.1f} s"
                )

                print(
                    f"    Fault ramp: "
                    f"{fault_ramp:.1f} s"
                )

            mission_number += 1

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(" DATASET GENERATION COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Total missions: "
        f"{len(FAULT_TYPES) * MISSIONS_PER_FAULT}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()