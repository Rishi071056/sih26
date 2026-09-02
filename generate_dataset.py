import os

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

# 180 seconds = 3 minutes per mission
SIMULATION_DURATION = 180.0

OUTPUT_DIR = "data/missions"


# ============================================================
# FAULTS TO GENERATE
# ============================================================

FAULTS = [
    ("NORMAL", 0.0, 0.0),

    ("COOLING_DEGRADATION", 8.0, 30.0),

    ("INJECTOR_DEGRADATION", 8.0, 30.0),

    ("LUBRICATION_DEGRADATION", 8.0, 30.0),

    ("MISFIRE", 8.0, 30.0),

    ("MECHANICAL_FAILURE", 8.0, 30.0),

    ("SENSOR_DRIFT", 8.0, 30.0),
]


# ============================================================
# TELEMETRY GENERATION
# ============================================================

def generate_mission(
    fault_type,
    start_time,
    ramp_duration,
    mission_number
):

    print()
    print("=" * 70)
    print(f"Generating mission: {mission_number}")
    print(f"Fault: {fault_type}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create fresh engine for every mission
    # --------------------------------------------------------

    engine = AeroPistonEngine()

    fault_generator = FaultGenerator()

    fault_generator.set_fault(
        fault_type,
        start_time=start_time,
        ramp_duration=ramp_duration
    )

    sensors = SensorModel()

    # --------------------------------------------------------
    # File name
    # --------------------------------------------------------

    safe_name = fault_type.lower()

    csv_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_{mission_number:03d}.csv"
    )

    logger = TelemetryLogger(csv_file)

    simulation_time = 0.0

    try:

        while simulation_time <= SIMULATION_DURATION:

            # =================================================
            # 1. MISSION
            # =================================================

            mission = get_mission_state(
                simulation_time
            )

            # =================================================
            # 2. FAULT
            # =================================================

            fault = fault_generator.update(
                simulation_time
            )

            # =================================================
            # 3. ENVIRONMENT
            # =================================================

            environment = create_environment(
                mission.altitude_ft,
                mission.ambient_temperature_c
            )

            # =================================================
            # 4. ENGINE INPUTS
            # =================================================

            engine_inputs = EngineInputs(

                throttle_pct=(
                    mission.throttle_pct
                ),

                engine_load_pct=(
                    mission.engine_load_pct
                ),

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
            # 7. TELEMETRY
            # =================================================

            telemetry = {

                # Time
                "timestamp_s": round(
                    simulation_time,
                    2
                ),

                # Mission
                "mission_phase": (
                    mission.phase
                ),

                # Operating conditions
                "throttle_pct": round(
                    mission.throttle_pct,
                    2
                ),

                "engine_load_pct": round(
                    mission.engine_load_pct,
                    2
                ),

                "altitude_ft": round(
                    mission.altitude_ft,
                    2
                ),

                "ambient_temperature_c": round(
                    mission.ambient_temperature_c,
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

                # =================================================
                # TRUE ENGINE VALUES
                # =================================================

                "true_rpm": round(
                    engine_state.rpm,
                    2
                ),

                "true_cht_c": round(
                    engine_state.cht_c,
                    2
                ),

                "true_egt_c": round(
                    engine_state.egt_c,
                    2
                ),

                "true_oil_pressure_bar": round(
                    engine_state.oil_pressure_bar,
                    3
                ),

                "true_oil_temperature_c": round(
                    engine_state.oil_temperature_c,
                    2
                ),

                "true_fuel_flow_lph": round(
                    engine_state.fuel_flow_lph,
                    3
                ),

                "true_injection_timing_deg": round(
                    engine_state.injection_timing_deg,
                    3
                ),

                "true_vibration_mm_s": round(
                    engine_state.vibration_mm_s,
                    3
                ),

                "true_battery_voltage_v": round(
                    engine_state.battery_voltage_v,
                    3
                ),

                "true_alternator_current_a": round(
                    engine_state.alternator_current_a,
                    3
                ),

                # =================================================
                # SENSOR / OBSERVED VALUES
                # =================================================

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

                # =================================================
                # FAULT LABEL
                # =================================================

                "fault_type": (
                    fault.fault_type
                ),

                "fault_severity": round(
                    fault.severity,
                    3
                ),

                # =================================================
                # HEALTH LABELS
                # =================================================

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
                ),

                # =================================================
                # SENSOR FAULT GROUND TRUTH
                # =================================================

                "sensor_drift_cht_c": round(
                    fault.sensor_drift_cht_c,
                    3
                ),

                "sensor_drift_egt_c": round(
                    fault.sensor_drift_egt_c,
                    3
                ),

                "sensor_drift_oil_pressure_bar": round(
                    fault.sensor_drift_oil_pressure_bar,
                    3
                )
            }

            # =================================================
            # 8. SAVE
            # =================================================

            logger.write(
                telemetry
            )

            simulation_time += DT

    finally:

        logger.close()

    print(
        f"Saved: {csv_file}"
    )


# ============================================================
# MAIN DATASET GENERATOR
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(" AERO-PISTON ENGINE DATASET GENERATOR")
    print("=" * 70)
    print()

    for fault_type, start_time, ramp_duration in FAULTS:

        generate_mission(
            fault_type,
            start_time,
            ramp_duration,
            1
        )

    print()
    print("=" * 70)
    print(" DATASET GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(
        f"Dataset location: {OUTPUT_DIR}"
    )
    print()


if __name__ == "__main__":
    main()