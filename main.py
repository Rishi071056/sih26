import time
from simulator.faults import FaultGenerator

from simulator.environment import (
    create_environment
)

from simulator.engine import (
    AeroPistonEngine,
    EngineInputs
)

from simulator.mission import (
    get_mission_state
)

from simulator.sensors import (
    SensorModel
)

from simulator.telemetry import (
    TelemetryLogger,
    telemetry_to_json
)


# ============================================================
# CONFIGURATION
# ============================================================

DT = 0.1              # 10 Hz
SIMULATION_DURATION = 180.0

CSV_FILE = (
    "data/missions/mission_001.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    engine = AeroPistonEngine()

    fault_generator = FaultGenerator()

    fault_generator.set_fault(
        "COOLING_DEGRADATION",
        start_time=8.0,
        ramp_duration=30.0
    )

    sensors = SensorModel()

    logger = TelemetryLogger(
        CSV_FILE
    )

    print()
    print("=" * 80)
    print(" VIRTUAL AERO-PISTON ENGINE SIMULATOR")
    print("=" * 80)
    print()
    print("Simulation rate : 10 Hz")
    print("Engine model    : Reduced-order physics-informed")
    print("Telemetry       : Synthetic / simulated")
    print()
    print("Starting mission...")
    print()

    simulation_time = 0.0

    try:

        while simulation_time <= SIMULATION_DURATION:

            # ------------------------------------------------
            # 1. GET MISSION CONDITIONS
            # ------------------------------------------------

            mission = get_mission_state(
                simulation_time
            )

            fault = fault_generator.update(
                simulation_time
            )

            # ------------------------------------------------
            # 2. CALCULATE ENVIRONMENT
            # ------------------------------------------------

            environment = create_environment(
                mission.altitude_ft,
                mission.ambient_temperature_c
            )

            # ------------------------------------------------
            # 3. CREATE ENGINE INPUTS
            # ------------------------------------------------

            engine_inputs = EngineInputs(

                throttle_pct=(
                    mission.throttle_pct
                ),

                engine_load_pct=(
                    mission.engine_load_pct
                ),

                environment=environment,

                # Healthy engine for V1.
                cooling_health=fault.cooling_health,
                injector_health=fault.injector_health,
                lubrication_health=fault.lubrication_health,
                mechanical_health=fault.mechanical_health,
                misfire_severity=fault.misfire_severity
            )

            # ------------------------------------------------
            # 4. UPDATE ENGINE
            # ------------------------------------------------

            engine_state = engine.update(
                engine_inputs,
                DT
            )

            # ------------------------------------------------
            # 5. SENSOR MODEL
            # ------------------------------------------------

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

            # ------------------------------------------------
            # 6. BUILD TELEMETRY PACKET
            # ------------------------------------------------

            telemetry = {

                "timestamp_s": round(
                    simulation_time,
                    2
                ),

                "mission_phase": (
                    mission.phase
                ),

                # Inputs
                "throttle_pct": round(
                    mission.throttle_pct,
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

                "engine_load_pct": round(
                    mission.engine_load_pct,
                    2
                ),

                # Engine telemetry
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

            # ------------------------------------------------
            # 7. SAVE CSV
            # ------------------------------------------------

            logger.write(
                telemetry
            )

            # ------------------------------------------------
            # 8. PRINT LIVE TELEMETRY
            # ------------------------------------------------

            print(
                f"\r"
                f"T={simulation_time:6.1f}s | "
                f"{mission.phase:<9} | "
                f"FAULT={fault.fault_type:<24} | "
                f"SEV={fault.severity:.2f} | "
                f"RPM={sensor.rpm:7.0f} | "
                f"CHT={sensor.cht_c:6.1f}C | "
                f"EGT={sensor.egt_c:6.1f}C | "
                f"OilP={sensor.oil_pressure_bar:4.2f}bar | "
                f"OilT={sensor.oil_temperature_c:5.1f}C | "
                f"Fuel={sensor.fuel_flow_lph:5.1f}L/h | "
                f"Vib={sensor.vibration_mm_s:4.2f}",
                end=""
            )

            # ------------------------------------------------
            # 9. REAL-TIME DELAY
            # ------------------------------------------------

            time.sleep(DT)

            simulation_time += DT

    except KeyboardInterrupt:

        print()
        print()
        print("Simulation stopped by user.")

    finally:

        logger.close()

    print()
    print()
    print("=" * 80)
    print(" SIMULATION COMPLETE")
    print("=" * 80)
    print()
    print(
        f"Telemetry saved to: {CSV_FILE}"
    )
    print()


if __name__ == "__main__":
    main()