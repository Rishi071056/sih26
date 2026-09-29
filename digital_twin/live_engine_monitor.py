import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import os
import time

from simulator.faults import FaultGenerator
from simulator.environment import create_environment
from simulator.engine import AeroPistonEngine, EngineInputs
from simulator.mission import get_mission_state
from simulator.sensors import SensorModel

from digital_twin.twin_engine import DigitalTwinEngine
from digital_twin.residuals import ResidualAnalyzer
from digital_twin.live_fault_classifier import LiveFaultClassifier
from digital_twin.health_index import HealthIndex
from digital_twin.rul_predictor import RULPredictor


# =========================================================
# SETTINGS
# =========================================================

DT = 0.1
SIMULATION_DURATION = 180.0

LIVE_CSV = "data/live_telemetry.csv"
CONTROL_FILE = "data/fault_control.json"


# =========================================================
# SUPPORTED FAULTS
# =========================================================

SUPPORTED_FAULTS = [
    "NORMAL",
    "COOLING_DEGRADATION",
    "INJECTOR_DEGRADATION",
    "LUBRICATION_DEGRADATION",
    "MECHANICAL_DEGRADATION",
    "MISFIRE",
    "SENSOR_DRIFT",
]


# =========================================================
# DASHBOARD CONTROL
# =========================================================

def create_default_control():

    control = {
        "fault_type": "NORMAL",
        "severity": 0.0,
    }

    os.makedirs("data", exist_ok=True)

    with open(
        CONTROL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            control,
            file,
            indent=2
        )


def read_fault_control():

    if not os.path.exists(CONTROL_FILE):
        create_default_control()

    try:

        with open(
            CONTROL_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            control = json.load(file)

        fault_type = str(
            control.get(
                "fault_type",
                "NORMAL"
            )
        )

        severity = float(
            control.get(
                "severity",
                0.0
            )
        )

        if fault_type not in SUPPORTED_FAULTS:
            fault_type = "NORMAL"

        severity = max(
            0.0,
            min(
                severity,
                1.0
            )
        )

        return fault_type, severity

    except Exception:

        return "NORMAL", 0.0


# =========================================================
# ML RESULT HANDLING
# =========================================================

def extract_ml_result(
    ml_result,
    classifier
):

    # The classifier needs 50 samples before its first
    # prediction. During that period preserve the last
    # stable classifier state instead of forcing NORMAL.

    if ml_result is None:

        return (
            classifier.current_fault,
            classifier.current_confidence
        )

    if not isinstance(
        ml_result,
        dict
    ):

        return (
            classifier.current_fault,
            classifier.current_confidence
        )

    confidence = float(
        ml_result.get(
            "confidence",
            ml_result.get(
                "fault_confidence",
                0.0
            )
        )
    )

    # The actual LiveFaultClassifier returns "fault".
    # Keep fallbacks for compatibility.
    fault_type = ml_result.get(
        "fault",
        ml_result.get(
            "fault_type",
            ml_result.get(
                "prediction",
                classifier.current_fault
            )
        )
    )

    if fault_type is None:
        fault_type = classifier.current_fault

    return str(fault_type), confidence


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # COMPONENTS
    # -----------------------------------------------------

    engine = AeroPistonEngine()

    fault_generator = FaultGenerator()

    sensors = SensorModel()

    twin = DigitalTwinEngine()

    residual_analyzer = ResidualAnalyzer()

    classifier = LiveFaultClassifier()

    health_index = HealthIndex()

    rul_predictor = RULPredictor()

    # -----------------------------------------------------
    # LIVE CSV
    # -----------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )

    live_file = open(
        LIVE_CSV,
        "w",
        newline="",
        encoding="utf-8"
    )

    live_writer = None

    # -----------------------------------------------------
    # INITIAL CONTROL
    # -----------------------------------------------------

    current_control = None

    fault_type = "NORMAL"

    fault_severity = 0.0

    simulation_time = 0.0

    print("=" * 75)
    print("AERO-PISTON ENGINE LIVE MONITOR")
    print("=" * 75)
    print()

    print(
        f"Live telemetry: {LIVE_CSV}"
    )

    print(
        f"Fault control:  {CONTROL_FILE}"
    )

    print()

    print(
        "Waiting for dashboard fault control..."
    )

    print()

    try:

        # =================================================
        # SIMULATION LOOP
        # =================================================

        while simulation_time <= SIMULATION_DURATION:

            # ---------------------------------------------
            # READ DASHBOARD CONTROL
            # ---------------------------------------------

            requested_fault, requested_severity = (
                read_fault_control()
            )

            new_control = (
                requested_fault,
                round(
                    requested_severity,
                    3
                )
            )

            # ---------------------------------------------
            # APPLY NEW CONTROL
            # ---------------------------------------------

            if new_control != current_control:

                fault_type = requested_fault

                fault_severity = requested_severity

                # IMPORTANT:
                # Every new dashboard scenario starts a clean
                # diagnostic episode. This prevents the ML
                # buffer, Health Index smoothing, and RUL
                # history from carrying the previous fault
                # into the new scenario.

                classifier = LiveFaultClassifier()

                health_index = HealthIndex()

                rul_predictor = RULPredictor()

                if fault_type == "NORMAL":

                    fault_generator.set_fault(
                        "NORMAL"
                    )

                else:

                    fault_generator.set_fault(
                        fault_type,
                        start_time=simulation_time,
                        ramp_duration=1.0
                    )

                    fault_generator.max_severity = (
                        fault_severity
                    )

                current_control = new_control

                print()
                print(
                    ">>> DASHBOARD COMMAND"
                )

                print(
                    f"    Fault:    {fault_type}"
                )

                print(
                    f"    Severity: "
                    f"{fault_severity * 100:.1f}%"
                )

                print(
                    "    Diagnostic pipeline reset"
                )

                print()

            # ---------------------------------------------
            # MISSION
            # ---------------------------------------------

            mission = get_mission_state(
                simulation_time
            )

            # ---------------------------------------------
            # FAULT STATE
            # ---------------------------------------------

            fault = fault_generator.update(
                simulation_time
            )

            # ---------------------------------------------
            # ENVIRONMENT
            # ---------------------------------------------

            environment = create_environment(
                mission.altitude_ft,
                mission.ambient_temperature_c
            )

            # ---------------------------------------------
            # ENGINE INPUTS
            # ---------------------------------------------

            engine_inputs = EngineInputs(

                throttle_pct=mission.throttle_pct,

                engine_load_pct=mission.engine_load_pct,

                environment=environment,

                cooling_health=fault.cooling_health,

                injector_health=fault.injector_health,

                lubrication_health=fault.lubrication_health,

                mechanical_health=fault.mechanical_health,

                misfire_severity=fault.misfire_severity
            )

            # ---------------------------------------------
            # ENGINE
            # ---------------------------------------------

            engine_state = engine.update(
                engine_inputs,
                DT
            )

            # ---------------------------------------------
            # SENSOR MODEL
            # ---------------------------------------------

            sensor = sensors.read(

                engine_state,

                cht_bias=fault.sensor_drift_cht_c,

                egt_bias=fault.sensor_drift_egt_c,

                oil_pressure_bias=(
                    fault.sensor_drift_oil_pressure_bar
                )
            )

            # ---------------------------------------------
            # ACTUAL TELEMETRY
            # ---------------------------------------------

            telemetry = {

                "timestamp_s":
                    round(
                        simulation_time,
                        2
                    ),

                "mission_phase":
                    mission.phase,

                "throttle_pct":
                    round(
                        mission.throttle_pct,
                        2
                    ),

                "altitude_ft":
                    round(
                        mission.altitude_ft,
                        2
                    ),

                "ambient_temperature_c":
                    round(
                        mission.ambient_temperature_c,
                        2
                    ),

                "ambient_pressure_kpa":
                    round(
                        environment.ambient_pressure_kpa,
                        3
                    ),

                "air_density_kg_m3":
                    round(
                        environment.air_density_kg_m3,
                        4
                    ),

                "engine_load_pct":
                    round(
                        mission.engine_load_pct,
                        2
                    ),

                "rpm":
                    round(
                        sensor.rpm,
                        2
                    ),

                "cht_c":
                    round(
                        sensor.cht_c,
                        2
                    ),

                "egt_c":
                    round(
                        sensor.egt_c,
                        2
                    ),

                "oil_pressure_bar":
                    round(
                        sensor.oil_pressure_bar,
                        3
                    ),

                "oil_temperature_c":
                    round(
                        sensor.oil_temperature_c,
                        2
                    ),

                "fuel_flow_lph":
                    round(
                        sensor.fuel_flow_lph,
                        3
                    ),

                "injection_timing_deg":
                    round(
                        sensor.injection_timing_deg,
                        3
                    ),

                "vibration_mm_s":
                    round(
                        sensor.vibration_mm_s,
                        3
                    ),

                "battery_voltage_v":
                    round(
                        sensor.battery_voltage_v,
                        3
                    ),

                "alternator_current_a":
                    round(
                        sensor.alternator_current_a,
                        3
                    ),

                # Simulation ground-truth information.
                # Dashboard should use detected_fault for
                # the AI diagnosis.
                "fault_type":
                    fault.fault_type,

                "fault_severity":
                    round(
                        fault.severity,
                        3
                    ),

                "cooling_health":
                    round(
                        fault.cooling_health,
                        3
                    ),

                "injector_health":
                    round(
                        fault.injector_health,
                        3
                    ),

                "lubrication_health":
                    round(
                        fault.lubrication_health,
                        3
                    ),

                "mechanical_health":
                    round(
                        fault.mechanical_health,
                        3
                    ),

                "misfire_severity":
                    round(
                        fault.misfire_severity,
                        3
                    )
            }

            # ---------------------------------------------
            # DIGITAL TWIN
            # ---------------------------------------------

            expected = twin.estimate(

                throttle_pct=mission.throttle_pct,

                engine_load_pct=mission.engine_load_pct,

                altitude_ft=mission.altitude_ft,

                ambient_temperature_c=(
                    mission.ambient_temperature_c
                ),

                air_density_kg_m3=(
                    environment.air_density_kg_m3
                )
            )

            # ---------------------------------------------
            # RESIDUALS
            # ---------------------------------------------

            residuals = residual_analyzer.calculate(

                telemetry,

                expected
            )

            # Store the healthy Digital Twin predictions
            # and actual-minus-expected residuals directly
            # in the live telemetry stream.

            telemetry.update(
                expected
            )

            telemetry.update(
                residuals
            )

            # ---------------------------------------------
            # ML CLASSIFIER
            # ---------------------------------------------

            # The trained classifier uses mission phase as
            # one of its input features.

            residual_sample = dict(
                residuals
            )

            residual_sample["mission_phase"] = (
                mission.phase
            )

            ml_result = classifier.add_sample(
                residual_sample
            )

            ml_fault, ml_confidence = (
                extract_ml_result(
                    ml_result,
                    classifier
                )
            )

            # ---------------------------------------------
            # HEALTH INDEX
            # ---------------------------------------------

            health = health_index.calculate(

                residuals,

                ml_confidence,

                ml_fault
            )

            # ---------------------------------------------
            # RUL
            # ---------------------------------------------

            rul = rul_predictor.update(

                simulation_time,

                health["health_index"],

                health["fault_type"]
            )

            # ---------------------------------------------
            # FINAL LIVE DIAGNOSTIC OUTPUT
            # ---------------------------------------------

            telemetry["detected_fault"] = (
                health["fault_type"]
            )

            telemetry["fault_confidence"] = round(
                ml_confidence,
                3
            )

            telemetry["health_index"] = round(
                health["health_index"],
                2
            )

            telemetry["health_status"] = (
                health["status"]
            )

            telemetry["residual_score"] = round(
                health["residual_score"],
                3
            )

            telemetry["twin_health"] = round(
                health["twin_health"],
                2
            )

            telemetry["ml_health"] = round(
                health["ml_health"],
                2
            )

            telemetry["rul_hours"] = round(
                rul["rul_hours"],
                2
            )

            telemetry[
                "degradation_rate_per_hour"
            ] = round(

                rul[
                    "degradation_rate_per_hour"
                ],

                2
            )

            telemetry["rul_trend"] = (
                rul["trend"]
            )

            # ---------------------------------------------
            # WRITE LIVE CSV
            # ---------------------------------------------

            if live_writer is None:

                live_writer = csv.DictWriter(

                    live_file,

                    fieldnames=telemetry.keys()
                )

                live_writer.writeheader()

            live_writer.writerow(
                telemetry
            )

            live_file.flush()

            # ---------------------------------------------
            # TERMINAL
            # ---------------------------------------------

            print(

                f"[{simulation_time:6.1f}s] "

                f"Phase={mission.phase:<12} "

                f"RPM={sensor.rpm:7.0f} "

                f"CHT={sensor.cht_c:6.1f}C "

                f"EGT={sensor.egt_c:6.1f}C "

                f"OilP={sensor.oil_pressure_bar:5.2f} "

                f"Vib={sensor.vibration_mm_s:5.2f} "

                f"Health={health['health_index']:6.1f} "

                f"Status={health['status']:<9} "

                f"ML={ml_fault:<24} "

                f"Conf={ml_confidence * 100:5.1f}% "

                f"RUL={rul['rul_hours']:6.1f}h"
            )

            # ---------------------------------------------
            # REAL-TIME STEP
            # ---------------------------------------------

            time.sleep(
                DT
            )

            simulation_time += DT

    except KeyboardInterrupt:

        print()
        print(
            "Simulation stopped by user."
        )

    finally:

        try:

            live_file.close()

        except Exception:

            pass

        print()
        print("=" * 75)
        print("LIVE MONITOR STOPPED")
        print(
            f"Telemetry saved to: {LIVE_CSV}"
        )
        print("=" * 75)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()