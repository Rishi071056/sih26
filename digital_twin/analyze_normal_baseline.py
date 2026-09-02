import csv
import statistics

from twin_engine import DigitalTwinEngine
from residuals import ResidualAnalyzer


INPUT_FILE = "../data/missions/normal_001.csv"


def main():

    twin = DigitalTwinEngine()
    analyzer = ResidualAnalyzer()

    phase_residuals = {}

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            phase = row["mission_phase"]

            expected = twin.estimate(
                throttle_pct=float(row["throttle_pct"]),
                engine_load_pct=float(row["engine_load_pct"]),
                altitude_ft=float(row["altitude_ft"]),
                ambient_temperature_c=float(row["ambient_temperature_c"]),
                air_density_kg_m3=float(row["air_density_kg_m3"])
            )

            actual = {
                "rpm": float(row["rpm"]),
                "cht_c": float(row["cht_c"]),
                "egt_c": float(row["egt_c"]),
                "oil_pressure_bar": float(row["oil_pressure_bar"]),
                "oil_temperature_c": float(row["oil_temperature_c"]),
                "fuel_flow_lph": float(row["fuel_flow_lph"]),
                "injection_timing_deg": float(row["injection_timing_deg"]),
                "vibration_mm_s": float(row["vibration_mm_s"]),
                "battery_voltage_v": float(row["battery_voltage_v"]),
                "alternator_current_a": float(row["alternator_current_a"]),
            }

            residuals = analyzer.calculate(actual, expected)

            if phase not in phase_residuals:
                phase_residuals[phase] = {}

            for name, value in residuals.items():

                if name not in phase_residuals[phase]:
                    phase_residuals[phase][name] = []

                phase_residuals[phase][name].append(value)

    # ---------------------------------------------------------
    # Print baseline statistics
    # ---------------------------------------------------------

    print("\nHEALTHY DIGITAL TWIN BASELINE")
    print("=" * 80)

    for phase, residual_data in phase_residuals.items():

        print(f"\nMISSION PHASE: {phase}")
        print("-" * 80)

        for name, values in residual_data.items():

            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0

            print(
                f"{name:35s} "
                f"mean={mean:8.3f}  "
                f"std={std:8.3f}"
            )


if __name__ == "__main__":
    main()