import csv
import os

from twin_engine import DigitalTwinEngine
from residuals import ResidualAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/missions/cooling_degradation_001.csv"
)

OUTPUT_FILE = (
    "data/missions/"
    "cooling_twin_analysis.csv"
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    twin = DigitalTwinEngine()
    analyzer = ResidualAnalyzer()

    print()
    print("=" * 80)
    print(" DIGITAL TWIN — MISSION ANALYSIS")
    print("=" * 80)
    print()
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if not os.path.exists(INPUT_FILE):

        print("ERROR: Input CSV not found.")
        print(f"Check: {INPUT_FILE}")
        return

    # --------------------------------------------------------
    # Open input CSV
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as input_file:

        reader = csv.DictReader(input_file)

        # ----------------------------------------------------
        # Output file
        # ----------------------------------------------------

        with open(
            OUTPUT_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as output_file:

            writer = None

            row_count = 0

            # =================================================
            # PROCESS EVERY TELEMETRY ROW
            # =================================================

            for row in reader:

                # ------------------------------------------------
                # Operating conditions
                # ------------------------------------------------

                throttle = float(
                    row["throttle_pct"]
                )

                load = float(
                    row["engine_load_pct"]
                )

                altitude = float(
                    row["altitude_ft"]
                )

                ambient_temperature = float(
                    row["ambient_temperature_c"]
                )

                air_density = float(
                    row["air_density_kg_m3"]
                )

                # ------------------------------------------------
                # Digital Twin prediction
                # ------------------------------------------------

                expected = twin.estimate(

                    throttle_pct=throttle,

                    engine_load_pct=load,

                    altitude_ft=altitude,

                    ambient_temperature_c=(
                        ambient_temperature
                    ),

                    air_density_kg_m3=(
                        air_density
                    )
                )

                # ------------------------------------------------
                # Actual sensor values
                # ------------------------------------------------

                actual = {

                    "rpm": float(
                        row["rpm"]
                    ),

                    "cht_c": float(
                        row["cht_c"]
                    ),

                    "egt_c": float(
                        row["egt_c"]
                    ),

                    "oil_pressure_bar": float(
                        row["oil_pressure_bar"]
                    ),

                    "oil_temperature_c": float(
                        row["oil_temperature_c"]
                    ),

                    "fuel_flow_lph": float(
                        row["fuel_flow_lph"]
                    ),

                    "injection_timing_deg": float(
                        row["injection_timing_deg"]
                    ),

                    "vibration_mm_s": float(
                        row["vibration_mm_s"]
                    ),

                    "battery_voltage_v": float(
                        row["battery_voltage_v"]
                    ),

                    "alternator_current_a": float(
                        row["alternator_current_a"]
                    )
                }

                # ------------------------------------------------
                # Calculate residuals
                # ------------------------------------------------

                residuals = analyzer.calculate(
                    actual,
                    expected
                )

                # ------------------------------------------------
                # Combine everything
                # ------------------------------------------------

                output_row = {

                    "timestamp_s": row[
                        "timestamp_s"
                    ],

                    "mission_phase": row[
                        "mission_phase"
                    ],

                    "fault_type": row[
                        "fault_type"
                    ],

                    "fault_severity": row[
                        "fault_severity"
                    ],

                    # Actual
                    **actual,

                    # Expected
                    **expected,

                    # Residuals
                    **residuals
                }

                # ------------------------------------------------
                # Create CSV header
                # ------------------------------------------------

                if writer is None:

                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=output_row.keys()
                    )

                    writer.writeheader()

                writer.writerow(
                    output_row
                )

                row_count += 1

    print()
    print("=" * 80)
    print(" ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(
        f"Rows analyzed : {row_count}"
    )
    print(
        f"Saved to      : {OUTPUT_FILE}"
    )
    print()


if __name__ == "__main__":
    main()