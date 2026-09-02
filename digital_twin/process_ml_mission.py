import os
import glob
import pandas as pd


from .twin_engine import DigitalTwinEngine
from .residuals import ResidualAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "data/missions/ml"
OUTPUT_DIR = "data/missions/ml_residuals"


# ============================================================
# PROCESS ONE MISSION
# ============================================================

def process_mission(
    input_file,
    output_file,
    twin,
    analyzer
):

    df = pd.read_csv(
        input_file
    )

    results = []

    for _, row in df.iterrows():

        # ----------------------------------------------------
        # Digital Twin estimates healthy engine behavior
        # ----------------------------------------------------

        expected = twin.estimate(

            throttle_pct=row["throttle_pct"],

            engine_load_pct=row["engine_load_pct"],

            altitude_ft=row["altitude_ft"],

            ambient_temperature_c=(
                row["ambient_temperature_c"]
            ),

            air_density_kg_m3=(
                row["air_density_kg_m3"]
            )
        )

        # ----------------------------------------------------
        # Calculate actual - expected residuals
        # ----------------------------------------------------

        residuals = analyzer.calculate(
            row.to_dict(),
            expected
        )

        # ----------------------------------------------------
        # Combine original data + expected + residuals
        # ----------------------------------------------------

        result = row.to_dict()

        result.update(
            expected
        )

        result.update(
            residuals
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        output_file,
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    input_files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.csv"
            )
        )
    )

    print("=" * 70)
    print("DIGITAL TWIN BATCH PROCESSOR")
    print("=" * 70)

    print(
        f"\nInput missions: {len(input_files)}"
    )

    twin = DigitalTwinEngine()

    analyzer = ResidualAnalyzer()

    processed = 0

    for input_file in input_files:

        filename = os.path.basename(
            input_file
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            filename
        )

        print(
            f"Processing: {filename}"
        )

        process_mission(
            input_file,
            output_file,
            twin,
            analyzer
        )

        processed += 1

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Processed missions: {processed}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()