import csv
import os

from sensor_fusion import SensorFusion


INPUT_FILE = "../data/missions/cooling_twin_analysis.csv"
OUTPUT_FILE = "../data/analysis/cooling_sensor_fusion.csv"


fusion = SensorFusion()


def main():

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(INPUT_FILE, "r", newline="", encoding="utf-8") as infile:

        reader = csv.DictReader(infile)

        rows = []

        for row in reader:

            # Convert residual columns from text to float
            residuals = {}

            for key, value in row.items():

                if key.endswith("_residual") and value != "":
                    residuals[key] = float(value)

            # Run Sensor Fusion
            fusion_result = fusion.calculate(residuals,row["mission_phase"])

            # Add fusion results to the original row
            for key, value in fusion_result.items():
                row[key] = value

            rows.append(row)

    # Write output CSV
    fieldnames = rows[0].keys()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nSensor Fusion analysis complete.")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    # ---------------------------------------------------------
    # Print selected checkpoints
    # ---------------------------------------------------------

    print("\nCHECKPOINTS")
    print("-" * 70)

    checkpoint_times = [8.0, 23.0, 38.0, 60.0, 90.0, 120.0, 150.0, 180.0]

    for target_time in checkpoint_times:

        closest_row = min(
            rows,
            key=lambda r: abs(float(r["timestamp_s"]) - target_time)
        )

        print(
            f"t={float(closest_row['timestamp_s']):6.1f}s | "
            f"severity={float(closest_row['fault_severity']):.2f} | "
            f"thermal={float(closest_row['thermal_deviation']):.3f} | "
            f"overall={float(closest_row['overall_anomaly_score']):.3f}"
        )


if __name__ == "__main__":
    main()