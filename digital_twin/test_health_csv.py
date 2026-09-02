import csv

from health import HealthIndex


INPUT_FILE = "../data/analysis/cooling_sensor_fusion.csv"


health_system = HealthIndex()

rows = []

with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        # Build the Sensor Fusion result for this row.
        fusion_result = {
            "overall_anomaly_score":
                float(row["overall_anomaly_score"]),

            "thermal_deviation":
                float(row["thermal_deviation"]),

            "lubrication_deviation":
                float(row["lubrication_deviation"]),

            "combustion_deviation":
                float(row["combustion_deviation"]),

            "mechanical_deviation":
                float(row["mechanical_deviation"]),

            "electrical_deviation":
                float(row["electrical_deviation"]),
        }

        # Calculate Health Index.
        health_result = health_system.calculate(
            fusion_result
        )

        rows.append({
        "timestamp_s": float(row["timestamp_s"]),
        "mission_phase": row["mission_phase"],
        "fault_severity": float(row["fault_severity"]),
        "overall_anomaly_score": float(
            row["overall_anomaly_score"]
        ),
        **health_result,
    })


print("\nENGINE HEALTH ANALYSIS")
print("=" * 80)

print(
    f"{'Time':>8} "
    f"{'Phase':>10} "
    f"{'Severity':>10} "
    f"{'Anomaly':>10} "
    f"{'Health':>10} "
    f"{'Status':>12} "
    f"{'Subsystem':>14}"
)

print("-" * 80)


# Selected mission checkpoints.
checkpoint_times = [
    8.0,
    23.0,
    38.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
]


for target_time in checkpoint_times:

    closest_row = min(
        rows,
        key=lambda r:
        abs(r["timestamp_s"] - target_time)
    )

    print(
        f"{closest_row['timestamp_s']:8.1f} "
        f"{closest_row['mission_phase']:>10} "
        f"{closest_row['fault_severity']:10.2f} "
        f"{float(closest_row['overall_anomaly_score']):10.3f} "
        f"{float(closest_row['health_index']):10.1f} "
        f"{closest_row['health_status']:>12} "
        f"{closest_row['dominant_subsystem']:>14}"
    )