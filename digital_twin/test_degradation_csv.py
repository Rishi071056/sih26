import csv

from degradation import DegradationAnalyzer


INPUT_FILE = "../data/missions/cooling_twin_analysis.csv"


analyzer = DegradationAnalyzer(
    window_size=100
)


timestamps = []

residual_history = {
    "cht_c_residual": [],
    "oil_temperature_c_residual": [],
    "egt_c_residual": [],
    "vibration_mm_s_residual": [],
    "rpm_residual": [],
    "oil_pressure_bar_residual": [],
    "fuel_flow_lph_residual": [],
    "injection_timing_deg_residual": [],
    "battery_voltage_v_residual": [],
    "alternator_current_a_residual": [],
}


with open(INPUT_FILE, "r", newline="", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    for row in reader:

        timestamps.append(
            float(row["timestamp_s"])
        )

        for parameter in residual_history:

            residual_history[parameter].append(
                float(row[parameter])
            )


checkpoint_times = [
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
]


print("\nSUBSYSTEM DEGRADATION ANALYSIS")
print("=" * 100)

print(
    f"{'Time':>8} "
    f"{'Thermal':>12} "
    f"{'Lubrication':>14} "
    f"{'Combustion':>14} "
    f"{'Mechanical':>14} "
    f"{'Electrical':>14}"
)

print("-" * 100)


for target_time in checkpoint_times:

    index = min(
        range(len(timestamps)),
        key=lambda i:
        abs(timestamps[i] - target_time)
    )

    end_index = index + 1

    start_index = max(
        0,
        end_index - analyzer.window_size
    )

    current_timestamps = timestamps[
        start_index:end_index
    ]

    current_history = {}

    for parameter, values in residual_history.items():

        current_history[parameter] = values[
            start_index:end_index
        ]

    result = analyzer.analyze(
        current_timestamps,
        current_history
    )

    print(
        f"{timestamps[index]:8.1f} "
        f"{result['thermal_degradation_trend']:12.5f} "
        f"{result['lubrication_degradation_trend']:14.5f} "
        f"{result['combustion_degradation_trend']:14.5f} "
        f"{result['mechanical_degradation_trend']:14.5f} "
        f"{result['electrical_degradation_trend']:14.5f}"
    )