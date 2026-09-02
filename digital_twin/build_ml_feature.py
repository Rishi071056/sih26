import os
import glob
import numpy as np
import pandas as pd


INPUT_DIR = "data/missions/ml_residuals"
OUTPUT_FILE = "data/ml_features.csv"

# Telemetry = 10 Hz
# 5 seconds = 50 samples
WINDOW_SIZE = 50

# Update every 1 second = 10 samples
STEP_SIZE = 10


RESIDUAL_COLUMNS = [
    "rpm_residual",
    "cht_c_residual",
    "egt_c_residual",
    "oil_pressure_bar_residual",
    "oil_temperature_c_residual",
    "fuel_flow_lph_residual",
    "injection_timing_deg_residual",
    "vibration_mm_s_residual",
    "battery_voltage_v_residual",
    "alternator_current_a_residual",
]


def calculate_slope(values):
    """Calculate linear trend over the window."""

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    return float(np.polyfit(x, values, 1)[0])


def extract_window_features(window):
    """Extract statistical and trend features from one window."""

    features = {}

    for column in RESIDUAL_COLUMNS:

        values = window[column].astype(float).values

        features[f"{column}_mean"] = float(np.mean(values))
        features[f"{column}_std"] = float(np.std(values))
        features[f"{column}_max"] = float(np.max(values))
        features[f"{column}_min"] = float(np.min(values))
        features[f"{column}_trend"] = calculate_slope(values)

    return features


def process_file(filepath):

    df = pd.read_csv(filepath)

    rows = []

    for start in range(
        0,
        len(df) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):

        window = df.iloc[
            start:start + WINDOW_SIZE
        ]

        features = extract_window_features(window)

        # -----------------------------
        # Metadata
        # -----------------------------

        features["timestamp_s"] = float(
            window["timestamp_s"].iloc[-1]
        )

        features["mission_phase"] = (
            window["mission_phase"].iloc[-1]
        )

        # Ground-truth label.
        # This will NOT be used as an ML input.
        features["fault_type"] = (
            window["fault_type"].iloc[-1]
        )

        features["fault_severity"] = float(
            window["fault_severity"].iloc[-1]
        )

        # Used later for mission-level train/test split.
        features["mission_id"] = os.path.basename(
            filepath
        )

        rows.append(features)

    return rows


def main():

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    files = sorted(
        glob.glob(
            os.path.join(
                INPUT_DIR,
                "*.csv"
            )
        )
    )

    print(f"Residual files found: {len(files)}")
    print(f"Window size: {WINDOW_SIZE} samples")
    print("Window duration: 5 seconds")
    print(f"Update interval: {STEP_SIZE / 10:.1f} seconds")
    print()

    all_rows = []

    for i, filepath in enumerate(files, start=1):

        rows = process_file(filepath)

        all_rows.extend(rows)

        print(
            f"[{i}/{len(files)}] "
            f"{os.path.basename(filepath)} -> "
            f"{len(rows)} windows"
        )

    feature_df = pd.DataFrame(all_rows)

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 50)
    print("Feature extraction complete.")
    print("=" * 50)

    print(f"Total samples: {len(feature_df)}")
    print(f"Total columns: {len(feature_df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print()
    print("Fault distribution:")
    print(
        feature_df["fault_type"]
        .value_counts()
    )


if __name__ == "__main__":
    main()