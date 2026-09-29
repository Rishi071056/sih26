# ============================================================
# BUILD ML FEATURES V2
# Time-Aware Fault Labeling
# ============================================================

import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

INPUT_DIR = "data/missions/ml_residuals"

OUTPUT_FILE = "data/ml_features_v2.csv"


# ============================================================
# 2. WINDOW SETTINGS
# ============================================================

# Telemetry frequency = 10 Hz
#
# 50 samples = 5 seconds
WINDOW_SIZE = 50

# Update every 1 second
#
# 10 samples = 1 second
STEP_SIZE = 10


# ============================================================
# 3. FAULT ACTIVATION THRESHOLD
# ============================================================

# Before this severity, treat the engine as NORMAL.
#
# This prevents healthy pre-fault windows from being
# incorrectly labelled as the fault.

FAULT_THRESHOLD = 0.05


# ============================================================
# 4. RESIDUAL COLUMNS
# ============================================================

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

    "alternator_current_a_residual"

]


# ============================================================
# 5. CALCULATE TREND
# ============================================================

def calculate_slope(values):

    if len(values) < 2:
        return 0.0

    x = np.arange(len(values))

    slope = np.polyfit(
        x,
        values,
        1
    )[0]

    return float(slope)


# ============================================================
# 6. EXTRACT WINDOW FEATURES
# ============================================================

def extract_window_features(window):

    features = {}

    for column in RESIDUAL_COLUMNS:

        values = (
            window[column]
            .astype(float)
            .values
        )

        features[
            f"{column}_mean"
        ] = float(
            np.mean(values)
        )

        features[
            f"{column}_std"
        ] = float(
            np.std(values)
        )

        features[
            f"{column}_max"
        ] = float(
            np.max(values)
        )

        features[
            f"{column}_min"
        ] = float(
            np.min(values)
        )

        features[
            f"{column}_trend"
        ] = calculate_slope(
            values
        )

    return features


# ============================================================
# 7. DETERMINE TIME-AWARE LABEL
# ============================================================

def determine_label(window):

    original_fault = (
        window["fault_type"]
        .iloc[-1]
    )

    severity = float(
        window["fault_severity"]
        .iloc[-1]
    )

    # Normal mission
    if original_fault == "NORMAL":
        return "NORMAL"

    # Fault mission but fault has not
    # meaningfully developed yet.
    if severity < FAULT_THRESHOLD:
        return "NORMAL"

    # Fault has developed.
    return original_fault


# ============================================================
# 8. PROCESS ONE MISSION
# ============================================================

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

        features = extract_window_features(
            window
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        features["timestamp_s"] = float(
            window["timestamp_s"].iloc[-1]
        )

        # ----------------------------------------------------
        # Mission phase
        # ----------------------------------------------------

        features["mission_phase"] = (
            window["mission_phase"].iloc[-1]
        )

        # ----------------------------------------------------
        # Time-aware label
        # ----------------------------------------------------

        features["fault_type"] = (
            determine_label(window)
        )

        # ----------------------------------------------------
        # Keep severity only for analysis.
        #
        # It will NOT be used as an ML feature.
        # ----------------------------------------------------

        features["fault_severity"] = float(
            window["fault_severity"].iloc[-1]
        )

        # ----------------------------------------------------
        # Mission ID
        #
        # Required later for mission-level splitting.
        # ----------------------------------------------------

        features["mission_id"] = (
            os.path.basename(filepath)
        )

        rows.append(features)

    return rows


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING ML FEATURES V2")
    print("TIME-AWARE FAULT LABELING")
    print("=" * 60)

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

    print(
        f"\nResidual files found: "
        f"{len(files)}"
    )

    print(
        f"Window size: "
        f"{WINDOW_SIZE} samples"
    )

    print(
        "Window duration: 5 seconds"
    )

    print(
        f"Update interval: "
        f"{STEP_SIZE / 10:.1f} seconds"
    )

    print(
        f"Fault threshold: "
        f"{FAULT_THRESHOLD}"
    )

    print()

    all_rows = []

    for i, filepath in enumerate(
        files,
        start=1
    ):

        rows = process_file(
            filepath
        )

        all_rows.extend(
            rows
        )

        print(
            f"[{i}/{len(files)}] "
            f"{os.path.basename(filepath)} "
            f"-> {len(rows)} windows"
        )

    # ========================================================
    # 10. CREATE DATAFRAME
    # ========================================================

    feature_df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # 11. SAVE
    # ========================================================

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # 12. SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"Total samples: "
        f"{len(feature_df)}"
    )

    print(
        f"Total columns: "
        f"{len(feature_df.columns)}"
    )

    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 60)
    print("TIME-AWARE LABEL DISTRIBUTION")
    print("=" * 60)

    print(
        feature_df[
            "fault_type"
        ].value_counts()
    )

    print()
    print("=" * 60)
    print("SAMPLE LABELS")
    print("=" * 60)

    print(
        feature_df[
            [
                "mission_id",
                "timestamp_s",
                "fault_type",
                "fault_severity"
            ]
        ].head(20)
    )


# ============================================================
# 13. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()