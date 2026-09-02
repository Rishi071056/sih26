import os
import pandas as pd

DATA_DIR = "../data/missions"

FILES = [
    "normal_001.csv",
    "cooling_degradation_001.csv",
    "injector_degradation_001.csv",
    "lubrication_degradation_001.csv",
    "misfire_001.csv",
    "mechanical_failure_001.csv",
    "sensor_drift_001.csv",
]

print("=" * 60)
print("ML DATASET INSPECTION")
print("=" * 60)

total_rows = 0

for filename in FILES:
    path = os.path.join(DATA_DIR, filename)

    print(f"\n--- {filename} ---")

    if not os.path.exists(path):
        print("ERROR: File not found")
        continue

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", df.isna().sum().sum())

    if "fault_type" in df.columns:
        print("Fault type:", df["fault_type"].unique())

    if "fault_severity" in df.columns:
        print(
            "Severity range:",
            round(df["fault_severity"].min(), 3),
            "to",
            round(df["fault_severity"].max(), 3)
        )

    total_rows += len(df)

print("\n" + "=" * 60)
print("TOTAL ROWS:", total_rows)
print("=" * 60)