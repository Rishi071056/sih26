import os
import glob
import pandas as pd


DATA_DIR = "../data/missions/ml"


print("=" * 70)
print("ML MISSION DATASET INSPECTION")
print("=" * 70)

files = sorted(
    glob.glob(
        os.path.join(DATA_DIR, "*.csv")
    )
)

print(f"\nTotal CSV files found: {len(files)}")


if len(files) == 0:
    print("ERROR: No CSV files found.")
    raise SystemExit


total_rows = 0
fault_counts = {}
all_phases = set()


for path in files:

    filename = os.path.basename(path)

    df = pd.read_csv(path)

    total_rows += len(df)

    # --------------------------------------------------------
    # Fault type
    # --------------------------------------------------------

    if "fault_type" in df.columns:

        fault_type = df["fault_type"].iloc[0]

        fault_counts[fault_type] = (
            fault_counts.get(fault_type, 0) + 1
        )

    # --------------------------------------------------------
    # Mission phases
    # --------------------------------------------------------

    if "mission_phase" in df.columns:

        all_phases.update(
            df["mission_phase"].dropna().unique()
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isna().sum().sum()

    # --------------------------------------------------------
    # Variation information
    # --------------------------------------------------------

    altitude_min = df["altitude_ft"].min()
    altitude_max = df["altitude_ft"].max()

    temperature_min = (
        df["ambient_temperature_c"].min()
    )

    temperature_max = (
        df["ambient_temperature_c"].max()
    )

    throttle_min = df["throttle_pct"].min()
    throttle_max = df["throttle_pct"].max()

    severity_max = df["fault_severity"].max()

    print(
        f"{filename:<40} "
        f"rows={len(df):>5}  "
        f"missing={missing:>2}"
    )

    print(
        f"    altitude: {altitude_min:7.1f} → "
        f"{altitude_max:7.1f} ft"
    )

    print(
        f"    temperature: {temperature_min:6.1f} → "
        f"{temperature_max:6.1f} °C"
    )

    print(
        f"    throttle: {throttle_min:5.1f} → "
        f"{throttle_max:5.1f} %"
    )

    print(
        f"    max fault severity: {severity_max:.2f}"
    )


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal missions : {len(files)}")
print(f"Total rows     : {total_rows:,}")

print("\nMissions per fault type:")

for fault_type, count in sorted(
    fault_counts.items()
):
    print(
        f"  {fault_type:<25} {count}"
    )

print("\nMission phases found:")

for phase in sorted(all_phases):
    print(f"  {phase}")

print("\n" + "=" * 70)