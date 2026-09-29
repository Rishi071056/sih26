# ============================================================
# XGBOOST FAULT CLASSIFICATION V2
# Time-Aware Labels + Class Weighting
# Mission-Level Train/Test Split
# ============================================================

import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# 1. PATHS
# ============================================================

DATA_FILE = "data/ml_features_v2.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_fault_classifier_v2.json"
)

METADATA_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_fault_classifier_v2_metadata.pkl"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("XGBOOST FAULT CLASSIFIER V2")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

missing = df.isnull().sum().sum()

print(
    f"\nMissing values: {missing}"
)

if missing > 0:

    raise ValueError(
        "Dataset contains missing values."
    )


# ============================================================
# 4. DEFINE FAULT CLASSES
# ============================================================

fault_classes = [
    "COOLING_DEGRADATION",
    "INJECTOR_DEGRADATION",
    "LUBRICATION_DEGRADATION",
    "MECHANICAL_DEGRADATION",
    "MISFIRE",
    "NORMAL",
    "SENSOR_DRIFT"
]


# Check that all expected classes exist.

available_classes = set(
    df["fault_type"].unique()
)

missing_classes = (
    set(fault_classes)
    - available_classes
)

if missing_classes:

    raise ValueError(
        f"Missing fault classes: "
        f"{missing_classes}"
    )


label_to_id = {
    label: index
    for index, label in enumerate(
        fault_classes
    )
}


id_to_label = {
    index: label
    for label, index in label_to_id.items()
}


print("\nFault classes:")

for label, index in label_to_id.items():

    print(
        f"{index}: {label}"
    )


# ============================================================
# 5. ENCODE WINDOW FAULT LABEL
# ============================================================

df["fault_label"] = (
    df["fault_type"]
    .map(label_to_id)
)


# ============================================================
# 6. ENCODE MISSION PHASE
# ============================================================

phase_mapping = {

    "TAKEOFF": 0,

    "CLIMB": 1,

    "CRUISE": 2,

    "LOITER": 3,

    "DESCENT": 4,

    "LANDING": 5
}


df["mission_phase_encoded"] = (
    df["mission_phase"]
    .map(phase_mapping)
)


if df[
    "mission_phase_encoded"
].isnull().any():

    unknown_phases = (
        df.loc[
            df[
                "mission_phase_encoded"
            ].isnull(),
            "mission_phase"
        ]
        .unique()
    )

    raise ValueError(
        f"Unknown mission phases: "
        f"{unknown_phases}"
    )


# ============================================================
# 7. DETERMINE ORIGINAL MISSION FAULT
# ============================================================

# IMPORTANT:
#
# fault_type is now a TIME-AWARE WINDOW LABEL.
#
# Example:
#
# cooling_degradation_006.csv
#
# Early windows:
#     NORMAL
#
# Later windows:
#     COOLING_DEGRADATION
#
# Therefore fault_type cannot be used to decide
# which mission belongs to train/test.
#
# We determine the original mission class from
# the mission filename.


def get_original_fault(mission_id):

    filename = str(
        mission_id
    ).lower()


    if filename.startswith(
        "cooling_degradation_"
    ):

        return "COOLING_DEGRADATION"


    if filename.startswith(
        "injector_degradation_"
    ):

        return "INJECTOR_DEGRADATION"


    if filename.startswith(
        "lubrication_degradation_"
    ):

        return "LUBRICATION_DEGRADATION"


    if filename.startswith(
        "mechanical_degradation_"
    ):

        return "MECHANICAL_DEGRADATION"


    if filename.startswith(
        "misfire_"
    ):

        return "MISFIRE"


    if filename.startswith(
        "sensor_drift_"
    ):

        return "SENSOR_DRIFT"


    if filename.startswith(
        "normal_"
    ):

        return "NORMAL"


    raise ValueError(
        f"Unknown mission filename: "
        f"{mission_id}"
    )


df["original_fault_type"] = (
    df["mission_id"]
    .apply(get_original_fault)
)


# ============================================================
# 8. MISSION-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("=" * 60)
print("MISSION-LEVEL TRAIN / TEST SPLIT")
print("=" * 60)


missions = sorted(
    df["mission_id"].unique()
)


print(
    f"Total missions: "
    f"{len(missions)}"
)


train_missions = []

test_missions = []


for fault_type in fault_classes:

    fault_missions = sorted(
        df.loc[
            df["original_fault_type"]
            == fault_type,
            "mission_id"
        ].unique()
    )


    print(
        f"\n{fault_type}: "
        f"{len(fault_missions)} missions"
    )


    if len(fault_missions) < 2:

        raise ValueError(
            f"Not enough missions for "
            f"{fault_type}"
        )


    # With 5 missions:
    #
    # 4 -> training
    # 1 -> testing

    split_index = int(
        len(fault_missions) * 0.8
    )


    split_index = max(
        1,
        min(
            split_index,
            len(fault_missions) - 1
        )
    )


    train_missions.extend(
        fault_missions[
            :split_index
        ]
    )


    test_missions.extend(
        fault_missions[
            split_index:
        ]
    )


train_df = df[
    df["mission_id"].isin(
        train_missions
    )
].copy()


test_df = df[
    df["mission_id"].isin(
        test_missions
    )
].copy()


print(
    f"\nTraining missions: "
    f"{len(train_missions)}"
)


print(
    f"Testing missions: "
    f"{len(test_missions)}"
)


print(
    f"Training samples: "
    f"{len(train_df)}"
)


print(
    f"Testing samples: "
    f"{len(test_df)}"
)


# ============================================================
# 9. LEAKAGE CHECK
# ============================================================

overlap = set(
    train_missions
).intersection(
    set(test_missions)
)


if overlap:

    raise ValueError(
        f"Mission leakage detected: "
        f"{overlap}"
    )


print(
    "\nMission leakage check: PASSED"
)


# ============================================================
# 10. CHECK ORIGINAL MISSION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING MISSION DISTRIBUTION")
print("=" * 60)


print(
    train_df[
        [
            "mission_id",
            "original_fault_type"
        ]
    ]
    .drop_duplicates()
    ["original_fault_type"]
    .value_counts()
    .sort_index()
)


print("\n")
print("=" * 60)
print("TEST MISSION DISTRIBUTION")
print("=" * 60)


print(
    test_df[
        [
            "mission_id",
            "original_fault_type"
        ]
    ]
    .drop_duplicates()
    ["original_fault_type"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. SELECT ML FEATURES
# ============================================================

# DO NOT give these to the model:
#
# fault_type
#     -> target label
#
# fault_label
#     -> encoded target
#
# original_fault_type
#     -> original mission fault; would cause leakage
#
# mission_id
#     -> only for train/test splitting
#
# fault_severity
#     -> directly reveals fault progression
#
# timestamp_s
#     -> absolute mission time
#
# mission_phase
#     -> string version
#
# mission_phase_encoded IS allowed.

EXCLUDED_COLUMNS = [

    "fault_type",

    "fault_label",

    "original_fault_type",

    "mission_id",

    "fault_severity",

    "timestamp_s",

    "mission_phase"

]


FEATURE_COLUMNS = [

    column

    for column in df.columns

    if column not in EXCLUDED_COLUMNS

    and pd.api.types.is_numeric_dtype(
        df[column]
    )

]


print("\n")
print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)


print(
    f"Number of features: "
    f"{len(FEATURE_COLUMNS)}"
)


for i, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 12. CREATE TRAIN / TEST DATA
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
].copy()


y_train = train_df[
    "fault_label"
].copy()


X_test = test_df[
    FEATURE_COLUMNS
].copy()


y_test = test_df[
    "fault_label"
].copy()


# ============================================================
# 13. FEATURE TYPE CHECK
# ============================================================

invalid_columns = [

    column

    for column in FEATURE_COLUMNS

    if not pd.api.types.is_numeric_dtype(
        X_train[column]
    )

]


if invalid_columns:

    raise ValueError(
        f"Non-numeric features found: "
        f"{invalid_columns}"
    )


print(
    "\nFeature datatype check: PASSED"
)


# ============================================================
# 14. WINDOW LABEL DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING WINDOW LABEL DISTRIBUTION")
print("=" * 60)


print(
    train_df[
        "fault_type"
    ]
    .value_counts()
    .sort_index()
)


print("\n")
print("=" * 60)
print("TEST WINDOW LABEL DISTRIBUTION")
print("=" * 60)


print(
    test_df[
        "fault_type"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 15. CALCULATE CLASS WEIGHTS
# ============================================================

# V2 contains more NORMAL windows because healthy
# pre-fault periods are correctly labelled NORMAL.
#
# We use inverse-frequency sample weighting so that
# NORMAL does not dominate the training process.


class_counts = (
    y_train
    .value_counts()
    .sort_index()
)


total_samples = len(
    y_train
)


num_classes = len(
    fault_classes
)


class_weights = {

    class_id:
        total_samples /
        (
            num_classes *
            count
        )

    for class_id, count
    in class_counts.items()

}


print("\n")
print("=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)


for class_id in sorted(
    class_weights
):

    print(
        f"{id_to_label[class_id]:<30}"
        f"{class_weights[class_id]:.4f}"
    )


sample_weights = (
    y_train.map(
        class_weights
    )
)


# ============================================================
# 16. BUILD XGBOOST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING XGBOOST V2")
print("=" * 60)


model = XGBClassifier(

    objective="multi:softprob",

    num_class=num_classes,

    n_estimators=350,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    reg_lambda=1.0,

    reg_alpha=0.0,

    random_state=42,

    eval_metric="mlogloss",

    tree_method="hist",

    n_jobs=-1

)


# ============================================================
# 17. TRAIN MODEL
# ============================================================

print(
    "\nTraining XGBoost V2..."
)


model.fit(

    X_train,

    y_train,

    sample_weight=sample_weights

)


print(
    "Training complete."
)


# ============================================================
# 18. MAKE PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)


y_pred = model.predict(
    X_test
)


y_probability = model.predict_proba(
    X_test
)


# ============================================================
# 19. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("XGBOOST V2 RESULTS")
print("=" * 60)


print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Macro F1: "
    f"{macro_f1:.4f}"
)


print(
    f"Weighted F1: "
    f"{weighted_f1:.4f}"
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_test,

        y_pred,

        labels=list(
            range(
                len(fault_classes)
            )
        ),

        target_names=fault_classes,

        digits=4,

        zero_division=0

    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=list(
        range(
            len(fault_classes)
        )
    )

)


cm_df = pd.DataFrame(

    cm,

    index=[
        f"Actual: {label}"
        for label in fault_classes
    ],

    columns=[
        f"Pred: {label}"
        for label in fault_classes
    ]

)


print(
    cm_df
)


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "feature":
        FEATURE_COLUMNS,

    "importance":
        model.feature_importances_

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\n")
print("=" * 60)
print("TOP 20 IMPORTANT FEATURES")
print("=" * 60)


print(
    importance_df
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 24. SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model.save_model(
    MODEL_FILE
)


print(
    "\nModel saved:"
)


print(
    MODEL_FILE
)


# ============================================================
# 25. SAVE METADATA
# ============================================================

metadata = {

    "fault_classes":
        fault_classes,

    "label_to_id":
        label_to_id,

    "id_to_label":
        id_to_label,

    "phase_mapping":
        phase_mapping,

    "feature_columns":
        FEATURE_COLUMNS,

    "class_weights":
        class_weights

}


joblib.dump(
    metadata,
    METADATA_FILE
)


print(
    "\nMetadata saved:"
)


print(
    METADATA_FILE
)


# ============================================================
# 26. SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


sample_count = min(
    20,
    len(X_test)
)


for i in range(
    sample_count
):

    actual = id_to_label[
        int(y_test.iloc[i])
    ]


    predicted = id_to_label[
        int(y_pred[i])
    ]


    confidence = float(
        y_probability[i].max()
    )


    print(

        f"{i + 1:02d}. "

        f"Actual = "
        f"{actual:<28} "

        f"Predicted = "
        f"{predicted:<28} "

        f"Confidence = "
        f"{confidence:.2%}"

    )


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)


print(
    "Model          : XGBoost V2"
)


print(
    "Task           : 7-Class Fault Classification"
)


print(
    "Labeling       : Time-Aware"
)


print(
    "Train missions : "
    f"{len(train_missions)}"
)


print(
    "Test missions  : "
    f"{len(test_missions)}"
)


print(
    "Train samples  : "
    f"{len(X_train)}"
)


print(
    "Test samples   : "
    f"{len(X_test)}"
)


print(
    "Features       : "
    f"{len(FEATURE_COLUMNS)}"
)


print(
    f"Accuracy       : "
    f"{accuracy * 100:.2f}%"
)


print(
    f"Macro F1       : "
    f"{macro_f1:.4f}"
)


print(
    f"Weighted F1    : "
    f"{weighted_f1:.4f}"
)


print(
    "Model file     : "
    f"{MODEL_FILE}"
)


print(
    "Metadata file  : "
    f"{METADATA_FILE}"
)


print("=" * 60)


print(
    "\nXGBoost V2 training pipeline complete."
)