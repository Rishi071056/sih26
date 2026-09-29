# ============================================================
# XGBOOST FAULT CLASSIFICATION - UAV ENGINE
# Mission-Level Training + Evaluation Pipeline
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. PATHS
# ============================================================

DATA_FILE = "data/ml_features.csv"

MODEL_DIR = "models"

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_fault_classifier.json"
)

METADATA_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_fault_classifier_metadata.pkl"
)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING ML FEATURE DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("\nMissing values:")

missing_values = df.isnull().sum().sum()

print(f"Total missing values: {missing_values}")

if missing_values > 0:
    raise ValueError(
        "Dataset contains missing values."
    )


# ============================================================
# 5. DEFINE FAULT CLASSES
# ============================================================

fault_classes = sorted(
    df["fault_type"].unique()
)

label_to_id = {
    label: index
    for index, label in enumerate(fault_classes)
}

id_to_label = {
    index: label
    for label, index in label_to_id.items()
}


print("\nFault classes:")

for label, index in label_to_id.items():
    print(f"{index}: {label}")


# ============================================================
# 6. ENCODE FAULT LABEL
# ============================================================

df["fault_label"] = (
    df["fault_type"].map(label_to_id)
)


# ============================================================
# 7. ENCODE MISSION PHASE
# ============================================================

# Mission phase is useful information because engine
# behavior changes between takeoff, climb, cruise, etc.

phase_mapping = {
    "TAKEOFF": 0,
    "CLIMB": 1,
    "CRUISE": 2,
    "LOITER": 3,
    "DESCENT": 4,
    "LANDING": 5
}


df["mission_phase_encoded"] = (
    df["mission_phase"].map(
        phase_mapping
    )
)


# Check that every phase was successfully encoded.

if df["mission_phase_encoded"].isnull().any():

    unknown_phases = (
        df.loc[
            df["mission_phase_encoded"].isnull(),
            "mission_phase"
        ]
        .unique()
    )

    raise ValueError(
        f"Unknown mission phases found: {unknown_phases}"
    )


# ============================================================
# 8. MISSION-LEVEL TRAIN / TEST SPLIT
# ============================================================

print("\n")
print("=" * 60)
print("MISSION-LEVEL TRAIN / TEST SPLIT")
print("=" * 60)


# IMPORTANT:
#
# We DO NOT randomly split rows.
#
# Adjacent windows from the same mission are highly correlated.
#
# Instead:
#
# 4 missions/class -> training
# 1 mission/class  -> testing

missions = sorted(
    df["mission_id"].unique()
)

print(
    f"Total missions: {len(missions)}"
)


train_missions = []
test_missions = []


for fault_type in fault_classes:

    fault_missions = sorted(
        df.loc[
            df["fault_type"] == fault_type,
            "mission_id"
        ].unique()
    )

    print(
        f"\n{fault_type}: "
        f"{len(fault_missions)} missions"
    )

    if len(fault_missions) < 2:

        raise ValueError(
            f"Not enough missions for {fault_type}"
        )

    # First 80% missions for training
    # Last 20% mission(s) for testing

    split_index = int(
        len(fault_missions) * 0.8
    )

    # Ensure at least one test mission
    split_index = min(
        split_index,
        len(fault_missions) - 1
    )

    train_missions.extend(
        fault_missions[:split_index]
    )

    test_missions.extend(
        fault_missions[split_index:]
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


print("\nTraining missions:")
print(len(train_missions))

print("\nTesting missions:")
print(len(test_missions))

print("\nTraining samples:")
print(len(train_df))

print("\nTesting samples:")
print(len(test_df))


# ============================================================
# 9. CHECK TRAIN / TEST MISSION SEPARATION
# ============================================================

overlap = set(train_missions).intersection(
    set(test_missions)
)

if overlap:

    raise ValueError(
        f"Mission leakage detected: {overlap}"
    )

print("\nMission leakage check: PASSED")


# ============================================================
# 10. SELECT MODEL FEATURES
# ============================================================

# IMPORTANT:
#
# These columns MUST NOT be inputs:
#
# fault_type
#     -> actual answer
#
# fault_label
#     -> encoded actual answer
#
# mission_id
#     -> used for splitting only
#
# fault_severity
#     -> directly reveals degradation severity
#
# timestamp_s
#     -> absolute mission time
#
# mission_phase
#     -> original string version
#
# We use mission_phase_encoded instead.

EXCLUDED_COLUMNS = [

    "fault_type",

    "fault_label",

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


# ============================================================
# 11. CREATE X AND Y
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    "fault_label"
]


X_test = test_df[
    FEATURE_COLUMNS
]

y_test = test_df[
    "fault_label"
]


print("\n")
print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print(
    f"Number of features: "
    f"{len(FEATURE_COLUMNS)}"
)


print("\nFeatures:")

for i, feature in enumerate(
    FEATURE_COLUMNS,
    start=1
):

    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# 12. CHECK FEATURE DATA TYPES
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
        "Non-numeric features found: "
        f"{invalid_columns}"
    )


print(
    "\nFeature datatype check: PASSED"
)


# ============================================================
# 13. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 60)

print(
    y_train
    .map(id_to_label)
    .value_counts()
)


print("\n")
print("=" * 60)
print("TEST CLASS DISTRIBUTION")
print("=" * 60)

print(
    y_test
    .map(id_to_label)
    .value_counts()
)


# ============================================================
# 14. BUILD XGBOOST CLASSIFIER
# ============================================================

print("\n")
print("=" * 60)
print("BUILDING XGBOOST MODEL")
print("=" * 60)


model = XGBClassifier(

    # Multi-class classification
    objective="multi:softprob",

    # Number of fault classes
    num_class=len(fault_classes),

    # Number of boosting trees
    n_estimators=300,

    # Maximum depth of each tree
    max_depth=6,

    # Learning rate
    learning_rate=0.05,

    # Random row sampling
    subsample=0.8,

    # Random feature sampling
    colsample_bytree=0.8,

    # Minimum child weight
    min_child_weight=3,

    # L2 regularization
    reg_lambda=1.0,

    # Reproducibility
    random_state=42,

    # Evaluation metric
    eval_metric="mlogloss",

    # Fast histogram algorithm
    tree_method="hist",

    # Use all available CPU threads
    n_jobs=-1
)


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# 16. MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)


# ============================================================
# 17. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("XGBOOST FAULT CLASSIFIER RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=fault_classes,
        digits=4
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
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


print(cm_df)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": (
        model.feature_importances_
    )

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
    .to_string(index=False)
)


# ============================================================
# 21. SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


model.save_model(
    MODEL_FILE
)


print("\nModel saved:")
print(
    MODEL_FILE
)


# ============================================================
# 22. SAVE METADATA
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
        FEATURE_COLUMNS
}


joblib.dump(
    metadata,
    METADATA_FILE
)


print("\nMetadata saved:")

print(
    METADATA_FILE
)


# ============================================================
# 23. SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)


sample_count = min(
    10,
    len(X_test)
)


for i in range(
    sample_count
):

    actual_label = id_to_label[
        int(y_test.iloc[i])
    ]


    predicted_label = id_to_label[
        int(y_pred[i])
    ]


    confidence = float(
        y_probability[i].max()
    )


    print(

        f"{i + 1:02d}. "

        f"Actual = "
        f"{actual_label:<25} "

        f"Predicted = "
        f"{predicted_label:<25} "

        f"Confidence = "
        f"{confidence:.2%}"

    )


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("FINAL MODEL SUMMARY")
print("=" * 60)

print(
    "Model        : XGBoost"
)

print(
    "Task         : 7-Class Fault Classification"
)

print(
    f"Train missions: "
    f"{len(train_missions)}"
)

print(
    f"Test missions : "
    f"{len(test_missions)}"
)

print(
    f"Train samples : "
    f"{len(X_train)}"
)

print(
    f"Test samples  : "
    f"{len(X_test)}"
)

print(
    f"Features      : "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Accuracy      : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Model file    : "
    f"{MODEL_FILE}"
)

print("=" * 60)

print(
    "\nXGBoost fault-classification pipeline complete."
)