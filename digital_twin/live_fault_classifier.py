from collections import deque

import joblib
import numpy as np
import xgboost as xgb


class LiveFaultClassifier:
    """
    Real-time XGBoost fault classifier.

    Uses the same feature columns produced during training
    and applies temporal persistence to prevent rapid switching
    between NORMAL and a detected fault.
    """

    def __init__(
        self,
        model_path="models/xgboost_fault_classifier_v2.json",
        metadata_path="models/xgboost_fault_classifier_v2_metadata.pkl",
    ):
        # ==================================================
        # LOAD MODEL
        # ==================================================
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        # ==================================================
        # LOAD METADATA
        # ==================================================
        self.metadata = joblib.load(metadata_path)

        # Your trained metadata uses these keys:
        # feature_columns
        # fault_classes
        # label_to_id
        # id_to_label
        # phase_mapping
        # class_weights

        self.feature_names = list(
            self.metadata["feature_columns"]
        )

        self.class_names = list(
            self.metadata["fault_classes"]
        )

        self.phase_mapping = self.metadata.get(
            "phase_mapping",
            {
                "GROUND": 0,
                "TAKEOFF": 1,
                "CLIMB": 2,
                "CRUISE": 3,
                "DESCENT": 4,
                "LANDING": 5,
            },
        )

        # ==================================================
        # RESIDUAL FEATURES
        # ==================================================
        self.residual_names = [
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

        # ==================================================
        # LIVE BUFFER
        # ==================================================
        # 10 Hz × 50 samples = 5 second window.
        self.window_size = 50

        # Predict every 10 samples = 1 second.
        self.update_interval = 10

        self.buffer = deque(
            maxlen=self.window_size
        )

        self.sample_count = 0

        # ==================================================
        # FAULT PERSISTENCE
        # ==================================================
        self.current_fault = "NORMAL"
        self.current_confidence = 1.0

        self.pending_fault = None
        self.pending_count = 0

        # Fault must appear twice before confirmation.
        self.fault_confirmations = 2

        # NORMAL must appear four times before clearing
        # an active fault.
        self.normal_confirmations = 4

    # ======================================================
    # BUILD LIVE FEATURES
    # ======================================================

    def _build_features(self, window):

        features = {}

        for residual_name in self.residual_names:

            values = np.array(
                [
                    float(
                        row.get(
                            residual_name,
                            0.0,
                        )
                    )
                    for row in window
                ],
                dtype=float,
            )

            if len(values) == 0:
                values = np.array([0.0])

            # ----------------------------------------------
            # MEAN
            # ----------------------------------------------
            features[
                residual_name + "_mean"
            ] = float(np.mean(values))

            # ----------------------------------------------
            # STANDARD DEVIATION
            # ----------------------------------------------
            features[
                residual_name + "_std"
            ] = float(np.std(values))

            # ----------------------------------------------
            # MAXIMUM
            # ----------------------------------------------
            features[
                residual_name + "_max"
            ] = float(np.max(values))

            # ----------------------------------------------
            # MINIMUM
            # ----------------------------------------------
            features[
                residual_name + "_min"
            ] = float(np.min(values))

            # ----------------------------------------------
            # TREND
            # ----------------------------------------------
            if len(values) >= 2:

                x = np.arange(len(values))

                try:
                    trend = np.polyfit(
                        x,
                        values,
                        1,
                    )[0]
                except Exception:
                    trend = 0.0

            else:
                trend = 0.0

            features[
                residual_name + "_trend"
            ] = float(trend)

        # ==================================================
        # MISSION PHASE
        # ==================================================

        phase = window[-1].get(
            "mission_phase",
            "GROUND",
        )

        phase_encoded = self.phase_mapping.get(
            phase,
            0,
        )

        features[
            "mission_phase_encoded"
        ] = phase_encoded

        # ==================================================
        # EXACT TRAINING FEATURE ORDER
        # ==================================================

        feature_vector = []

        for feature_name in self.feature_names:

            feature_vector.append(
                features.get(
                    feature_name,
                    0.0,
                )
            )

        return np.array(
            feature_vector,
            dtype=float,
        ).reshape(1, -1)

    # ======================================================
    # FAULT PERSISTENCE
    # ======================================================

    def _apply_persistence(
        self,
        predicted_fault,
        confidence,
    ):

        # ==================================================
        # CURRENTLY NORMAL
        # ==================================================

        if self.current_fault == "NORMAL":

            if predicted_fault == "NORMAL":

                self.pending_fault = None
                self.pending_count = 0

                self.current_confidence = confidence

                return (
                    "NORMAL",
                    confidence,
                )

            # Candidate fault detected.

            if (
                self.pending_fault
                == predicted_fault
            ):
                self.pending_count += 1

            else:
                self.pending_fault = (
                    predicted_fault
                )
                self.pending_count = 1

            # Confirm fault after repeated detection.

            if (
                self.pending_count
                >= self.fault_confirmations
            ):

                self.current_fault = (
                    predicted_fault
                )

                self.current_confidence = (
                    confidence
                )

                self.pending_fault = None
                self.pending_count = 0

                return (
                    self.current_fault,
                    self.current_confidence,
                )

            # One noisy prediction is not enough.
            return (
                "NORMAL",
                confidence,
            )

        # ==================================================
        # ACTIVE FAULT CONTINUES
        # ==================================================

        if (
            predicted_fault
            == self.current_fault
        ):

            self.pending_fault = None
            self.pending_count = 0

            self.current_confidence = confidence

            return (
                self.current_fault,
                self.current_confidence,
            )

        # ==================================================
        # NORMAL PREDICTION DURING ACTIVE FAULT
        # ==================================================

        if predicted_fault == "NORMAL":

            if self.pending_fault == "NORMAL":

                self.pending_count += 1

            else:

                self.pending_fault = "NORMAL"
                self.pending_count = 1

            # Do not immediately clear the fault.

            if (
                self.pending_count
                >= self.normal_confirmations
            ):

                self.current_fault = "NORMAL"
                self.current_confidence = confidence

                self.pending_fault = None
                self.pending_count = 0

                return (
                    "NORMAL",
                    confidence,
                )

            # Keep reporting the active fault.
            return (
                self.current_fault,
                self.current_confidence,
            )

        # ==================================================
        # DIFFERENT FAULT TYPE
        # ==================================================

        if (
            self.pending_fault
            == predicted_fault
        ):

            self.pending_count += 1

        else:

            self.pending_fault = (
                predicted_fault
            )

            self.pending_count = 1

        if (
            self.pending_count
            >= self.fault_confirmations
        ):

            self.current_fault = (
                predicted_fault
            )

            self.current_confidence = (
                confidence
            )

            self.pending_fault = None
            self.pending_count = 0

        return (
            self.current_fault,
            self.current_confidence,
        )

    # ======================================================
    # ADD LIVE TELEMETRY SAMPLE
    # ======================================================

    def add_sample(self, residuals):

        sample = dict(residuals)

        self.buffer.append(sample)

        self.sample_count += 1

        # Need a complete 5-second window.
        if (
            len(self.buffer)
            < self.window_size
        ):
            return None

        # Predict once per second.
        if (
            self.sample_count
            % self.update_interval
            != 0
        ):
            return None

        # ==================================================
        # BUILD FEATURES
        # ==================================================

        X = self._build_features(
            list(self.buffer)
        )

        # ==================================================
        # XGBOOST PREDICTION
        # ==================================================

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        predicted_id = int(
            np.argmax(probabilities)
        )

        predicted_fault = (
            self.class_names[predicted_id]
        )

        confidence = float(
            probabilities[predicted_id]
        )

        # ==================================================
        # APPLY TEMPORAL PERSISTENCE
        # ==================================================

        (
            stable_fault,
            stable_confidence,
        ) = self._apply_persistence(
            predicted_fault,
            confidence,
        )

        stable_id = self.class_names.index(
            stable_fault
        )

        # ==================================================
        # RETURN RESULT
        # ==================================================

        return {
            "fault": stable_fault,

            "fault_id": stable_id,

            "confidence": round(
                stable_confidence,
                4,
            ),

            # Useful for debugging/demo.
            "raw_fault": predicted_fault,

            "raw_confidence": round(
                confidence,
                4,
            ),

            "probabilities": {
                self.class_names[i]: round(
                    float(probabilities[i]),
                    4,
                )
                for i in range(
                    len(self.class_names)
                )
            },
        }