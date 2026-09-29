import os
import json

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Aero-Piston Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
)


# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=1000,
    key="engine_live_refresh"
)


# =========================================================
# FILES
# =========================================================

DATA_FILE = "data/live_telemetry.csv"
CONTROL_FILE = "data/fault_control.json"


SUPPORTED_FAULTS = [
    "NORMAL",
    "COOLING_DEGRADATION",
    "INJECTOR_DEGRADATION",
    "LUBRICATION_DEGRADATION",
    "MECHANICAL_DEGRADATION",
    "MISFIRE",
    "SENSOR_DRIFT",
]


# =========================================================
# TITLE
# =========================================================

st.title(
    "✈️ Aero-Piston Engine Health Monitor"
)

st.caption(
    "Live Synthetic Telemetry | Digital Twin | "
    "Residual Analysis | ML Fault Detection | "
    "Health Index | Prototype RUL"
)


# =========================================================
# SIDEBAR CONTROL
# =========================================================

st.sidebar.header(
    "🎛️ Fault Injection Control"
)

selected_fault = st.sidebar.selectbox(
    "Fault Type",
    SUPPORTED_FAULTS
)

severity_percent = st.sidebar.slider(
    "Fault Severity",
    min_value=0,
    max_value=100,
    value=0,
    step=5
)

apply_fault = st.sidebar.button(
    "🚨 APPLY FAULT",
    width="stretch"
)


# =========================================================
# WRITE CONTROL
# =========================================================

if apply_fault:

    os.makedirs(
        "data",
        exist_ok=True
    )

    control = {
        "fault_type": selected_fault,
        "severity": severity_percent / 100.0
    }

    with open(
        CONTROL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            control,
            file,
            indent=2
        )

    st.sidebar.success(
        f"{selected_fault.replace('_', ' ')} "
        f"at {severity_percent}% applied."
    )


# =========================================================
# RESET BUTTON
# =========================================================

if st.sidebar.button(
    "✅ RETURN TO NORMAL",
    width="stretch"
):

    os.makedirs(
        "data",
        exist_ok=True
    )

    control = {
        "fault_type": "NORMAL",
        "severity": 0.0
    }

    with open(
        CONTROL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            control,
            file,
            indent=2
        )

    st.sidebar.success(
        "Engine returned to NORMAL."
    )


# =========================================================
# LOAD LIVE TELEMETRY
# =========================================================

if not os.path.exists(DATA_FILE):

    st.warning(
        "Waiting for live telemetry..."
    )

    st.info(
        "Start the engine monitor with:\n\n"
        "python -m digital_twin.live_engine_monitor"
    )

    st.stop()


try:

    df = pd.read_csv(
        DATA_FILE
    )

except Exception:

    st.warning(
        "Telemetry file is being updated..."
    )

    st.stop()


if df.empty:

    st.warning(
        "Waiting for telemetry..."
    )

    st.stop()


latest = df.iloc[-1]


# =========================================================
# LIVE VALUES
# =========================================================

timestamp = float(
    latest.get(
        "timestamp_s",
        0
    )
)

phase = str(
    latest.get(
        "mission_phase",
        "UNKNOWN"
    )
)

health = float(
    latest.get(
        "health_index",
        100.0
    )
)

status = str(
    latest.get(
        "health_status",
        "HEALTHY"
    )
)

detected_fault = str(
    latest.get(
        "detected_fault",
        "NORMAL"
    )
)

fault_confidence = float(
    latest.get(
        "fault_confidence",
        0.0
    )
)

fault_severity = float(
    latest.get(
        "fault_severity",
        0.0
    )
)

rul = float(
    latest.get(
        "rul_hours",
        500.0
    )
)

degradation_rate = float(
    latest.get(
        "degradation_rate_per_hour",
        0.0
    )
)

rul_trend = str(
    latest.get(
        "rul_trend",
        "STABLE"
    )
)


rpm = float(
    latest.get(
        "rpm",
        0
    )
)

cht = float(
    latest.get(
        "cht_c",
        0
    )
)

egt = float(
    latest.get(
        "egt_c",
        0
    )
)

oil_temp = float(
    latest.get(
        "oil_temperature_c",
        0
    )
)

oil_pressure = float(
    latest.get(
        "oil_pressure_bar",
        0
    )
)

fuel_flow = float(
    latest.get(
        "fuel_flow_lph",
        0
    )
)

vibration = float(
    latest.get(
        "vibration_mm_s",
        0
    )
)

battery = float(
    latest.get(
        "battery_voltage_v",
        0
    )
)

alternator = float(
    latest.get(
        "alternator_current_a",
        0
    )
)


# =========================================================
# DISPLAY VALUES
# =========================================================

fault_display = detected_fault.replace(
    "_",
    " "
)

phase_display = phase.replace(
    "_",
    " "
)

if status == "HEALTHY":

    status_display = "🟢 HEALTHY"

elif status == "DEGRADED":

    status_display = "🟡 DEGRADED"

elif status == "WARNING":

    status_display = "🟠 WARNING"

else:

    status_display = "🔴 CRITICAL"


# =========================================================
# ENGINE STATUS
# =========================================================

st.subheader(
    "Engine Health Status"
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Health Index",
        f"{health:.1f} / 100"
    )


with c2:

    st.metric(
        "Engine Status",
        status_display
    )


with c3:

    st.metric(
        "Detected Fault",
        fault_display
    )


with c4:

    st.metric(
        "Prototype RUL",
        f"{rul:.1f} h"
    )


# =========================================================
# AI DIAGNOSTICS
# =========================================================

st.divider()

st.subheader(
    "AI Diagnostic Output"
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "ML Confidence",
        f"{fault_confidence * 100:.1f}%"
    )


with c2:

    st.metric(
        "Fault Severity",
        f"{fault_severity * 100:.1f}%"
    )


with c3:

    st.metric(
        "Degradation Rate",
        f"{degradation_rate:.2f} pts/h"
    )


with c4:

    st.metric(
        "RUL Trend",
        rul_trend
    )


# =========================================================
# MISSION
# =========================================================

st.divider()

st.subheader(
    "Mission Information"
)

c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Mission Phase",
        phase_display
    )


with c2:

    st.metric(
        "Simulation Time",
        f"{timestamp:.1f} s"
    )


with c3:

    st.metric(
        "Telemetry Samples",
        f"{len(df):,}"
    )


# =========================================================
# LIVE ENGINE PARAMETERS
# =========================================================

st.divider()

st.subheader(
    "Live Engine Parameters"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)


with c1:

    st.metric(
        "RPM",
        f"{rpm:,.0f}"
    )


with c2:

    st.metric(
        "CHT",
        f"{cht:.1f} °C"
    )


with c3:

    st.metric(
        "EGT",
        f"{egt:.1f} °C"
    )


with c4:

    st.metric(
        "Oil Temp",
        f"{oil_temp:.1f} °C"
    )


with c5:

    st.metric(
        "Oil Pressure",
        f"{oil_pressure:.2f} bar"
    )


with c6:

    st.metric(
        "Fuel Flow",
        f"{fuel_flow:.2f} L/h"
    )


# =========================================================
# ADDITIONAL PARAMETERS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Vibration",
        f"{vibration:.2f} mm/s"
    )


with c2:

    st.metric(
        "Battery",
        f"{battery:.2f} V"
    )


with c3:

    st.metric(
        "Alternator Current",
        f"{alternator:.2f} A"
    )


with c4:

    st.metric(
        "Fault Severity",
        f"{fault_severity * 100:.1f}%"
    )


# =========================================================
# MAINTENANCE
# =========================================================

st.divider()

st.subheader(
    "Maintenance Recommendation"
)


if detected_fault == "NORMAL":

    st.success(
        "✓ Engine operating normally. "
        "Continue operation and monitor engine health."
    )

elif detected_fault == "COOLING_DEGRADATION":

    st.warning(
        "⚠ Cooling degradation detected. "
        "Inspect cooling system, airflow and thermal "
        "management before continued high-load operation."
    )

elif detected_fault == "INJECTOR_DEGRADATION":

    st.warning(
        "⚠ Injector degradation detected. "
        "Inspect injector performance and fuel delivery."
    )

elif detected_fault == "LUBRICATION_DEGRADATION":

    st.warning(
        "⚠ Lubrication degradation detected. "
        "Inspect oil pressure, oil temperature and "
        "lubrication system."
    )

elif detected_fault == "MECHANICAL_DEGRADATION":

    st.error(
        "⚠ Mechanical degradation detected. "
        "Schedule detailed mechanical inspection."
    )

elif detected_fault == "MISFIRE":

    st.error(
        "⚠ Misfire detected. "
        "Inspect ignition, fuel delivery and combustion."
    )

elif detected_fault == "SENSOR_DRIFT":

    st.warning(
        "⚠ Sensor drift detected. "
        "Validate affected sensors and instrumentation."
    )

else:

    st.warning(
        "⚠ Abnormal engine condition detected. "
        "Investigate telemetry and diagnostic outputs."
    )


# =========================================================
# TELEMETRY TRENDS
# =========================================================

st.divider()

st.subheader(
    "Live Telemetry Trends"
)

chart_df = df.tail(
    300
).copy()


if "timestamp_s" in chart_df.columns:

    chart_df = chart_df.set_index(
        "timestamp_s"
    )


c1, c2 = st.columns(2)


with c1:

    st.markdown("**RPM**")

    st.line_chart(
        chart_df["rpm"]
    )

    st.markdown(
        "**Cylinder Head Temperature**"
    )

    st.line_chart(
        chart_df["cht_c"]
    )

    st.markdown(
        "**Oil Temperature**"
    )

    st.line_chart(
        chart_df["oil_temperature_c"]
    )


with c2:

    st.markdown(
        "**Exhaust Gas Temperature**"
    )

    st.line_chart(
        chart_df["egt_c"]
    )

    st.markdown(
        "**Oil Pressure**"
    )

    st.line_chart(
        chart_df["oil_pressure_bar"]
    )

    st.markdown(
        "**Fuel Flow**"
    )

    st.line_chart(
        chart_df["fuel_flow_lph"]
    )


# =========================================================
# HEALTH / RUL
# =========================================================

st.divider()

st.subheader(
    "Health & Prognostics"
)

c1, c2 = st.columns(2)


with c1:

    if "health_index" in chart_df.columns:

        st.markdown(
            "**Health Index Trend**"
        )

        st.line_chart(
            chart_df["health_index"]
        )


with c2:

    if "rul_hours" in chart_df.columns:

        st.markdown(
            "**RUL Trend**"
        )

        st.line_chart(
            chart_df["rul_hours"]
        )


# =========================================================
# LATEST DATA
# =========================================================

st.divider()

with st.expander(
    "Show Latest Telemetry"
):

    st.dataframe(
        df.tail(10),
        width="stretch"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Prototype demonstration using synthetic engine telemetry. "
    "RUL is an illustrative estimate and is not certified "
    "for physical-engine operation."
)