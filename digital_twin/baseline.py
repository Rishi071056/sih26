# Healthy Digital Twin residual baseline
#
# These values were calculated from the healthy normal_001.csv mission.
#
# mean = average healthy residual
# std  = normal variation of the residual
#
# These are prototype statistical baselines, not certified engine limits.


BASELINE = {

    "TAKEOFF": {

        "rpm_residual": {
            "mean": -140.666,
            "std": 481.870
        },

        "cht_c_residual": {
            "mean": -0.037,
            "std": 0.490
        },

        "egt_c_residual": {
            "mean": -30.706,
            "std": 45.936
        },

        "oil_pressure_bar_residual": {
            "mean": -0.041,
            "std": 0.253
        },

        "oil_temperature_c_residual": {
            "mean": -1.027,
            "std": 0.888
        },

        "fuel_flow_lph_residual": {
            "mean": 0.005,
            "std": 0.097
        },

        "injection_timing_deg_residual": {
            "mean": -0.001,
            "std": 0.052
        },

        "vibration_mm_s_residual": {
            "mean": -0.014,
            "std": 0.066
        },

        "battery_voltage_v_residual": {
            "mean": -0.001,
            "std": 0.058
        },

        "alternator_current_a_residual": {
            "mean": -0.020,
            "std": 0.148
        },
    },


    "CLIMB": {

        "rpm_residual": {
            "mean": 15.761,
            "std": 37.975
        },

        "cht_c_residual": {
            "mean": -0.026,
            "std": 0.491
        },

        "egt_c_residual": {
            "mean": 2.407,
            "std": 3.456
        },

        "oil_pressure_bar_residual": {
            "mean": 0.063,
            "std": 0.036
        },

        "oil_temperature_c_residual": {
            "mean": -3.543,
            "std": 1.195
        },

        "fuel_flow_lph_residual": {
            "mean": -0.003,
            "std": 0.100
        },

        "injection_timing_deg_residual": {
            "mean": 0.002,
            "std": 0.051
        },

        "vibration_mm_s_residual": {
            "mean": 0.002,
            "std": 0.049
        },

        "battery_voltage_v_residual": {
            "mean": -0.002,
            "std": 0.051
        },

        "alternator_current_a_residual": {
            "mean": -0.002,
            "std": 0.103
        },
    }
}