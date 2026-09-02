from degradation import DegradationAnalyzer


analyzer = DegradationAnalyzer()


timestamps = [
    0,
    1,
    2,
    3,
    4,
    5
]

cht_residuals = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5
]

oil_temperature_residuals = [
    0.5,
    0.7,
    0.9,
    1.1,
    1.3,
    1.5
]

egt_residuals = [
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5
]

vibration_residuals = [
    0.1,
    0.1,
    0.2,
    0.2,
    0.3,
    0.3
]


residual_history = {

    "cht_c_residual":
        cht_residuals,

    "oil_temperature_c_residual":
        oil_temperature_residuals,

    "egt_c_residual":
        egt_residuals,

    "vibration_mm_s_residual":
        vibration_residuals,
}


result = analyzer.analyze(
    timestamps,
    residual_history
)


print("\nDEGRADATION ANALYSIS")
print("=" * 50)

for key, value in result.items():

    print(
        f"{key:35s}: {value:.5f}"
    )