from health import HealthIndex


health_system = HealthIndex()


test_fusion_result = {
    "overall_anomaly_score": 1.2,

    "thermal_deviation": 2.4,
    "lubrication_deviation": 0.5,
    "combustion_deviation": 0.3,
    "mechanical_deviation": 0.2,
    "electrical_deviation": 0.1,
}


result = health_system.calculate(
    test_fusion_result
)


print("\nENGINE HEALTH")
print("=" * 40)

print(
    f"Health Index       : "
    f"{result['health_index']:.1f}/100"
)

print(
    f"Status             : "
    f"{result['health_status']}"
)

print(
    f"Dominant subsystem : "
    f"{result['dominant_subsystem']}"
)