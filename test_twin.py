from digital_twin.twin_engine import DigitalTwinEngine


twin = DigitalTwinEngine()


result = twin.estimate(

    throttle_pct=70.0,

    engine_load_pct=65.0,

    altitude_ft=5000.0,

    ambient_temperature_c=25.0,

    air_density_kg_m3=0.736
)


print()
print("=" * 60)
print(" DIGITAL TWIN TEST")
print("=" * 60)
print()

for parameter, value in result.items():

    print(
        f"{parameter:<35} : {value:.2f}"
    )

print()