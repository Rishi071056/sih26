import math
from dataclasses import dataclass


R_AIR = 287.05  # J/(kg·K)


@dataclass
class EnvironmentState:
    altitude_ft: float
    ambient_temperature_c: float
    ambient_pressure_kpa: float
    air_density_kg_m3: float


def pressure_from_altitude(altitude_ft: float) -> float:
    """
    Simplified standard-atmosphere pressure model.

    Prototype approximation only.
    Not intended for flight certification.
    """

    altitude_m = max(0.0, altitude_ft) * 0.3048

    sea_level_pressure_kpa = 101.325
    scale_height_m = 8434.5

    pressure_kpa = (
        sea_level_pressure_kpa
        * math.exp(-altitude_m / scale_height_m)
    )

    return pressure_kpa


def calculate_air_density(
    pressure_kpa: float,
    temperature_c: float
) -> float:

    pressure_pa = pressure_kpa * 1000.0
    temperature_k = temperature_c + 273.15

    return pressure_pa / (R_AIR * temperature_k)


def create_environment(
    altitude_ft: float,
    ambient_temperature_c: float
) -> EnvironmentState:

    pressure_kpa = pressure_from_altitude(
        altitude_ft
    )

    density = calculate_air_density(
        pressure_kpa,
        ambient_temperature_c
    )

    return EnvironmentState(
        altitude_ft=altitude_ft,
        ambient_temperature_c=ambient_temperature_c,
        ambient_pressure_kpa=pressure_kpa,
        air_density_kg_m3=density
    )