from dataclasses import dataclass
import random

from simulator.engine import EngineState


@dataclass
class SensorTelemetry:

    rpm: float
    cht_c: float
    egt_c: float

    oil_pressure_bar: float
    oil_temperature_c: float

    fuel_flow_lph: float
    injection_timing_deg: float

    vibration_mm_s: float

    battery_voltage_v: float
    alternator_current_a: float


class SensorModel:

    def __init__(self):

        # Standard deviations for prototype sensor noise.

        self.rpm_noise = 5.0

        self.cht_noise = 0.5

        self.egt_noise = 1.5

        self.oil_pressure_noise = 0.03

        self.oil_temperature_noise = 0.3

        self.fuel_flow_noise = 0.1

        self.injection_timing_noise = 0.05

        self.vibration_noise = 0.05

        self.battery_noise = 0.05

        self.alternator_noise = 0.1

    def read(
        self,
        state: EngineState,
        cht_bias: float = 0.0,
        egt_bias: float = 0.0,
        oil_pressure_bias: float = 0.0
    ) -> SensorTelemetry:

        return SensorTelemetry(

            rpm=(
                state.rpm
                + random.gauss(
                    0,
                    self.rpm_noise
                )
            ),

            cht_c=(
                state.cht_c
                + cht_bias
                + random.gauss(
                    0,
                    self.cht_noise
                )
            ),

            egt_c=(
                state.egt_c
                +egt_bias
                + random.gauss(
                    0,
                    self.egt_noise
                )
            ),

            oil_pressure_bar=(
                state.oil_pressure_bar
                + oil_pressure_bias
                + random.gauss(
                    0,
                    self.oil_pressure_noise
                )
            ),

            oil_temperature_c=(
                state.oil_temperature_c
                + random.gauss(
                    0,
                    self.oil_temperature_noise
                )
            ),

            fuel_flow_lph=(
                state.fuel_flow_lph
                + random.gauss(
                    0,
                    self.fuel_flow_noise
                )
            ),

            injection_timing_deg=(
                state.injection_timing_deg
                + random.gauss(
                    0,
                    self.injection_timing_noise
                )
            ),

            vibration_mm_s=(
                state.vibration_mm_s
                + random.gauss(
                    0,
                    self.vibration_noise
                )
            ),

            battery_voltage_v=(
                state.battery_voltage_v
                + random.gauss(
                    0,
                    self.battery_noise
                )
            ),

            alternator_current_a=(
                state.alternator_current_a
                + random.gauss(
                    0,
                    self.alternator_noise
                )
            )
        )