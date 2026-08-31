from dataclasses import dataclass

from simulator.environment import EnvironmentState


@dataclass
class EngineInputs:

    throttle_pct: float
    engine_load_pct: float

    environment: EnvironmentState

    # ---------------------------------------------------------
    # HEALTH / DEGRADATION FACTORS
    #
    # 1.0 = healthy
    # 0.0 = severely degraded
    # ---------------------------------------------------------

    cooling_health: float = 1.0
    injector_health: float = 1.0
    lubrication_health: float = 1.0
    mechanical_health: float = 1.0

    # 0.0 = no misfire
    # 1.0 = severe misfire
    misfire_severity: float = 0.0


@dataclass
class EngineState:

    rpm: float = 2500.0

    fuel_flow_lph: float = 5.0

    egt_c: float = 400.0

    cht_c: float = 80.0

    oil_temperature_c: float = 70.0

    oil_pressure_bar: float = 3.5

    injection_timing_deg: float = 24.0

    vibration_mm_s: float = 1.0

    battery_voltage_v: float = 24.0

    alternator_current_a: float = 5.0


class AeroPistonEngine:

    def __init__(self):

        self.state = EngineState()

        # -----------------------------------------------------
        # PROTOTYPE AERO-PISTON ENGINE ASSUMPTIONS
        # -----------------------------------------------------

        self.idle_rpm = 2500.0

        self.max_rpm = 7500.0

        # Response times
        self.rpm_response_time = 2.0
        self.egt_response_time = 8.0

        # Reduced from 35 s slightly so degradation becomes
        # visible during our SIH demonstration.
        self.cht_response_time = 25.0

        self.oil_response_time = 60.0

    # =========================================================
    # MAIN UPDATE
    # =========================================================

    def update(
        self,
        inputs: EngineInputs,
        dt: float
    ) -> EngineState:

        throttle = max(
            0.0,
            min(inputs.throttle_pct / 100.0, 1.0)
        )

        load = max(
            0.0,
            min(inputs.engine_load_pct / 100.0, 1.0)
        )

        environment = inputs.environment

        # Clamp health values so invalid values cannot
        # destabilize the engine model.
        cooling_health = max(
            0.0,
            min(inputs.cooling_health, 1.0)
        )

        injector_health = max(
            0.0,
            min(inputs.injector_health, 1.0)
        )

        lubrication_health = max(
            0.0,
            min(inputs.lubrication_health, 1.0)
        )

        mechanical_health = max(
            0.0,
            min(inputs.mechanical_health, 1.0)
        )

        misfire_severity = max(
            0.0,
            min(inputs.misfire_severity, 1.0)
        )

        # =====================================================
        # 1. AIR DENSITY EFFECT
        # =====================================================

        sea_level_density = 1.225

        density_ratio = (
            environment.air_density_kg_m3
            / sea_level_density
        )

        density_ratio = max(
            0.55,
            min(1.10, density_ratio)
        )

        # =====================================================
        # 2. TARGET RPM
        # =====================================================

        target_rpm = (
            self.idle_rpm
            + throttle
            * (self.max_rpm - self.idle_rpm)
        )

        # Engine load reduces available RPM slightly.
        target_rpm *= (
            1.0
            - 0.05 * load
        )

        # Reduced air density at altitude.
        target_rpm *= (
            0.92
            + 0.08 * density_ratio
        )

        # Mechanical degradation causes some loss of RPM.
        target_rpm *= (
            1.0
            - 0.08 * (1.0 - mechanical_health)
        )

        # Misfire reduces effective combustion torque.
        target_rpm *= (
            1.0
            - 0.10 * misfire_severity
        )

        # =====================================================
        # 3. RPM DYNAMICS
        # =====================================================

        rpm_error = (
            target_rpm
            - self.state.rpm
        )

        self.state.rpm += (
            rpm_error
            * dt
            / self.rpm_response_time
        )

        self.state.rpm = max(
            self.idle_rpm * 0.8,
            min(self.state.rpm, self.max_rpm * 1.02)
        )

        # =====================================================
        # 4. FUEL FLOW
        # =====================================================

        base_fuel = 3.0

        fuel_from_throttle = (
            18.0 * throttle
        )

        fuel_from_load = (
            4.0 * load
        )

        fuel_density_effect = (
            2.0 * (1.0 - density_ratio)
        )

        # A degraded injector requires abnormal fuel delivery.
        injector_effect = (
            1.0
            + 0.15 * (1.0 - injector_health)
        )

        # Misfire produces inefficient combustion.
        misfire_effect = (
            1.0
            + 0.05 * misfire_severity
        )

        self.state.fuel_flow_lph = (
            base_fuel
            + fuel_from_throttle
            + fuel_from_load
            + fuel_density_effect
        ) * injector_effect * misfire_effect

        # =====================================================
        # 5. INJECTION TIMING
        # =====================================================

        # Timing becomes slightly abnormal with injector
        # degradation.
        injector_timing_error = (
            2.5 * (1.0 - injector_health)
        )

        self.state.injection_timing_deg = (
            24.0
            + 2.0 * throttle
            + injector_timing_error
        )

        # =====================================================
        # 6. EGT
        # =====================================================

        mixture_effect = (
            1.0
            + 0.5 * (1.0 - injector_health)
        )

        target_egt = (
            400.0
            + 280.0
            * load
            * mixture_effect
            + 50.0 * throttle
        )

        # Injector degradation increases abnormal combustion
        # temperature.
        target_egt += (
            60.0 * (1.0 - injector_health)
        )

        # Altitude effect.
        target_egt += (
            25.0 * (1.0 - density_ratio)
        )

        # Misfire creates combustion instability.
        # We don't completely remove heat because a misfiring
        # cylinder still produces intermittent combustion.
        target_egt *= (
            1.0
            - 0.08 * misfire_severity
        )

        egt_error = (
            target_egt
            - self.state.egt_c
        )

        self.state.egt_c += (
            egt_error
            * dt
            / self.egt_response_time
        )

        # =====================================================
        # 7. CHT / THERMAL MODEL
        # =====================================================

        # -----------------------------------------------------
        # Combustion heat generation
        # -----------------------------------------------------

        heat_generation = (
            0.70 * load
            + 0.25 * throttle
        )

        # Injector degradation can increase abnormal heat
        # release.
        heat_generation *= (
            1.0
            + 0.15 * (1.0 - injector_health)
        )

        # Misfire reduces average combustion heat.
        heat_generation *= (
            1.0
            - 0.10 * misfire_severity
        )

        # -----------------------------------------------------
        # Cooling effectiveness
        # -----------------------------------------------------

        airflow_factor = (
            max(self.state.rpm, 1000.0)
            / self.max_rpm
        )

        airflow_factor = max(
            0.20,
            min(1.10, airflow_factor)
        )

        # Cooling capability depends on:
        #
        # 1. cooling system health
        # 2. air density
        # 3. engine RPM / airflow
        #
        cooling_effectiveness = (
            cooling_health
            * density_ratio
            * (
                0.35
                + 0.65 * airflow_factor
            )
        )

        # -----------------------------------------------------
        # Ambient-to-cylinder temperature difference
        # -----------------------------------------------------

        temperature_difference = max(
            self.state.cht_c
            - environment.ambient_temperature_c,
            0.0
        )

        # -----------------------------------------------------
        # Heat removal
        #
        # Stronger than previous model so that a cooling fault
        # produces a clearly measurable CHT response.
        # -----------------------------------------------------

        heat_removed = (
            0.045
            * cooling_effectiveness
            * temperature_difference
        )

        # -----------------------------------------------------
        # Additional cooling degradation penalty
        #
        # This represents reduced heat transfer capability
        # due to degraded cooling performance.
        # -----------------------------------------------------

        degradation_heat_penalty = (
            0.65
            * (1.0 - cooling_health)
            * (
                0.30
                + load
                + 0.50 * throttle
            )
        )

        # -----------------------------------------------------
        # Net thermal rate
        # -----------------------------------------------------

        cht_rate = (
            3.0 * heat_generation
            - heat_removed
            + degradation_heat_penalty
        )

        self.state.cht_c += (
            cht_rate
            * dt
            / self.cht_response_time
        )

        # Prevent unrealistic low CHT.
        self.state.cht_c = max(
            environment.ambient_temperature_c + 20.0,
            self.state.cht_c
        )

        # Prevent runaway temperatures in the prototype model.
        self.state.cht_c = min(
            self.state.cht_c,
            220.0
        )

        # =====================================================
        # 8. OIL TEMPERATURE
        # =====================================================

        target_oil_temperature = (
            environment.ambient_temperature_c
            + 40.0
            + 0.28
            * (
                self.state.cht_c
                - 100.0
            )
        )

        # Cooling degradation increases oil temperature.
        target_oil_temperature += (
            20.0
            * (1.0 - cooling_health)
        )

        # Lubrication degradation also causes additional
        # thermal stress.
        target_oil_temperature += (
            8.0
            * (1.0 - lubrication_health)
        )

        oil_error = (
            target_oil_temperature
            - self.state.oil_temperature_c
        )

        self.state.oil_temperature_c += (
            oil_error
            * dt
            / self.oil_response_time
        )

        self.state.oil_temperature_c = max(
            environment.ambient_temperature_c + 10.0,
            min(self.state.oil_temperature_c, 150.0)
        )

        # =====================================================
        # 9. OIL PRESSURE
        # =====================================================

        rpm_factor = (
            self.state.rpm
            / 7000.0
        )

        temperature_factor = max(
            0.65,
            1.0
            - 0.0025
            * (
                self.state.oil_temperature_c
                - 80.0
            )
        )

        lubrication_factor = (
            0.55
            + 0.45 * lubrication_health
        )

        self.state.oil_pressure_bar = (
            2.0
            + 3.5
            * rpm_factor
            * temperature_factor
            * lubrication_factor
        )

        self.state.oil_pressure_bar = max(
            0.8,
            min(self.state.oil_pressure_bar, 6.5)
        )

        # =====================================================
        # 10. VIBRATION
        # =====================================================

        rpm_component = (
            0.00010
            * self.state.rpm
        )

        load_component = (
            0.8 * load
        )

        mechanical_component = (
            3.5
            * (1.0 - mechanical_health)
        )

        misfire_component = (
            2.5
            * misfire_severity
        )

        # Injector degradation contributes a smaller vibration
        # signature because combustion becomes less uniform.
        injector_vibration_component = (
            0.8
            * (1.0 - injector_health)
        )

        self.state.vibration_mm_s = (
            0.5
            + rpm_component
            + load_component
            + mechanical_component
            + misfire_component
            + injector_vibration_component
        )

        # =====================================================
        # 11. ELECTRICAL SYSTEM
        # =====================================================

        alternator_efficiency = (
            0.80
            + 0.20
            * min(
                self.state.rpm / 5000.0,
                1.0
            )
        )

        self.state.alternator_current_a = (
            5.0
            + 15.0
            * throttle
            * alternator_efficiency
        )

        self.state.battery_voltage_v = (
            24.0
            + 0.6
            * min(
                self.state.rpm / 5000.0,
                1.0
            )
        )

        return self.state