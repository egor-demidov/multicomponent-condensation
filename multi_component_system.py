import abc
import typing
import numpy as np


# Gas constant
R_GAS = 8.314  # J/mol/K
# Molecular accommodation coefficient
ALPHA = 1.0


class Component:

    # Component name, reference vapor pressure, reference temperature, enthalpy of vaporization, molar mass
    def __init__(
            self,
            name: str,                 # component name
            pressure_ref: float,       # reference vapor pressure
            temperature_ref: float,    # reference temperature
            dh_vap: float,             # enthalpy of vaporization
            molar_mass: float,         # molar mass
            density: float,            # condensed phase density
            surface_tension: float     # condensed phase surface tension
    ):
        self.name = name
        self.pressure_ref = pressure_ref
        self.temperature_ref = temperature_ref
        self.dh_vap = dh_vap
        self.molar_mass = molar_mass
        self.density = density
        self.surface_tension = surface_tension

    # Evaluate saturated vapor pressure (Clausius-Clapeyron eq-n)
    def pressure_sat(self, temperature_sat: float) -> float:
        return self.pressure_ref * np.exp(-self.dh_vap / R_GAS * (1.0 / temperature_sat - 1.0 / self.temperature_ref))


class GeometryModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def compute_area(self, r_core: float, condensed_phase_volume: float) -> float:
        pass

    @abc.abstractmethod
    def compute_kappa(self, r_core: float, condensed_phase_volume: float) -> float:
        pass


class MultiComponentSystem:

    def __init__(
            self,
            components: list[Component],                                # list of Component objects
            xi_function: typing.Callable[[float], np.ndarray[float]],   # function that returns saturation ratio of components as a function of time
            r_part_0: float,                                            # initial (core) radius
            n_tot_0: float,                                             # initial condensed phase total mole count
            pressure_atm: float,                                        # atmospheric pressure
            temperature_sat: float,                                     # ambient temperature
            geometry_model: GeometryModel                               # geometry model
    ):
        self.components = components
        self.r_part_0 = r_part_0
        self.temperature_sat = temperature_sat
        self.xi_function = xi_function
        self.geometry_model = geometry_model

        assert len(components) == len(xi_function(0.0))  # number of elements must match

        # Initialize saturated concentrations
        self.concentrations_sat = np.array([
            component.pressure_sat(temperature_sat) / R_GAS / temperature_sat for component in self.components
        ])

        # Initialize surface tensions
        self.surface_tensions = np.array([component.surface_tension for component in self.components])

        # Initialize densities
        self.densities = np.array([component.density for component in self.components])

        # Initialize molar masses
        self.molar_masses = np.array([component.molar_mass for component in self.components])

        # Initialize molar volumes
        self.molar_volumes = self.molar_masses / self.densities

        # Initialize liquid phase components mole counts
        self.concentration_atm = pressure_atm / R_GAS / temperature_sat
        vapor_phase_compositions_0 = self.concentrations_sat * xi_function(0.0) / self.concentration_atm
        condensed_phase_compositions_0 = vapor_phase_compositions_0 / np.sum(vapor_phase_compositions_0)
        self.condensed_phase_mole_counts_0 = n_tot_0 * condensed_phase_compositions_0

    def compute_condensed_phase_volume(self, condensed_phase_mole_counts: np.ndarray[float]) -> float:
        return np.dot(condensed_phase_mole_counts, self.molar_volumes)

    # @constrain([0, np.inf])
    def ode_function(self, t: float, condensed_phase_mole_counts: np.ndarray[float]) -> np.ndarray[float]:
        # find condensed phase mole fractions
        condensed_phase_compositions = condensed_phase_mole_counts / np.sum(condensed_phase_mole_counts)
        # find effective surface tension of the mixture
        surface_tension_effective = np.dot(condensed_phase_compositions, self.surface_tensions)
        # find condensed phase volume
        condensed_phase_volume = self.compute_condensed_phase_volume(condensed_phase_mole_counts)

        # find Kelvin correction factors
        kelvin_correction_factors = np.exp(2.0 * self.geometry_model.compute_kappa(self.r_part_0, condensed_phase_volume) * surface_tension_effective
                                          * self.molar_volumes
                                          / R_GAS / self.temperature_sat)

        return (1.0 / 4.0 * self.geometry_model.compute_area(self.r_part_0, condensed_phase_volume)
                * (8.0 * R_GAS * self.temperature_sat / np.pi / self.molar_masses) ** (1.0 / 2.0)
                * ALPHA * self.concentrations_sat
                * (self.xi_function(t) - condensed_phase_compositions * kelvin_correction_factors))
