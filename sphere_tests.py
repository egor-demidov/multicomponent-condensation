import unittest
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from multi_component_system import MultiComponentSystem, Component
from spherical_model import SphericalModel

ARO_1_COMPONENT = Component('ARO1', 5.7e-5, 298, 156e3, 150e-3, 1200.0, 0.01)
ARO_2_COMPONENT = Component('ARO2', 1.6e-3, 298, 156e3, 150e-3, 1100.0, 0.01)

def xi_function_binary(t: float) -> np.ndarray[float]:
    return np.array([1.5, 1.5]) * 0.5

def xi_function_unary(t: float) -> np.ndarray[float]:
    return np.array([1.5])


class TestSingleMultiComponentParity(unittest.TestCase):

    def setUp(self):
        self.r_part_0 = 14e-9
        self.n_tot_0 = 1.0e-21
        self.pressure_atm = 101_325.0
        self.temperature_sat = 298.0

        self.system_single = MultiComponentSystem(
            [ARO_1_COMPONENT],
            xi_function_unary,
            self.r_part_0,
            self.n_tot_0,
            self.pressure_atm,
            self.temperature_sat,
            SphericalModel()
        )

        self.system_double = MultiComponentSystem(
            [ARO_1_COMPONENT, ARO_1_COMPONENT],
            xi_function_binary,
            self.r_part_0,
            self.n_tot_0,
            self.pressure_atm,
            self.temperature_sat,
            SphericalModel()
        )

    def test_parity(self):

        double_system_compositions = (self.system_double.condensed_phase_mole_counts_0
                                      / np.sum(self.system_double.condensed_phase_mole_counts_0))

        np.testing.assert_allclose(self.system_single.molar_volumes[0],
                                   np.dot(self.system_double.molar_volumes, double_system_compositions))

        np.testing.assert_allclose(self.system_single.molar_masses[0],
                                   np.dot(self.system_double.molar_masses, double_system_compositions))

        np.testing.assert_allclose(self.system_single.surface_tensions[0],
                                  np.dot(self.system_double.surface_tensions, double_system_compositions))

        # np.testing.assert_allclose(self.system_single.total_volume(self.system_single.condensed_phase_mole_counts_0),
        #                            self.system_double.total_volume(self.system_double.condensed_phase_mole_counts_0))
        #
        # np.testing.assert_allclose(self.system_single.r_part(self.system_single.condensed_phase_mole_counts_0),
        #                            self.system_double.r_part(self.system_double.condensed_phase_mole_counts_0))

        np.testing.assert_allclose(self.system_single.ode_function(0.0, self.system_single.condensed_phase_mole_counts_0)[0],
                                   np.sum(self.system_double.ode_function(0.0, self.system_double.condensed_phase_mole_counts_0)))


class TestSingleComponentSystem(unittest.TestCase):

    def setUp(self):
        self.r_part_0 = 14e-9
        self.n_tot_0 = 1.0e-21
        self.pressure_atm = 101_325.0
        self.temperature_sat = 298.0

        self.system = MultiComponentSystem(
            [ARO_2_COMPONENT],
            xi_function_unary,
            self.r_part_0,
            self.n_tot_0,
            self.pressure_atm,
            self.temperature_sat
        )

    # def test_condensation(self):
    #     t_span = np.linspace(0, 1000, 100)
    #
    #     sol = odeint(self.system.ode_function, self.system.condensed_phase_mole_counts_0, t_span)
    #
    #     plt.plot(t_span, self.system.r_part(sol) * 1e9)
    #     plt.show()

class TestMultiComponentSystem(unittest.TestCase):

    def setUp(self):
        self.r_part_0 = 14e-9
        self.n_tot_0 = 1.0e-21
        self.pressure_atm = 101_325.0
        self.temperature_sat = 298.0

        self.system = MultiComponentSystem(
            [ARO_1_COMPONENT, ARO_2_COMPONENT],
            xi_function_binary,
            self.r_part_0,
            self.n_tot_0,
            self.pressure_atm,
            self.temperature_sat,
            SphericalModel()
        )

    def test_init(self):
        # Assert that initial condensed phase compositions add up to one
        np.testing.assert_allclose(np.sum(self.system.condensed_phase_mole_counts_0 / self.n_tot_0), 1.0)

        # Assert that core volume is correct
        # np.testing.assert_allclose(self.system.core_volume, 4.0 / 3.0 * np.pi * self.r_part_0 ** 3.0)

        # Assert that molar volumes are correct
        np.testing.assert_allclose(self.system.molar_volumes, self.system.molar_masses / self.system.densities)

        # Assert that total volume is correct
        condensed_phase_mole_counts = np.array([2.0, 3.0])
        # np.testing.assert_allclose(self.system.total_volume(condensed_phase_mole_counts), np.dot(condensed_phase_mole_counts, self.system.molar_volumes))

        # Assert that r_part is r_part_0 when there is no condensate
        # np.testing.assert_allclose(self.system.r_part([0, 0]), self.system.r_part_0)


if __name__ == '__main__':
    unittest.main()
