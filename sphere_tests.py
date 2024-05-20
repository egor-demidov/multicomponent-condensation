import unittest
import numpy as np
from sphere import MultiComponentSystem, Component

ARO_1_COMPONENT = Component('ARO1', 5.7e-5, 298, 156e3, 150e-3, 1200.0, 0.01)
ARO_2_COMPONENT = Component('ARO2', 1.6e-3, 298, 156e3, 150e-3, 1100.0, 0.01)

def xi_function(t: float) -> np.ndarray[float]:
    return [1.2, 0.5]


class TestMultiComponentSystem(unittest.TestCase):

    def setUp(self):
        self.r_part_0 = 14e-9
        self.n_tot_0 = 5.0
        self.pressure_atm = 101_325.0
        self.temperature_sat = 298.0

        self.system = MultiComponentSystem(
            [ARO_1_COMPONENT, ARO_2_COMPONENT],
            xi_function,
            self.r_part_0,
            self.n_tot_0,
            self.pressure_atm,
            self.temperature_sat
        )

    def test_init(self):
        # Assert that initial condensed phase compositions add up to one
        np.testing.assert_allclose(np.sum(self.system.condensed_phase_mole_counts_0 / self.n_tot_0), 1.0)

        # Assert that core volume is correct
        np.testing.assert_allclose(self.system.core_volume, 4.0 / 3.0 * np.pi * self.r_part_0 ** 3.0)

        # Assert that molar volumes are correct
        np.testing.assert_allclose(self.system.molar_volumes, self.system.molar_masses / self.system.densities)

        # Assert that total volume is correct
        condensed_phase_mole_counts = np.array([2.0, 3.0])
        np.testing.assert_allclose(self.system.total_volume(condensed_phase_mole_counts), np.dot(condensed_phase_mole_counts, self.system.molar_volumes))

        # Assert that r_part is r_part_0 when there is no condensate
        np.testing.assert_allclose(self.system.r_part([0, 0]), self.system.r_part_0)

if __name__ == '__main__':
    unittest.main()
