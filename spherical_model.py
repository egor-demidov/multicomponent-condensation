import numpy as np
from multi_component_system import GeometryModel


def compute_r_part_effective(r_core: float, condensed_phase_volume: float) -> float:
    total_particle_volume = 4.0 / 3.0 * np.pi * r_core ** 3.0 + condensed_phase_volume
    return (3.0 * total_particle_volume / 4.0 / np.pi) ** (1.0 / 3.0)


class SphericalModel(GeometryModel):

    def compute_area(self, r_core: float, condensed_phase_volume: float) -> float:
        return 4.0 * np.pi * compute_r_part_effective(r_core, condensed_phase_volume) ** 2.0

    def compute_kappa(self, r_core: float, condensed_phase_volume: float) -> float:
        return 1.0 / compute_r_part_effective(r_core, condensed_phase_volume)
