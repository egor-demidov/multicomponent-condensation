import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import CubicSpline
from multi_component_system import GeometryModel

import matplotlib.pyplot as plt


NUM_CATENOID_POINTS = 200


def catenoid_ode(t, y, kappa):
    return np.array([
        y[1],
        (1.0 + y[1] ** 2.0) ** (3.0 / 2.0) * (2.0 * kappa[0] + 1.0 / (y[0] * (1.0 + y[1] ** 2.0) ** (1.0 / 2.0)))
    ])


def catenoid_bc(ya, yb, kappa, filling_angle, contact_angle):
    return np.array([
        ya[0] - np.sin(filling_angle),
        ya[1] + np.tan(np.pi / 2.0 - filling_angle - contact_angle),
        yb[1]
    ])


def calculate_volume(t_span, y, filling_angle):
    integrand = np.pi * y ** 2
    cap_volume = np.pi * 1.0 / 3.0 * (np.cos(filling_angle) - 1.0) ** 2.0 * (2.0 + np.cos(filling_angle))
    return np.trapz(integrand, t_span) - 2.0 * cap_volume


def calculate_area(t_span, y, y_prime):
    integrand = 2.0 * np.pi * y * np.sqrt(1 + y_prime ** 2.0)
    return np.trapz(integrand, t_span)


class VolumeOutOfBoundsException(Exception):
    pass

class CatenoidModel(GeometryModel):

    def __init__(self, contact_angle, n_filling_angles):

        filling_angles = np.linspace(0.05, np.pi / 2.0, n_filling_angles)
        # catenoid_data_table columns: filling angle, volume, area, curvature
        self.catenoid_data_table = np.zeros([len(filling_angles), 4])

        self.catenoid_data_table[:, 0] = filling_angles

        for i in range(len(filling_angles)):

            filling_angle = filling_angles[i]

            kappa_initial_guess = np.array([0])

            t_span = np.linspace(0.0, 1.0 - np.cos(filling_angle), NUM_CATENOID_POINTS)
            y_initial_guess = np.zeros((2, t_span.size))
            y_initial_guess[0, :] = np.sin(filling_angle)

            sol = solve_bvp(
                catenoid_ode,
                lambda ya, yb, kappa: catenoid_bc(ya, yb, kappa, filling_angle, contact_angle),
                t_span, y_initial_guess, p=kappa_initial_guess
            )

            curvature = -sol.p[0]

            t_catenoid = np.concatenate((sol.x, sol.x[1:] + sol.x[-1]))
            y_catenoid = np.concatenate((sol.y[0, :], np.flip(sol.y[0, :-1])))
            y_prime_catenoid = np.concatenate((sol.y[1, :], -np.flip(sol.y[1, :-1])))

            volume = calculate_volume(t_catenoid, y_catenoid, filling_angle)
            area = calculate_area(t_catenoid, y_catenoid, y_prime_catenoid)

            self.catenoid_data_table[i, 1] = volume
            self.catenoid_data_table[i, 2] = area
            self.catenoid_data_table[i, 3] = curvature

        increasing_volume_order = np.argsort(self.catenoid_data_table[:, 1])
        self.catenoid_data_table[:, :] = self.catenoid_data_table[increasing_volume_order, :]

        self.reduced_area_spline = CubicSpline(self.catenoid_data_table[:, 1], self.catenoid_data_table[:, 2])
        self.reduced_kappa_spline = CubicSpline(self.catenoid_data_table[:, 1], self.catenoid_data_table[:, 3])

    def validate_volume(self, reduced_condensed_phase_volume: float):
        if (reduced_condensed_phase_volume < self.catenoid_data_table[0, 1]
                or reduced_condensed_phase_volume > self.catenoid_data_table[-1, 1]):
            raise VolumeOutOfBoundsException("Condensed phase volume out of bounds")

    def compute_area(self, r_core: float, condensed_phase_volume: float) -> float:
        reduced_condensed_phase_volume = condensed_phase_volume / r_core ** 3.0
        self.validate_volume(reduced_condensed_phase_volume)

        return self.reduced_area_spline(reduced_condensed_phase_volume) * r_core ** 2.0

    def compute_kappa(self, r_core: float, condensed_phase_volume: float) -> float:
        reduced_condensed_phase_volume = condensed_phase_volume / r_core ** 3.0
        self.validate_volume(reduced_condensed_phase_volume)

        return self.reduced_kappa_spline(reduced_condensed_phase_volume) / r_core

