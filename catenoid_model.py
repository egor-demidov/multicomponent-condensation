import numpy as np
from scipy.integrate import solve_bvp
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

class CatenoidModel(GeometryModel):

    def __init__(self, contact_angle, n_filling_angles):

        for filling_angle in np.linspace(0.05, np.pi / 2.0, n_filling_angles):

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

            # TODO: compare volumes and areas to Elly's code
            # TODO: use different Kelvin correction factors for each component
            volume = calculate_volume(sol.x, sol.y[0, :], filling_angle)
            area = calculate_area(sol.x, sol.y[0, :], sol.y[1, :])

            print(f'{curvature} {volume} {area}')

    def compute_area(self, r_core: float, condensed_phase_volume: float) -> float:
        pass

    def compute_kappa(self, r_core: float, condensed_phase_volume: float) -> float:
        pass


if __name__ == '__main__':
    cat = CatenoidModel(0.0, 100)
