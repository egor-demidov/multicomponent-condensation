# Kinetic regime condensation of a single component on a sphere (neglected Kelvin effect)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


R_GAS = 8.314
MW = 150.0e-3
R_PART_0 = 14e-9
ALPHA = 0.1
DENSITY = 1200.0
MOLAR_VOLUME = MW / DENSITY
TEMPERATURE = 298.0
P_SAT = 1.6e-3
C_SAT = P_SAT / R_GAS / TEMPERATURE
TIME = 1000.0


def xi_function(t: float) -> float:
    return 1.1


def calculate_volume(n_condensed: float) -> float:
    return 4.0 / 3.0 * np.pi * R_PART_0 ** 3.0 + n_condensed * MOLAR_VOLUME


def calculate_radius(n_condensed: float) -> float:
    return (3.0 * calculate_volume(n_condensed) / 4.0 / np.pi) ** (1.0 / 3.0)


def ode_function(t: float, n_condensed: float) -> float:
    return (np.pi
            * calculate_radius(n_condensed) ** 2.0
            * (8.0 * R_GAS * TEMPERATURE / np.pi / MW) ** (1.0 / 2.0)
            * ALPHA
            * C_SAT
            * (xi_function(t) - 1))


sol = solve_ivp(ode_function, [0, TIME], [0.0], t_eval=np.linspace(0, TIME, 100))

plt.plot(sol.t, sol.y[0, :])
# plt.plot(sol.t, calculate_radius(sol.y[0, :]) * 1e9)
plt.show()

