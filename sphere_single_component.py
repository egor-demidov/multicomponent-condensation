# Kinetic regime condensation of a single component on a sphere (neglected Kelvin effect)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


R_GAS = 8.314
MW = 150.0e-3
R_PART_0 = 14e-9
ALPHA = 0.1
DENSITY = 1200.0
MOLAR_VOLUME = MW / DENSITY
TEMPERATURE = 298.0
P_SAT = 1.6e-3
C_SAT = P_SAT / R_GAS / TEMPERATURE


def xi_function(t: float) -> float:
    return 1.1


def calculate_volume(n_condensed: float) -> float:
    return 4.0 / 3.0 * np.pi * R_PART_0 ** 3.0 + n_condensed * MOLAR_VOLUME


def calculate_radius(n_condensed: float) -> float:
    return (3.0 * calculate_volume(n_condensed) / 4.0 / np.pi) ** (1.0 / 3.0)


def ode_function(n_condensed: float, t: float) -> float:
    return (np.pi
            * calculate_radius(n_condensed) ** 2.0
            * (8.0 * R_GAS * TEMPERATURE / np.pi / MW) ** (1.0 / 2.0)
            * ALPHA
            * C_SAT
            * (xi_function(t) - 1))

t_span = np.linspace(0, 29500, 100)

dt = 29500 / 100
sol_self = np.zeros(len(t_span))
for i in range(1, len(t_span)):
    sol_self[i] = sol_self[i-1] + ode_function(sol_self[i-1], t_span[i]) * dt

sol = odeint(ode_function, 0.0, t_span)

plt.plot(t_span, sol)
plt.plot(t_span, sol_self)
# plt.plot(t_span, ode_function(sol, t_span))

# plt.plot(t_span, calculate_radius(sol) * 1e9)
plt.show()
