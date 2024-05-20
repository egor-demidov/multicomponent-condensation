import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sphere import Component, MultiComponentSystem


ARO_1_COMPONENT = Component('ARO1', 5.7e-5, 298, 156e3, 150e-3, 1200.0, 0.01)
ARO_2_COMPONENT = Component('ARO2', 1.6e-3, 298, 156e3, 150e-3, 1100.0, 0.01)

r_part_0 = 14e-9
n_tot_0 = 1.0e-21
pressure_atm = 101_325.0
temperature_sat = 298.0

def xi_function(t: float) -> np.ndarray[float]:
    return [1.2, 0.5]

system = MultiComponentSystem(
    [ARO_1_COMPONENT, ARO_1_COMPONENT],
    xi_function,
    r_part_0,
    n_tot_0,
    pressure_atm,
    temperature_sat
)

print(f'Initial coating thickness: {(system.r_part(system.condensed_phase_mole_counts_0) - r_part_0) / 1e-9 :.04f} nm')
print(f'Initial saturation ratios: {", ".join([str(ratio) for ratio in xi_function(0.0)])}')
print(f'Components: {", ".join([component.name for component in system.components])}')

t_span = np.linspace(0, 1, 100)

sol = odeint(system.ode_function, system.condensed_phase_mole_counts_0, t_span)

plt.plot(t_span, sol[:, 1] / (sol[:, 0] + sol[:, 1]))
plt.show()
