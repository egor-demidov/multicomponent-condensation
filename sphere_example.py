import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sphere import Component, MultiComponentSystem


ARO_1_COMPONENT = Component('ARO1', 5.7e-5, 298, 156e3, 150e-3, 1200.0, 0.01)
ARO_2_COMPONENT = Component('ARO2', 1.6e-3, 298, 156e3, 150e-3, 1100.0, 0.01)

r_part_0 = 14e-9
n_tot_0 = 1.0e-21
pressure_atm = 101_325.0
temperature_sat = 298.0
duration = 200.0
saturation_ratio_drop = 20.0

def xi_function(t: float) -> np.ndarray[float]:
    return np.array([1.5, 1.5]) - np.array([1.0, 1.0]) * np.heaviside(t - saturation_ratio_drop, 0.5)

system = MultiComponentSystem(
    [ARO_1_COMPONENT, ARO_2_COMPONENT],
    xi_function,
    r_part_0,
    n_tot_0,
    pressure_atm,
    temperature_sat
)

print(f'Initial coating thickness: {(system.r_part(system.condensed_phase_mole_counts_0) - r_part_0) / 1e-9 :.04f} nm')
print(f'Initial saturation ratios: {", ".join([str(ratio) for ratio in xi_function(0.0)])}')
print(f'Components: {", ".join([component.name for component in system.components])}')

t_span = np.linspace(0, duration, 1000)

sol = solve_ivp(system.ode_function, [0, duration], system.condensed_phase_mole_counts_0, t_eval=t_span, method='LSODA')

mole_fractions = sol.y / np.sum(sol.y, axis=0)

fig, axs = plt.subplots(2, 2, sharex=True)

# fig.set_figwidth(10)
# fig.set_figheight(4)

axs[0, 0].set_title('Moles condensed')
axs[0, 0].plot(sol.t, sol.y[0, :])
axs[0, 0].plot(sol.t, sol.y[1, :])

axs[0, 1].set_title('Mole fractions')
axs[0, 1].plot(sol.t, mole_fractions[0, :])
axs[0, 1].plot(sol.t, mole_fractions[1, :])

axs[1, 0].set_title('Effective radius')
axs[1, 0].plot(sol.t, system.r_part(np.transpose(sol.y)) * 1e9)

axs[1, 1].set_title('Saturation ratio')
axs[1, 1].plot(sol.t, [xi_function(t) for t in sol.t])

plt.tight_layout()
plt.show()
