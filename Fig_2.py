import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 2.5,
    "axes.linewidth": 1.2
})

r = 0.02
k_val = 2
s_param = 1.0

N0 = 1.0
delta_c = 0.5
gamma = 1.0
tau = 1.0

alpha = 1.0
beta = 1.0

A_tau2 = 2.0
E_tau2 = 2000.0

A_gamma2 = 2.0
E_gamma2 = 2000.0

delta_g_base = 0.2
delta_t_base = 0.2

A_delta_g = 1.0
E_delta_g = 2000.0

A_delta_t = 1.0
E_delta_t = 2000.0

k_c_base = 1.0
A_c = 1.0
E_c = 2000.0

k_t_base = 1.0
E_t_nonmut = 2000.0

R_const = 8.314

E_t_mut = R_const * np.log(10) / ((1 / 310.0) - (1 / 315.0))
A_g = 1.0
E_g = R_const * np.log(50) / ((1 / 310.0) - (1 / 315.0))

T_low = 310.0
T_high = 315.0
t_shock = 40

T_ref = 310.0

def arrhenius_ref(T, A, E, T_ref=T_ref):
    return A * np.exp(-E / R_const * (1 / T_ref - 1 / T))

def k_cat(sigma):
    return np.exp(-sigma**2 / s_param)

def temperature(t):
    return T_low if t < t_shock else T_high

def k_g_func(t):
    Tcurr = temperature(t)
    return A_g * np.exp(-E_g / R_const * (1 / T_ref - 1 / Tcurr))

def k_t_func_nonmutant(t):
    return k_t_base * np.exp(-E_t_nonmut / R_const * (1 / T_ref - 1 / T_low))

def k_t_func_mutant(t):
    if t < t_shock:
        return k_t_base * np.exp(-E_t_nonmut / R_const * (1 / T_ref - 1 / T_low))
    else:
        Tcurr = temperature(t)
        return k_t_base * np.exp(-E_t_mut / R_const * (1 / T_ref - 1 / Tcurr))

def delta_g_func(t):
    return A_delta_g * np.exp(-E_delta_g / R_const * (1 / T_ref - 1 / T_low))

def delta_t_func(t):
    return A_delta_t * np.exp(-E_delta_t / R_const * (1 / T_ref - 1 / T_low))

def delta_c_func(t):
    return A_delta_t * np.exp(-E_delta_t / R_const * (1 / T_ref - 1 / T_low))

def k_c_func(t):
    return k_c_base * np.exp(-E_c / R_const * (1 / T_ref - 1 / T_low))

def tau2_func(t):
    return A_tau2 * np.exp(-E_tau2 / R_const * (1 / T_ref - 1 / T_low))

def gamma2_func(t):
    return A_gamma2 * np.exp(-E_gamma2 / R_const * (1 / T_ref - 1 / T_low))

def temperature_profile(tarr):
    return np.array([temperature(tt) for tt in tarr])

def ode_system(y, t, k_t_func_topo):
    P, sigma, Np, Nm, G_, T_ = y

    dPdt = r * k_cat(sigma) * (P - (P**2) / k_val)

    dNpdt = N0 * k_cat(sigma) - (delta_c + gamma * G_) * Np
    dNmdt = N0 * k_cat(sigma) - (delta_c + tau * T_) * Nm

    dsigmadt = (
        delta_c * (Np - Nm)
        - gamma2_func(t) * G_ * np.exp(alpha * sigma)
        + tau2_func(t) * T_ * np.exp(-beta * sigma)
    )

    dGdt = k_g_func(t) * k_cat(sigma) - delta_g_func(t) * G_
    dTdt = k_t_func_topo(t) * k_cat(sigma) - delta_t_func(t) * T_

    return [dPdt, dsigmadt, dNpdt, dNmdt, dGdt, dTdt]

def solve_strain(k_t_production_func):
    P0 = 0.5
    sigma0 = 0.0
    Np0 = 0.01
    Nm0 = 0.01
    G0 = 0.2
    T0 = 0.2

    y0 = [P0, sigma0, Np0, Nm0, G0, T0]

    t_span = np.linspace(0, 100, 600)

    sol = odeint(ode_system, y0, t_span, args=(k_t_production_func,))

    return t_span, sol

t_nonmut, sol_nonmut = solve_strain(k_t_func_nonmutant)
P_nm, s_nm, Np_nm, Nm_nm, G_nm, T_nm = sol_nonmut.T

t_mut, sol_mut = solve_strain(k_t_func_mutant)
P_mt, s_mt, Np_mt, Nm_mt, G_mt, T_mt = sol_mut.T

T_vals = temperature_profile(t_nonmut)

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t_nonmut, P_nm, 'b-')
axs[0].plot(t_mut, P_mt, 'r-')
axs[0].axvline(t_shock, color='k', linestyle=':', linewidth=1.8)
axs[0].set_ylabel('OD600')
axs[0].grid(True, alpha=0.3)
axs[0].tick_params(labelbottom=False)

axs[1].plot(t_nonmut, s_nm, 'b-')
axs[1].plot(t_mut, s_mt, 'r-')
axs[1].axvline(t_shock, color='k', linestyle=':', linewidth=1.8)
axs[1].set_ylabel(r'$\sigma$ (turns/bp)')
axs[1].grid(True, alpha=0.3)
axs[1].tick_params(labelbottom=False)

axs[2].plot(t_nonmut, Np_nm, 'b-')
axs[2].plot(t_nonmut, Nm_nm, 'b--')
axs[2].plot(t_mut, Np_mt, 'r-')
axs[2].plot(t_mut, Nm_mt, 'r--')
axs[2].axvline(t_shock, color='k', linestyle=':', linewidth=1.8)
axs[2].set_ylabel(r'$\sigma^{d}_{L_i}$ and $\sigma^{u}_{L_i}$')
axs[2].grid(True, alpha=0.3)
axs[2].tick_params(labelbottom=False)

axs[3].plot(t_nonmut, T_vals, 'k-')
axs[3].axvline(t_shock, color='k', linestyle=':', linewidth=1.8)
axs[3].set_ylabel('Temperature (K)')
axs[3].set_xlabel('Time (minutes)')
axs[3].grid(True, alpha=0.3)

plt.subplots_adjust(
    top=0.98,
    bottom=0.08,
    left=0.12,
    right=0.97,
    hspace=0.12
)

plt.savefig("temperature_shock_supercoiling_dynamics.png", dpi=300)
plt.show()
