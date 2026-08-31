"""
Elastic scattering collision term: chi+f->chi+f and A+f->A+f.

Framework:
  - |M|^2(s,t): derived from QFT trace calculation for
    t-channel massive vector exchange between Dirac fermions.
  - gamma(T): Binder et al. 1706.07433 Eqs.(6)-(7), full t-integral,
    valid across all temperatures (no NR assumption on chi).
  - Collision term: Binder et al. Eq.(8), semi-relativistic Fokker-Planck,
    valid for arbitrary f_chi(p).

Assumptions:
  - SM bath is thermal at T
  - t-channel mediator is the dark photon A (for chi+f)
  - FP valid: m_bath << m_scatter
  - Fermi-Dirac for SM fermions (included exactly in integral)
"""

import numpy as np
from scipy.integrate import quad
from dphase.constants import FERMIONS
from dphase.model import gD_of_alpha, M2_chi_t_averaged


# ============================================================
# GAMMA(T) -- Binder et al. Eqs.(6)-(7)
# ============================================================

def gamma_single_fermion(T, m_scatter, mf, Qf, Nc, gD, epsilon, mA):
    """
    Momentum exchange rate gamma(T) for one DM particle
    (mass m_scatter) scattering off one SM fermion species.
    took from Binder et al. Eq.(6), integrated by parts to avoid
    differentiating the  noisy integrand:

    gamma = 1/(48*pi^3 * g_scatter * m_scatter^3 * T)
                * int_{mf}^{inf} domega * g_FD(omega)*(1-g_FD(omega))
                        * k_cm^4(omega) * <|M|^2>_t(omega) * Nc

    where omega is bath fermion energy,
    s = m_scatter^2 + mf^2 + 2*m_scatter*omega (scatter rest frame),
    k_cm^2 = kallen(s, m_scatter^2, mf^2)/(4s).

    Valid for arbitrary chi temperature (no NR assumption on chi).
    exact FD statistics for SM bath included.

    Params
    ----------
    T         : SM temperature [GeV]
    m_scatter : dark sector particle mass [GeV]
    mf        : SM fermion mass [GeV]
    Qf        : fermion charge
    Nc        : color factor (1 for leptons, 3 for quarks)
    gD        : dark gauge coupling
    epsilon   : kinetic mixing
    mA        : dark photon mass [GeV]

    Returns
    -------
    gamma [GeV]
    """
    # 4 for Dirac fermion (2 spins * particle + antiparticle)
    g_scatter = 4.0

    def integrand(omega):
        if omega < mf:
            return 0.0

        s = m_scatter**2 + mf**2 + 2.0 * m_scatter * omega

        # CM momentum squared
        lam  = (s - (m_scatter + mf)**2) * (s - (m_scatter - mf)**2)
        if lam <= 0.0:
            return 0.0
        k2_cm = lam / (4.0 * s)

        exponent = min(omega / T, 500.0)
        g        = 1.0 / (np.exp(exponent) + 1.0)
        dg       = g * (1.0 - g) / T

        # approximate t-averaged |M|^2 by evaluating at t = -k2_cm
        M2avg = M2_chi_t_averaged(s, m_scatter, mf, mA, gD, epsilon, Qf)
        return dg * k2_cm**2 * M2avg * Nc

    omega_max = mf + 30.0 * T # upper limit for omega integral, well into the Boltzmann tail
    result, _ = quad(integrand, mf, omega_max,
                     limit=150, epsabs=0.0, epsrel=1e-4)

    prefactor = 1.0 / (48.0 * np.pi**3 * g_scatter * m_scatter**3)
    return prefactor * result


def gamma_total(T, m_scatter, mA, gD, epsilon,
                fermions=FERMIONS):
    """
    Total gamma(T) summed over all active SM fermions.

    Parameters
    ----------
    T         : SM temperature [GeV]
    m_scatter : dark sector particle mass (mchi or mA) [GeV]
    mA        : dark photon mass [GeV]
    gD        : dark gauge coupling
    epsilon   : kinetic mixing
    fermions  : FERMIONS dict from CONSTANTS

    Returns
    -------
    gamma [GeV]
    """

    total = 0.0
    for  _, fdata in fermions.items():
        mf       = fdata["mass_GeV"]
        Nc       = fdata["Nc"]
        Qf       = abs(fdata["Q"])

        if T < 0.05 * mf:
            continue
        total += gamma_single_fermion(T, m_scatter, mf, Qf, Nc, gD, epsilon, mA)

    return total


# ============================================================
# PRECOMPUTE ON TEMPERATURE GRID
# ============================================================

def build_xfxf_kernels(T_grid, state):
    """
    Precompute gamma_chi(T) and gamma_A(T) on a temperature grid.

    Parameters
    ----------
    T_grid : array of temperatures [GeV]
    state  : PhaseSpaceState; requires state.epsilon, state.alphaD,
             state.species['chi'].mass_GeV, state.species['A'].mass_GeV

    Returns
    -------
    gamma_grid : list of dicts {'chi': float, 'A': float} per temperature

    After calling, only the chi grid needs assigning:
        state.gamma_grid_chi = np.array([g['chi'] for g in gamma_grid])

    The 'A' entries are identically zero -- A f -> A f is an eps^4 effect and
    is dropped -- and `state.gamma_grid_A` already defaults to zeros, so
    assigning it is unnecessary.
    """

    mchi    = state.species["chi"].mass_GeV
    mA      = state.species["A"].mass_GeV
    epsilon = state.epsilon
    gD      = gD_of_alpha(state.alphaD)

    gamma_grid = []

    for i, T in enumerate(T_grid):
        T = float(T)

        g_chi = gamma_total(T, mchi, mA, gD, epsilon)
        g_A   = 0.0  # A+f elastic scattering excluded: correct rate scales as eps^4, negligible
        gamma_grid.append({'chi': g_chi, 'A': g_A})

    return gamma_grid


def gamma_from_grid(T, T_grid, gamma_arr):
    """
    Interpolate a precomputed gamma(T) grid, log-linearly in both T and gamma.

    Log-linear because gamma spans many decades across the temperature range,
    so linear interpolation would badly misestimate it between grid points.

    Returns exactly 0.0 for an identically-zero grid (the normal case for
    gamma_A, and the state's default for any unassigned rate grid). That exact
    zero matters: it lets `collisions.fokker_planck_dfdt` short-circuit instead
    of evaluating the whole operator against the 1e-300 floor below.

    Parameters
    ----------
    T         : temperature to evaluate at [GeV]
    T_grid    : temperatures the grid was built on [GeV]
    gamma_arr : momentum exchange rate at each of those temperatures [GeV]

    Returns
    -------
    gamma(T) [GeV]
    """
    gamma_arr = np.asarray(gamma_arr, dtype=float)
    if not np.any(gamma_arr):
        return 0.0

    log_T = np.log(np.asarray(T_grid, dtype=float))
    log_g = np.log(np.maximum(gamma_arr, 1e-300))
    return float(np.exp(np.interp(np.log(float(T)), log_T, log_g)))
