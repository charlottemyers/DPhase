"""
Elastic scattering off light SM partners, chi B -> chi B, in the Fokker-Planck
(small momentum transfer) approximation.

Framework
---------
  - |M|^2(s, t): t-channel Z / Z_D exchange, from `model._M2_xB_xB_scalar`
    (B = SM fermion or W).
  - gamma(T): Binder et al. 1706.07433 Eqs.(6)-(7), with the full t-integral,
    valid at all temperatures (no NR assumption on chi).
  - Collision term: Binder et al. Eq.(8), semi-relativistic Fokker-Planck,
    valid for arbitrary f_chi(p), discretized in conservative Chang-Cooper
    flux form as a matrix (`fokker_planck_matrix_cc`).

Assumptions
-----------
  - The SM bath is thermal at T, with exact Fermi-Dirac / Bose-Einstein
    statistics in the gamma integral.
  - Fokker-Planck requires m_B << m_chi, which fails for heavy partners (the
    top, and the W when m_chi ~ m_W). In the phase-space solver those are
    treated with the full operator in `kernels.elastic_sm_heavy`: W is never
    part of the FP gamma there, and heavy fermions must be passed in
    `exclude` to avoid double counting.

Chi-W scattering
----------------
`gamma_single_W` / `gamma_total_W` give the W contribution to gamma(T) in the
same FP approximation. The cBE params["gamma_T"] should include W via
`gamma_total(..., include_W=True)`; phase-space solver should not.
"""

import numpy as np
from functools import lru_cache
from scipy.integrate import quad

from dphase.constants import FERMIONS, MW
from dphase.model import (compute_couplings, M2_xB_t_averaged,
                          xf_coupling_consts, xW_coupling_consts,
                          PROC_FERMION, PROC_W)

# Quadrature tolerances for the gamma(T) double integral.
# T_* : inner t-integral (M2_xB_t_averaged);  W_* : outer omega-integral.
T_EPSREL, T_LIMIT = 1e-3, 100
W_EPSREL, W_LIMIT = 1e-3, 100

N_ANTIFERMIONS = 2.0   # FERMIONS lists one entry per species; f and fbar both scatter
N_W_SPECIES = 2.0      # W+ and W-, with identical |M|^2


# ============================================================
# GAMMA(T), ONE BATH SPECIES -- Binder et al. Eqs.(6)-(7)
# ============================================================

def _gamma_B(T, m_scatter, mB, mZD, proc, consts, stat, g_chi):
    """
    Momentum exchange rate gamma(T) of a scatterer of mass m_scatter off one
    bath species B, Binder Eq.(6) integrated by parts (to avoid
    differentiating the noisy integrand):

        gamma = 1/(48 pi^3 g_chi m^3 T) int_{mB}^{inf} domega g (1 -/+ g)
                    * k_cm^4(omega) * <|M|^2>_t(omega)

    with g the bath occupancy, g(1-g) for Fermi-Dirac and g(1+g) for
    Bose-Einstein. omega is the bath-particle energy in the scatterer rest
    frame, s = m^2 + mB^2 + 2 m omega. The boundary term from the integration
    by parts vanishes since k_cm = 0 at omega = mB.

    Parameters
    ----------
    T         : SM temperature [GeV]
    m_scatter : scatterer mass [GeV]
    mB        : bath-particle mass [GeV]
    mZD       : dark photon mass [GeV], to split the t-integral at the pole
    proc, consts : `model.PROC_FERMION` / `PROC_W` and the matching
                `xf_coupling_consts` / `xW_coupling_consts` array; |M|^2 is
                summed over all spins/polarizations (Nc included)
    stat      : -1 for Fermi-Dirac, +1 for Bose-Einstein
    g_chi     : internal dof of the scatterer

    Returns
    -------
    gamma [GeV]
    """
    def integrand(omega):
        if omega <= mB:
            return 0.0

        s = m_scatter**2 + mB**2 + 2.0 * m_scatter * omega

        # CM momentum squared
        lam = (s - (m_scatter + mB)**2) * (s - (m_scatter - mB)**2)
        if lam <= 0.0:
            return 0.0
        k2_cm = lam / (4.0 * s)

        # dg/domega = -g(1-g)/T (FD) or -g(1+g)/T (BE); after integrating
        # by parts, int g dF/domega = +int g(1 -/+ g)/T F
        exponent = min(omega / T, 500.0)
        if stat > 0.0:
            g = 1.0 / np.expm1(exponent)
            dg = g * (1.0 + g) / T
        else:
            g = 1.0 / (np.exp(exponent) + 1.0)
            dg = g * (1.0 - g) / T

        M2avg = M2_xB_t_averaged(s, m_scatter, mB, mZD, proc, consts,
                                 epsrel=T_EPSREL, limit=T_LIMIT)
        return dg * k2_cm**2 * M2avg

    omega_max = mB + 30.0 * T   # well into the Boltzmann tail
    result, _ = quad(integrand, mB, omega_max,
                     limit=W_LIMIT, epsabs=0.0, epsrel=W_EPSREL)

    prefactor = 1.0 / (48.0 * np.pi**3 * g_chi * m_scatter**3)
    return prefactor * result


def gamma_single_fermion(T, m_scatter, mf, params, mZD, Nc, g_chi=2.0):
    """
    gamma(T) for chi scattering off ONE SM fermion species (not its
    antiparticle), via t-channel Z / Z_D, with Fermi-Dirac statistics.

    Parameters
    ----------
    T         : SM temperature [GeV]
    m_scatter : chi mass [GeV]
    mf        : SM fermion mass [GeV]
    params    : `model.compute_couplings` dict for this fermion
    mZD       : dark photon mass [GeV]
    Nc        : color factor (1 for leptons, 3 for quarks)
    g_chi     : internal dof of chi

    Returns
    -------
    gamma [GeV]
    """
    consts = xf_coupling_consts(params, mZD, Nc)
    return _gamma_B(T, m_scatter, mf, mZD, PROC_FERMION, consts,
                    stat=-1.0, g_chi=g_chi)


def gamma_single_W(T, m_scatter, params, mZD, mW=MW, g_chi=2.0):
    """
    gamma(T) for chi scattering off ONE W species (W+ or W-), via t-channel
    Z / Z_D through the WWZ vertex, with Bose-Einstein statistics.

    The |M|^2 is summed over all spins and polarizations, i.e.
    g_chi * g_W = 6 times the initial-averaged `model.M2_xW_xW` (without
    widths, negligible for t < 0).

    NOTE: Fokker-Planck assumes m_W << m_chi, which fails for m_chi ~ m_W. Use
    this for the cBE gamma(T); the phase-space solver treats W with the full
    operator in `kernels.elastic_sm_heavy`.

    Parameters
    ----------
    T         : SM temperature [GeV]
    m_scatter : chi mass [GeV]
    params    : `model.compute_couplings` dict (any fermion: only the
                fermion-independent mixing angle, masses and dark couplings
                are used)
    mZD       : dark photon mass [GeV]
    mW        : W mass [GeV]
    g_chi     : internal dof of chi

    Returns
    -------
    gamma [GeV]
    """
    return _gamma_B(T, m_scatter, mW, mZD, PROC_W, xW_coupling_consts(params),
                    stat=1.0, g_chi=g_chi)


# ============================================================
# GAMMA(T), SUMMED OVER THE BATH
# ============================================================

@lru_cache(maxsize=None)
def _couplings_cached(epsilon, fermion, mZD, alphaD):
    """compute_couplings is T-independent; memoize it across the T grid."""
    return compute_couplings(epsilon=epsilon, fermion=fermion, mZD=mZD,
                             alphaD=alphaD, fermions_info=FERMIONS)


def gamma_total_W(T, m_scatter, mZD, alphaD, epsilon, mW=MW):
    """
    gamma(T) for chi scattering off W+ and W-, which have identical |M|^2.

    The chi-W couplings are fermion-independent, so the "e" entry of
    `compute_couplings` is used only as a placeholder key.

    Returns
    -------
    gamma [GeV], including both W charges, the three polarizations and
    Bose-Einstein statistics.
    """
    params = _couplings_cached(epsilon, "e", mZD, alphaD)
    return N_W_SPECIES * gamma_single_W(T, m_scatter=m_scatter, params=params,
                                        mZD=mZD, mW=mW)


def gamma_total(T, m_scatter, mZD, alphaD, epsilon,
                fermions=FERMIONS, exclude=(), include_W=False):
    """
    Total chi momentum-exchange rate gamma(T) from SM fermions and
    antifermions, optionally plus W+ and W-.

    Parameters
    ----------
    T         : SM temperature [GeV]
    m_scatter : chi mass [GeV]. The amplitudes describe chi scattering;
                substituting m_ZD does not give Z_D scattering.
    mZD       : dark photon mass [GeV]
    alphaD    : dark fine structure constant
    epsilon   : kinetic mixing
    fermions  : fermion table, default `constants.FERMIONS`. One entry per
                species; the antifermion is added with the same rate.
    exclude   : fermion names to leave out, e.g. ("t",) when the top is
                treated with the full operator in `kernels.elastic_sm_heavy`.
                Does not affect W.
    include_W : add `gamma_total_W`. Default False, as the phase-space solver
                treats W with the full heavy-partner operator; set True for
                the cBE params["gamma_T"], which cannot.

    Returns
    -------
    gamma [GeV]
    """
    total = 0.0
    for name, fdata in fermions.items():
        if name in exclude:
            continue
        params = _couplings_cached(epsilon, name, mZD, alphaD)
        total += N_ANTIFERMIONS * gamma_single_fermion(
            T, m_scatter=m_scatter, mf=fdata["mass_GeV"], params=params,
            mZD=mZD, Nc=fdata["Nc"])

    if include_W:
        total += gamma_total_W(T, m_scatter, mZD, alphaD, epsilon)

    return total


# ============================================================
# PRECOMPUTE ON TEMPERATURE GRID
# ============================================================

def build_xfxf_kernels(T_grid, state, exclude=()):
    """
    Precompute gamma_chi(T) and gamma_ZD(T) for chi f -> chi f on a
    temperature grid, for the phase-space solver (fermions only; W is
    treated by `kernels.elastic_sm_heavy`).

    Parameters
    ----------
    T_grid  : array of temperatures [GeV]
    state   : PhaseSpaceState; requires state.epsilon, state.alphaD,
              state.species['chi'].mass_GeV, state.species['ZD'].mass_GeV
    exclude : fermions to leave out of the FP gamma (e.g. ("t",) when the
              top uses the full heavy-partner operator). Recorded as
              state.gamma_fp_excluded so double counting can be checked.

    Returns
    -------
    gamma_grid : list of dicts {'chi': float, 'ZD': float} per temperature

    After calling, only the chi grid needs assigning:
        state.gamma_grid_chi = np.array([g['chi'] for g in gamma_grid])

    The 'ZD' entries are identically zero: Z_D f -> Z_D f is an eps^4 effect
    and is dropped; `state.gamma_grid_ZD` already defaults to zeros.
    """
    mchi    = state.species["chi"].mass_GeV
    mZD     = state.species["ZD"].mass_GeV
    epsilon = state.epsilon
    alphaD  = state.alphaD

    state.gamma_fp_excluded = tuple(exclude)

    gamma_grid = []
    for T in T_grid:
        T = float(T)
        g_chi = gamma_total(T, mchi, mZD, alphaD, epsilon, exclude=exclude)
        g_ZD  = 0.0  # Z_D f elastic scattering excluded
        gamma_grid.append({'chi': g_chi, 'ZD': g_ZD})

    return gamma_grid


def gamma_from_grid(T, T_grid, gamma_arr):
    """
    Interpolate a precomputed gamma(T) grid, log-linearly in both T and gamma.

    Log-linear because gamma spans many decades across the temperature range,
    so linear interpolation would misestimate it between grid points.

    Returns exactly 0.0 for an identically-zero grid (the normal case for
    gamma_ZD, and the state's default for any unassigned rate grid).

    Parameters
    ----------
    T         : temperature to evaluate at [GeV]
    T_grid    : temperatures the grid was built on [GeV], either order
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
    # np.interp requires increasing x; T grids are usually built high -> low
    order = np.argsort(log_T)
    return float(np.exp(np.interp(np.log(float(T)), log_T[order], log_g[order])))


def gamma_chi_from_grid(T, state):
    """
    chi momentum-exchange rate at T from light SM fermions
    (state.gamma_grid_chi reads as zeros until assigned).
    """
    return gamma_from_grid(T, state.T_grid, state.gamma_grid_chi)


# ============================================================
# CONSERVATIVE (Chang-Cooper) FOKKER-PLANCK
# ------------------------------------------------------------
# Binder Eq. 8, semi-relativistic and valid only in the small-kick regime,
# written in flux form:
#     df/dt = (gamma/2) p^-2 d/dp [ p^2 (E T df/dp + p f) ]
#           = (gamma/2) p^-2 d/dp [ p^2 E T e^{-E/T} d/dp (f e^{E/T}) ]
# discretized with one flux per bin interface,
#     J_{i+1/2} = D [ f_{i+1} e^{(E_{i+1}-E_m)/T} - f_i e^{(E_i-E_m)/T} ],
# taken out of one bin and put into its neighbor, zero flux at the ends.
# Number conservation is exact, e^{-E/T} is an exact fixed point on any grid,
# and the eigenvalues of M are <= 0, so it can be treated implicitly with no
# step cap.
# ============================================================

def fokker_planck_matrix_cc(p, dlogp, T, m, gamma_T):
    """
    Conservative Fokker-Planck operator as a matrix: df/dt = M @ f.

    Parameters
    ----------
    p, dlogp : physical momentum grid at T [GeV] and its log bin widths
    T        : SM temperature [GeV]
    m        : mass of the scattering particle [GeV]
    gamma_T  : momentum exchange rate gamma(T) [GeV], from gamma_from_grid

    Returns
    -------
    M : shape (Np, Np) [GeV]
    """
    p = np.asarray(p, dtype=float)
    E = np.sqrt(p**2 + m**2)
    Np = p.size
    M = np.zeros((Np, Np))
    if gamma_T == 0.0:
        return M

    vol = p**3 * np.asarray(dlogp, dtype=float)      # p^2 dp per bin
    pm = np.sqrt(p[:-1] * p[1:])                      # interface momenta
    Em = np.sqrt(pm**2 + m**2)
    D = 0.5 * gamma_T * pm**2 * Em * T / (p[1:] - p[:-1])
    a = D * np.exp((E[1:] - Em) / T)                  # flux into bin i from i+1, per f_{i+1}
    b = D * np.exp((E[:-1] - Em) / T)                 # flux out of bin i into i+1, per f_i

    i = np.arange(Np - 1)
    M[i, i + 1] += a / vol[i]
    M[i, i] -= b / vol[i]
    M[i + 1, i + 1] -= a / vol[i + 1]
    M[i + 1, i] += b / vol[i + 1]
    return M


def build_fp_matrices(T_grid, state):
    """
    Conservative FP matrices on T_grid, from the gamma(T) grids
    (state.gamma_grid_chi and state.gamma_grid_ZD, both on state.T_grid).
    Build those first.

    Returns
    -------
    {"chi": (NT, Np, Np), "ZD": (NT, Np, Np)}. Assign
        state.FP_matrix_grid = build_fp_matrices(state.T_grid, state)
    for `solve_BE` / `collisions.elastic_collision_rhs`.
    """
    out = {"chi": [], "ZD": []}
    for T in T_grid:
        T = float(T)
        p = state.grid.p_phys(T, gstars_func=state.gstar_func)
        g_chi = gamma_chi_from_grid(T, state)
        g_ZD = gamma_from_grid(T, state.T_grid, state.gamma_grid_ZD)
        out["chi"].append(fokker_planck_matrix_cc(
            p, state.grid.dlogp, T, state.species["chi"].mass_GeV, g_chi))
        out["ZD"].append(fokker_planck_matrix_cc(
            p, state.grid.dlogp, T, state.species["ZD"].mass_GeV, g_ZD))
    return {k: np.array(v) for k, v in out.items()}


def fp_matrices_from_grid_nearest(T, state):
    """Nearest-T conservative FP matrices {"chi": M, "ZD": M}, or None if unset."""
    grids = state.FP_matrix_grid
    if not grids:
        return None
    i_T = int(np.argmin(np.abs(np.asarray(state.T_grid) - float(T))))
    return {k: v[i_T] for k, v in grids.items()}
