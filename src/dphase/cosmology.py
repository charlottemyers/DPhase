"""
Background cosmology: g_*(T), Hubble rate, entropy density, and the
radiation-dominated time-temperature relation. Everything here describes the Standard Model bath and the expansion it
drives.

Units are natural (GeV) throughout; see `dphase.constants`.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import kv, kve

from dphase.constants import (
    GSTAR_S_DEFAULT,
    LOG_HUGE,
    LOG_TINY,
    MPL,
    VAL_FLOOR,
)


# ---------------------------------------------------------------------------
# Effective degrees of freedom
# ---------------------------------------------------------------------------

_GSTAR_TABLE_NAME = "gstars.dat"


@lru_cache(maxsize=None)
def _load_gstar_table(path=None):
    """
    Load and cache the tabulated g_*(T).

    Cached because `gstar_interp` is called from inside the ODE right-hand side.

    The table ships as package data. `path` overrides it with a two-column
    file of (T [GeV], g_*), ascending in T.
    """
    if path is None:
        path_obj = Path(__file__).resolve().parent / "data" / _GSTAR_TABLE_NAME
    else:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            if not path_obj.exists():
                repo_root = Path(__file__).resolve().parents[2]
                path_obj = (repo_root / path_obj).resolve()

    if not path_obj.exists():
        raise FileNotFoundError(f"Could not find g* data file at: {path_obj}")

    data = np.loadtxt(path_obj)
    return data[:, 0].copy(), data[:, 1].copy()


def gstar_interp(T, path=None):
    """
    Linearly interpolate the tabulated g_*(T).

    Outside the tabulated range NumPy's `interp` clamps to the endpoint
    values, which is the physically sensible behaviour at both ends (a
    constant relativistic content).

    CAVEAT: the shipped table has a single g_* column, so this one function
    supplies both the entropy g_{*s} (used by `s_SM` and the comoving grid)
    and the energy-density g_{*rho} (used by `H_of_T`). Those differ by a few
    percent around the QCD transition and around e+e- annihilation. Splitting
    them requires a two-column table.
    """
    T_data, g_eff = _load_gstar_table(path)
    return np.interp(T, T_data, g_eff)


def gstar_SM(T, t_dep=True):
    """
    SM effective degrees of freedom. With `t_dep=False`, return the constant
    `GSTAR_S_DEFAULT` instead.
    """
    return gstar_interp(T) if t_dep else GSTAR_S_DEFAULT


def dln_gstars_SM_dlnT(T, t_dep=True, delta=1e-2):
    """
    d ln g_{*s} / d ln T by centred finite difference.

    This is the correction factor relating dT/dt to the Hubble rate: entropy
    conservation gives a*T*g_{*s}^{1/3} = const, so T does not simply scale as
    1/a where g_{*s} is changing. Returns 0 when `t_dep=False`.
    """
    if not t_dep:
        return 0.0
    gstar_plus = gstar_interp(T * (1.0 + delta))
    gstar_minus = gstar_interp(T * (1.0 - delta))
    return (np.log(gstar_plus) - np.log(gstar_minus)) / (2.0 * np.log(1.0 + delta))


# ---------------------------------------------------------------------------
# Expansion history
# ---------------------------------------------------------------------------

def H_of_T(T, rho_h=0.0, t_dep=True, include_hs=True):
    """
    Hubble rate from the SM bath plus an optional hidden-sector contribution.

        H = sqrt(8 pi (rho_SM + rho_h) / 3) / MPL,
        rho_SM = (pi^2 / 30) g_*(T) T^4


    Parameters
    ----------
    T          : SM photon temperature [GeV]
    rho_h      : hidden-sector energy density [GeV^4]
    t_dep      : use the tabulated g_*(T) rather than the constant fallback
    include_hs : if False, ignore `rho_h` entirely

    Returns
    -------
    H [GeV]
    """
    rho_SM = (np.pi**2 / 30.0) * gstar_SM(T, t_dep) * T**4
    rho_tot = rho_SM + rho_h if include_hs else rho_SM
    return np.sqrt((8.0 * np.pi / 3.0) * rho_tot) / MPL


def H_RD(T, gstar):
    """
    Hubble rate for a purely radiation-dominated universe, with g_* passed in
    explicitly rather than looked up.

        H = sqrt(8 pi^3 g_* / 90) T^2 / MPL

    Parameters
    ----------
    T     : SM temperature [GeV]
    gstar : effective relativistic degrees of freedom at T

    Returns
    -------
    H [GeV]
    """
    rho = (np.pi**2 / 30.0) * gstar * T**4
    return np.sqrt((8.0 * np.pi / 3.0) * rho) / MPL


def t_of_T_RD(T, gstar):
    """
    Cosmic time at temperature T during radiation domination.

    In an exactly radiation-dominated universe a ~ t^{1/2}, so H = 1/(2t) and

        t(T) = 1 / (2 H(T)).

    The solver uses differences of this quantity to convert a step in T into a
    step in t.


    Parameters
    ----------
    T     : SM temperature [GeV]
    gstar : effective relativistic degrees of freedom at T

    Returns
    -------
    t [GeV^-1]
    """
    return 1.0 / (2.0 * H_RD(T, gstar))


# ---------------------------------------------------------------------------
# Thermodynamic quantities
# ---------------------------------------------------------------------------

def entropy_density(T, gstar):
    """
    SM entropy density with g_{*s} supplied explicitly,

        s = (2 pi^2 / 45) g_{*s} T^3   [GeV^3].
    """
    return (2.0 * np.pi**2 / 45.0) * gstar * T**3


def s_SM(T, t_dep=True):
    """SM entropy density using the tabulated g_{*s}(T). See `entropy_density`."""
    return entropy_density(T, gstar_SM(T, t_dep))


def neq(m, g, T):
    """
    Equilibrium number density for a Maxwell-Boltzmann species,

        n_eq = g m^2 T K_2(m/T) / (2 pi^2).
    """
    return g * (m**2 * T) / (2.0 * np.pi**2) * kv(2, m / T)


def ln_neq(m, g, T):
    """
    log of the equilibrium number density, evaluated without ever forming the
    exponentially small number itself:

        ln n_eq = ln[g m^2 T / (2 pi^2)] - z + ln kve(2, z),   z = m/T

    using the exponentially scaled Bessel kve(2, z) = e^{z} K_2(z).

    K_2(m/T) is what the momentum integral of the Maxwell-Boltzmann
    distribution evaluates to: n_eq = g/(2 pi^2) int dp p^2 exp(-E(p)/T).
    """
    z = m / T
    return (np.log(g) + 2 * np.log(m) + np.log(T) - np.log(2 * np.pi**2)
            - z + np.log(kve(2, z)))


def safe_exp(logx):
    """exp() with the argument clamped to a range that cannot over/underflow."""
    return np.exp(np.clip(logx, LOG_TINY, LOG_HUGE))


def neq_stable(m, g, T):
    """
    `neq` evaluated through log space -- see `ln_neq`. Returns a hard zero
    rather than an underflow warning deep in the Boltzmann tail.
    """
    T = np.maximum(T, VAL_FLOOR)
    return safe_exp(ln_neq(m, g, T))


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def Y(n, T, t_dep=True):
    """Comoving yield Y = n / s, the quantity conserved by expansion alone."""
    return n / s_SM(T, t_dep=t_dep)


def relic_abundance(Y_inf, m):
    """
    Present-day relic density from the frozen-out yield,

        Omega h^2 = 2.742e8 (m / GeV) Y_inf,

    with the numerical factor being s_0 / rho_crit h^-2 in GeV^-1. The
    observed value is Omega_DM h^2 = 0.120.

    `Y_inf` must count every DM degree of freedom whose mass energy survives
    today: for a Dirac fermion with no asymmetry that is Y_chi + Y_chibar =
    2 Y_chi (see `PhaseSpaceState.total_DM_number_density`).
    """
    return 2.742e8 * m * Y_inf




def compute_yields_from_snaps(T_vals, mchi, mA, st, snaps, gchi = 2, gA = 3):
    x_vals = mchi / T_vals
    Y_chi = np.zeros_like(x_vals, dtype=float)
    Y_A = np.zeros_like(x_vals, dtype=float)
    Y_eq_chi = np.zeros_like(x_vals, dtype=float)
    Y_eq_A = np.zeros_like(x_vals, dtype=float)

    n_chi_vals = np.zeros_like(x_vals, dtype=float)
    n_A_vals = np.zeros_like(x_vals, dtype=float)

    for k, T in enumerate(T_vals):
        snap = snaps[k]
        s_T = st.s_SM(T)

        st.f["chi"] = snap["chi"]
        n_chi = st.total_DM_number_density(T)
        Y_chi[k] = n_chi / s_T
        n_chi_vals[k] = n_chi

        st.f["A"] = snap["A"]
        n_A = st.number_density("A", T)
        Y_A[k] = n_A / s_T
        n_A_vals[k] = n_A

        n_chi_eq = 2 * neq_stable(m=mchi, g=gchi, T=T)
        n_A_eq = neq_stable(m=mA, g=gA, T=T)

        Y_eq_chi[k] = n_chi_eq / s_T
        Y_eq_A[k] = n_A_eq / s_T

    return Y_chi, Y_A, Y_eq_chi, Y_eq_A, n_chi_vals, n_A_vals
