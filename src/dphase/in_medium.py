"""
In-medium (thermal) suppression of Z_D <-> f fbar, following Caputo, Park &
Yun (2025). Per polarization i = T, L, the kinetic mixing is replaced by

    eps^2 -> eps_eff^2 = eps^2 m_ZD^4 / [(m_ZD^2 - Re Pi_i)^2 + (omega Gamma_gamma)^2]

and averaged over polarizations with g_T = 2, g_L = 1 (see `eps_eff2_ratio`).

Assumptions:
  - The Z_D coupling is photon-like (m_ZD << m_Z).
  - T_res is below the EW scale (no hypercharge contributions to Pi).

Used by `solver.solve_BE(..., in_medium_eps_eff=True)` for the phase-space
solver and by `cbe` (via `cbe_decay_factors`) for the number-density solver.
"""

import numpy as np
from scipy.interpolate import interp1d

from dphase.constants import ALPHA_EM, COS_THETA_W, FERMIONS, MW
from dphase.cosmology import t_of_T_RD


# Include the longitudinal polarization in eps_eff2_ratio. If False, the
# transverse factor is applied to all 3 polarizations.
INCLUDE_LONGITUDINAL = True


# ---------------------------------------------------------------------------
# Plasma frequency
# ---------------------------------------------------------------------------

def omega_p2_of_T(T):
    """
    Plasma frequency squared from charged SM fermions, with mass thresholds
    (Braaten & Segel, Eq. 4):

        omega_p^2 = (4 alpha/pi) sum_f Nc Q^2
                    * int dp (p^2/E)(1 - v^2/3) (n_f + n_fbar),

    with v = p/E, plus the W+ W- contribution from `_W_term`.

    Parameters
    ----------
    T         : SM temperature [GeV]

    Returns
    -------
    omega_p^2 [GeV^2]
    """
    x = np.linspace(1e-4, 40.0, 800)          # p/T
    total = 0.0
    for fdata in FERMIONS.values():
        # skip neutral fermions
        if fdata["Q2"] == 0.0:
            continue
        mu = fdata["mass_GeV"] / T
        if mu > 60.0:                   # Boltzmann-dead
            continue
        E = np.sqrt(x**2 + mu**2)
        live = E < 60.0
        fd = np.zeros_like(E)
        fd[live] = 1.0 / (np.exp(E[live]) + 1.0)
        integrand = (x**2 / E) * (1.0 - x**2 / (3.0 * E**2)) * 2.0 * fd
        total += fdata["Nc"] * fdata["Q2"] * np.trapezoid(integrand, x)
        # add W+ W- contribution
        total += _W_term(MW / T, x)
    return (4.0 * ALPHA_EM / np.pi) * total * T**2


def _W_term(mu, x):
    """
    W+ W- contribution to omega_p^2, in the units of the fermion sum in
    `omega_p2_of_T` (i.e. per unit-charge fermion, int dp (p^2/E)(1 - v^2/3) 2 f_FD).
    Same integral with Bose-Einstein statistics and Q^2 = 1, and 3
    polarizations instead of 2 spins (hence the overall factor 1.5).

    Parameters
    ----------
    mu : m_W / T
    x  : p / T integration grid, shared with the fermion sum

    Returns
    -------
    W contribution to the bracketed sum in `omega_p2_of_T` (dimensionless)
    """
    if mu > 60.0:                             # Boltzmann-dead
        return 0.0
    E = np.sqrt(x**2 + mu**2)
    live = E < 60.0
    fb = np.zeros_like(E)
    fb[live] = 1.0 / np.expm1(E[live])
    integrand = (x**2 / E) * (1.0 - x**2 / (3.0 * E**2)) * 2.0 * fb
    return 1.5 * np.trapezoid(integrand, x)


def make_omega_p2_interp(T_min, T_max, n=2000):
    """
    Tabulate omega_p^2(T) and return a log-log interpolant

    Parameters
    ----------
    T_min, T_max : temperature range to cover [GeV]; the table is padded by
                   a factor of 2 on each side
    n            : number of log-spaced table points

    Returns
    -------
    callable T -> omega_p^2(T) [GeV^2]
    """
    T_tab = np.geomspace(0.5 * T_min, 2.0 * T_max, n)
    w_tab = np.array([omega_p2_of_T(T) for T in T_tab])
    f = interp1d(np.log(T_tab), np.log(w_tab), kind="linear")
    return lambda T: np.exp(f(np.log(T)))


# ---------------------------------------------------------------------------
# Photon self-energies
# ---------------------------------------------------------------------------

def Pi_T_HTL(omega, k, wp2):
    """
    Transverse photon self-energy, relativistic plasma, leading order (HTL):

        Pi_T(omega, k) = (3/2) wp2 * [ omega^2/k^2
                         - (omega^2 - k^2)/k^2 * (omega/(2k)) * ln((omega+k)/(omega-k)) ]

    Braaten & Segel, PRD 48, 1478 (1993) (relativistic limit).

    Only the REAL part is returned. Im Pi comes from absorption/emission processes
    and enters separately as W in `eps_eff2_ratio`.

    Parameters
    ----------
    omega : Z_D energy [GeV]
    k     : Z_D momentum [GeV], same shape as omega
    wp2   : plasma frequency squared omega_p^2 [GeV^2]

    Returns
    -------
    Re Pi_T [GeV^2], same shape as k
    """
    omega = np.asarray(omega, dtype=float)
    k = np.asarray(k, dtype=float)
    out = np.empty_like(k)

    # for k << omega the two terms in the bracket cancel to O(1) and the
    # formula loses precision; use its Taylor expansion there instead
    small = k < 1e-3 * omega
    out[small] = wp2 * (1.0 + 0.2 * (k[small] / omega[small])**2)

    w, q = omega[~small], k[~small]
    log_term = (w / (2.0 * q)) * np.log((w + q) / (w - q))
    out[~small] = 1.5 * wp2 * (w**2 / q**2 - (w**2 - q**2) / q**2 * log_term)
    return out


def Pi_L_HTL(omega, k, wp2):
    """
    Longitudinal photon self-energy, relativistic plasma, leading order (HTL):

        Pi_L(omega, k) = 3 wp2 * (omega^2 - k^2)/k^2
                         * [ (omega/(2k)) * ln((omega+k)/(omega-k)) - 1 ]

    Caputo, Park & Yun (2025) Eq. (A2) with v_* = 1 (relativistic).

    Only the REAL part is returned, as for `Pi_T_HTL`.

    Parameters
    ----------
    omega : Z_D energy [GeV]
    k     : Z_D momentum [GeV], same shape as omega
    wp2   : plasma frequency squared omega_p^2 [GeV^2]

    Returns
    -------
    Re Pi_L [GeV^2], same shape as k
    """
    omega = np.asarray(omega, dtype=float)
    k = np.asarray(k, dtype=float)
    out = np.empty_like(k)

    # same cancellation issue as Pi_T for k << omega; Taylor expansion there
    small = k < 1e-3 * omega
    out[small] = wp2 * (1.0 - 0.4 * (k[small] / omega[small])**2)

    w, q = omega[~small], k[~small]
    log_term = (w / (2.0 * q)) * np.log((w + q) / (w - q))
    out[~small] = 3.0 * wp2 * (w**2 - q**2) / q**2 * (log_term - 1.0)
    return out


# ---------------------------------------------------------------------------
# Suppression factor for the phase-space solver
# ---------------------------------------------------------------------------

def eps_eff2_ratio(p, T, mZD, W, wp2_func, include_longitudinal=None):
    """
    Ratio of in-medium to vacuum Z_D production rate per momentum mode.
    From the first line of Caputo, Park & Yun (2025) Eq. (5):

        rate ~ sum_i g_i e^{-w/T} Gamma_i |m^2/(m^2 - pi_i)|^2,   i = T, L

    The vacuum rate is the same with pi_i = 0. Expanding the complex
    pi_i = Re pi_i + i Im pi_i,

        F_i = |m^2/(m^2 - pi_i)|^2 = m^4 / [ (m^2 - Re pi_i)^2 + (Im pi_i)^2 ].

    For decays / inverse decays Gamma_T = Gamma_L (polarization-independent
    in vacuum), so the ratio is

        F = (g_T F_T + g_L F_L)/(g_T + g_L) = (2 F_T + F_L)/3.

    Parameters
    ----------
    p                    : Z_D physical momenta [GeV]
    T                    : SM temperature [GeV]
    mZD                  : dark photon mass [GeV]
    W                    : |Im pi| = omega Gamma_gamma [GeV^2], scalar or
                           same shape as p
    wp2_func             : callable T -> omega_p^2 [GeV^2]
    include_longitudinal : include the L mode; None uses the module default
                           INCLUDE_LONGITUDINAL

    Returns
    -------
    F = eps_eff^2 / eps^2, same shape as p
    """
    if include_longitudinal is None:
        include_longitudinal = INCLUDE_LONGITUDINAL
    omega = np.sqrt(p**2 + mZD**2)
    wp2 = wp2_func(T)
    F_T = mZD**4 / ((mZD**2 - Pi_T_HTL(omega, p, wp2))**2 + W**2)
    if not include_longitudinal:
        return F_T
    F_L = mZD**4 / ((mZD**2 - Pi_L_HTL(omega, p, wp2))**2 + W**2)
    return (2.0 * F_T + F_L) / 3.0


def step_averaged_suppression(T_now, T_next, state, G_dec, wp2_func, n_sub=400):
    """
    Time average of eps_eff^2/eps^2 over [T_now, T_next], per bin.
    The resonance is ~1% wide in T, narrower than a typical step, so we need to
    subdivide into sub-steps to actually capture the resonance.

    The width comes from decay rate: omega Gamma_gamma = E_ZD * G / (c_W eps)^2.
    epsilon is the hypercharge kinetic mixing (eps_gamma = c_W eps), and the
    photon damping rate is the Z_D rate divided by eps_gamma^2.

    Parameters
    ----------
    T_now, T_next : endpoints of the solver step [GeV]
    state         : PhaseSpaceState; uses state.species['ZD'], state.grid,
                    state.gstar_func and state.epsilon
    G_dec         : (Np,) decay rate per bin at T_next, the G of the affine
                    decay term S - G f_ZD [GeV]
    wp2_func      : callable T -> omega_p^2 [GeV^2], from `make_omega_p2_interp`
    n_sub         : number of log-spaced sub-steps in T

    Returns
    -------
    (Np,) step-averaged F = eps_eff^2 / eps^2
    """
    mZD = state.species["ZD"].mass_GeV
    p_next = state.grid.p_phys(T_next, gstars_func=state.gstar_func)
    # photon damping rate = Z_D rate / eps_gamma^2, eps_gamma = c_W eps
    W = np.sqrt(p_next**2 + mZD**2) * G_dec / (COS_THETA_W * state.epsilon)**2
    W = np.maximum(W, 1e-30 * mZD**2)

    T_sub = np.geomspace(T_now, T_next, n_sub)
    t_sub = np.array([t_of_T_RD(T, state.gstar_func(T)) for T in T_sub])
    F = np.array([eps_eff2_ratio(state.grid.p_phys(T, gstars_func=state.gstar_func),
                                 T, mZD, W, wp2_func) for T in T_sub])
    return np.trapezoid(F, t_sub, axis=0) / (t_sub[-1] - t_sub[0])


# ---------------------------------------------------------------------------
# cBE (integrated) version of the same correction
# ---------------------------------------------------------------------------
# The fBE multiplies the decay term in each momentum bin by
# F(p, T) = eps_eff^2/eps^2 (eps_eff2_ratio). The cBE assumes a
# Maxwell-Boltzmann Z_D spectrum, so it needs the same F averaged over
# that spectrum:
#     Gamma <m/E>   ->  Gamma <(m/E) F>      (number equation)
#     m Gamma n     ->  m Gamma n <F>        (energy equation, Q_dec)
# with <.> an MB average at the temperature of the Z_D population (T for
# inverse decays from the bath, Th for decays) and F always evaluated with
# the plasma at the SM temperature T. Same F and same resonance width as
# the fBE, so the two solvers differ only in the spectrum F acts on.

class CBEInMedium:
    """
    Thermal averages of F(p, T) for the cBE decay terms.

    W = m_ZD * Gamma_ZD / (c_W eps)^2 is the resonance width, identical to the
    fBE value E_ZD * G / (c_W eps)^2 for MB statistics (G = Gamma_ZD m_ZD / E_ZD).
    Parameters
    ----------
    mZD          : dark photon mass [GeV]
    gamma_ZD     : total Z_D -> f fbar width [GeV]
    epsilon      : kinetic mixing
    T_min, T_max : temperature range for the omega_p^2 table [GeV]
    n_p          : number of momentum points in the MB averages

    Attributes
    ----------
    key   : (mZD, gamma_ZD, epsilon), used by `cbe_decay_factors` to detect
            when the cached object is stale
    """

    def __init__(self, mZD, gamma_ZD, epsilon, T_min=1e-3, T_max=1e3, n_p=4000):
        self.mZD = float(mZD)
        self.key = (float(mZD), float(gamma_ZD), float(epsilon))
        # photon damping rate = Z_D rate / eps_gamma^2, eps_gamma = c_W eps
        self.W = max(mZD * gamma_ZD / (COS_THETA_W * epsilon)**2, 1e-30 * mZD**2)
        self.wp2_func = make_omega_p2_interp(T_min, T_max)
        # momentum grid in units of the typical MB momentum (set per call);
        # fine enough to resolve the resonance, which is narrow in p at fixed T
        self.u = np.geomspace(1e-4, 40.0, n_p)
        self.lnu = np.log(self.u)

    def factors(self, T, Th):
        """
        MB-averaged suppression factors for a Z_D population at Th, with the
        plasma at T. Both are exactly 1 if F = 1, so turning the correction
        off recovers vacuum cBE.

        Parameters
        ----------
        T  : SM temperature, sets the plasma [GeV]
        Th : temperature of the Z_D MB spectrum being averaged over [GeV]

        Returns
        -------
        (r_dec, r_E):
            r_dec = <(m/E) F>_Th / <m/E>_Th   multiplies the existing K1/K2 factor
            r_E   = <F>_Th                    multiplies the Q_dec terms
        """
        mZD = self.mZD
        p_s = np.sqrt(Th**2 + 2.0 * mZD * Th)        # typical MB momentum at Th
        p = self.u * p_s
        E = np.sqrt(p**2 + mZD**2)
        with np.errstate(under="ignore"):
            # MB weight p^2 e^{-(E-m)/Th}, times p for the d ln p measure
            w = p**3 * np.exp(-(E - mZD) / Th)
        F = eps_eff2_ratio(p, T, mZD, self.W, self.wp2_func)
        norm = np.trapezoid(w, self.lnu)
        lorentz = np.trapezoid(w * mZD / E, self.lnu)
        r_dec = np.trapezoid(w * mZD / E * F, self.lnu) / lorentz
        r_E = np.trapezoid(w * F, self.lnu) / norm
        return r_dec, r_E


def cbe_decay_factors(params, T, Th):
    """
    In-medium factors for the cBE decay terms. Builds and caches the
    CBEInMedium object in params["_in_medium"] on first use.

    Parameters
    ----------
    params : cBE parameter dict; uses "in_medium_eps_eff", "mZD",
             "gamma_ZDff" and "epsilon"
    T      : SM temperature [GeV]
    Th     : hidden-sector temperature [GeV]

    Returns
    -------
    (r_dec_gain, r_dec_loss, r_E_gain, r_E_loss); all ones unless
    params["in_medium_eps_eff"] is True. The gain factors are averaged at T
    (inverse decays from the bath), the loss factors at Th (decays).
    """
    if not params.get("in_medium_eps_eff", False):
        return 1.0, 1.0, 1.0, 1.0
    key = (float(params["mZD"]), float(params["gamma_ZDff"]), float(params["epsilon"]))
    flags = (INCLUDE_LONGITUDINAL)
    im = params.get("_in_medium")
    if im is None or im.key != key or getattr(im, "flags", None) != flags:
        im = CBEInMedium(*key)
        im.flags = flags
        params["_in_medium"] = im
    r_dec_gain, r_E_gain = im.factors(T, T)       # inverse decays: Z_D from the bath at T
    r_dec_loss, r_E_loss = im.factors(T, Th)      # decays: Z_D population at Th
    return r_dec_gain, r_dec_loss, r_E_gain, r_E_loss
