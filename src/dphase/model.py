import numpy as np
from dphase.constants import FERMIONS, SIN_THETA_W, COS_THETA_W, SIN2_THETA_W, GAMMA_Z, G_SU2, MZ, E_EM
from scipy.special import kn
from scipy.integrate import quad
import numba


###### Cross sections for the standard dark photon model
###### conventions: x = chi, A = dark photon, f = SM fermion


def gD_of_alpha(alphaD):
    return np.sqrt(4.0*np.pi*alphaD)


def sigma_s_xxAA(alphaD, mchi, mA, s):
    """XX -> AA s-dependent cross section.
    From Aboubrahim, Feng, Nath, Wang 2103.15769 eqn. D.8
    """
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    valid = (s > 4*mchi**2) & (s > 4*mA**2)
    if not np.any(valid):
        return out
    sv = s[valid]

    m2, M2 = mchi**2, mA**2
    beta_chi2 = sv - 4*m2
    beta_A2   = sv - 4*M2

    num_B = (sv - 2*M2) + np.sqrt(beta_A2 * beta_chi2)
    den_B = (sv - 2*M2) - np.sqrt(beta_A2 * beta_chi2)
    logB = np.log(num_B / den_B)

    term1 = (sv**2 + 4*m2*(sv - 2*M2) + 4*M2**2 - 8*m2**2) \
            / ((sv - 2*M2) * beta_chi2) * logB
    term2 = np.sqrt(beta_A2 / beta_chi2) \
            * (m2*sv + 2*M2**2 + 4*m2**2) / ((sv - 4*M2)*m2 + M2**2)
    out[valid] = (2*np.pi*alphaD**2 / sv) * (term1 - term2)
    return out



@numba.njit(inline='always')
def _sigma_s_xxAA_scalar(s, mchi, mA, alphaD):
    """ Same as sigma_s_xxAA but for a single scalar s. Used in numba-compiled code.
    Returns 0 below threshold or whenever the bracketed expression
    is numerically unsafe (avoids NaNs leaking into the cache).
    """
    m2 = mchi * mchi
    M2 = mA * mA
    if s <= 4.0 * m2 or s <= 4.0 * M2:
        return 0.0
    beta_chi2 = s - 4.0 * m2
    beta_A2 = s - 4.0 * M2
    sqrt_prod = np.sqrt(beta_A2 * beta_chi2)
    s_2M2 = s - 2.0 * M2
    num_B = s_2M2 + sqrt_prod
    den_B = s_2M2 - sqrt_prod
    if den_B <= 0.0 or num_B <= 0.0:
        return 0.0
    logB = np.log(num_B / den_B)
    term1 = (s * s + 4.0 * m2 * (s - 2.0 * M2) + 4.0 * M2 * M2 - 8.0 * m2 * m2) \
            / (s_2M2 * beta_chi2) * logB
    den_T2 = (s - 4.0 * M2) * m2 + M2 * M2
    if den_T2 == 0.0:
        return 0.0
    term2 = np.sqrt(beta_A2 / beta_chi2) \
            * (m2 * s + 2.0 * M2 * M2 + 4.0 * m2 * m2) / den_T2
    return (2.0 * np.pi * alphaD * alphaD / s) * (term1 - term2)





def sigma_s_xxff(
    s,
    m_initial,
    m_final,
    Msq_const,
    const_xsec=None,
    params=None,
):
    s = np.asarray(s, dtype=float)
    sigma_out = np.zeros_like(s, dtype=float)

    tol = 1e-14
    valid = (
        (s > 4.0 * m_initial**2 + tol)
        & (s > 4.0 * m_final**2 + tol)
        & np.isfinite(s)
    )

    if not np.any(valid):
        return sigma_out

    sv = s[valid]

    if const_xsec is not None:
        sigma_out[valid] = const_xsec
        return sigma_out

    if params is None:
        sigma_val = (
            (1.0 / (16.0 * np.pi * sv))
            * Msq_const
            * np.sqrt(sv - 4.0 * m_final**2)
            / np.sqrt(sv - 4.0 * m_initial**2)
        )
        sigma_out[valid] = sigma_val
        return sigma_out

    couplings = params["couplings"]
    CA = couplings["C_A"]
    CV = couplings["C_V"]
    CVp = couplings["C_V_prime"]
    CAp = couplings["C_A_prime"]

    m_chi = params["mchi"]
    mf = params["mf"]
    alpha_D = params["alphaD"]

    mZD = couplings["m_ZD"]
    mZ = couplings["m_Z"]

    prop_Z2 = (sv - mZ**2) ** 2 + mZ**2 * GAMMA_Z**2
    prop_ZD2 = (sv - mZD**2) ** 2

    gX = gD_of_alpha(alpha_D)
    gZchi = gX * np.sin(couplings["alpha"])
    gAchi = gX * np.cos(couplings["alpha"])
    K_V = CV**2 * (2.0 * mf**2 + sv) + CA**2 * (sv - 4.0 * mf**2)
    K_Vp = CVp**2 * (2.0 * mf**2 + sv) + CAp**2 * (sv - 4.0 * mf**2)
    K_XV = (
        CVp * CV * (2.0 * mf**2 + sv) + CAp * CA * (sv - 4.0 * mf**2)
    ) * (sv - mZ**2) / (sv - mZD**2)

    prefactor = 0.5* 0.25 * (8.0/3) * (2.0 * m_chi**2 + sv) * (1.0 - 4.0 * m_final**2 / sv)

    term_ZD = gAchi**2 * (K_Vp / prop_ZD2)
    term_Z = gZchi**2 * (K_V / prop_Z2)
    term_int = 2*gAchi*gZchi * (K_XV / prop_Z2)

    Msq = prefactor * (term_ZD +term_int + term_Z)

    sigma_val = (
        (1.0 / (16.0 * np.pi * sv))
        * Msq
        * np.sqrt(sv - 4.0 * m_final**2)
        / np.sqrt(sv - 4.0 * m_initial**2)
    )

    sigma_out[valid] = sigma_val
    return sigma_out






######################
######## Mixing stuff
######################


def mixing_params(epsilon, mA):
    if abs(epsilon) >= 1.0:
        raise ValueError("|epsilon| must be < 1")

    delta = mA / MZ
    eta = epsilon / np.sqrt(1.0 - epsilon**2)

    numerator   = 2.0 * eta * SIN_THETA_W
    denominator = 1.0 - eta**2 * SIN2_THETA_W - delta**2
    alpha = 0.5 * np.arctan2(-numerator, denominator)

    return {"eta": eta, "alpha": alpha, "delta": delta}




def compute_couplings(
    epsilon,
    fermion,
    mA,
    T3=None,
    Y_L=None,
    Y_R=None,
    fermions_info=None,
):
    if fermions_info is None:
        raise ValueError("You must provide fermions_info.")

    if T3 is not None and Y_L is not None and Y_R is not None:
        fname = "custom"
    elif fermion is not None:
        if fermion not in fermions_info:
            raise ValueError(
                f"Unknown fermion '{fermion}'. Choose from: {list(fermions_info.keys())}"
            )

        fdata = fermions_info[fermion]
        T3 = fdata["T3_L"]
        Y_L = fdata["Y_L"]
        Y_R = fdata["Y_R"]
        fname = fermion
    else:
        raise ValueError("Provide either 'fermion' or explicit T3, Y_L, Y_R.")

    # --- mixing ---
    mp = mixing_params(epsilon, mA)
    eta = mp["eta"]
    alpha = mp["alpha"]
    delta = mp["delta"]

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ctW = COS_THETA_W
    stW = SIN_THETA_W

    prefactor = G_SU2 / (2.0 * ctW)

    # Eq. A.2
    # Z couplings (unprimed)
    iso_Z = ca * ctW**2 * T3
    mix_Z = eta * sa - ca * stW

    C_A = -prefactor * (iso_Z + mix_Z * (Y_L - Y_R) * stW)
    C_V =  prefactor * (iso_Z + mix_Z * (Y_L + Y_R) * stW)

    # Z_D couplings (primed)
    iso_ZD = sa * ctW**2 * T3
    mix_ZD = eta * ca + sa * stW

    C_A_prime =  prefactor * (iso_ZD - mix_ZD * (Y_L - Y_R) * stW)
    C_V_prime = -prefactor * (iso_ZD - mix_ZD * (Y_L + Y_R) * stW)

    radicand = (1 + delta**2 + eta**2 * SIN2_THETA_W)**2 - 4 * delta**2

    mZD2 = (MZ**2 / 2) * (
        1 + delta**2 + eta**2 * SIN2_THETA_W
        - np.sign(1 - delta**2) * np.sqrt(radicand)
    )
    mZ2 = (MZ**2 / 2) * (
        1 + delta**2 + eta**2 * SIN2_THETA_W
        + np.sign(1 - delta**2) * np.sqrt(radicand)
    )

    mZD = np.sqrt(mZD2)
    mZ = np.sqrt(mZ2)

    return {
        "fermion": fname,
        "T3_L": T3,
        "Y_L": Y_L,
        "Y_R": Y_R,
        "epsilon": epsilon,
        "delta": delta,
        "eta": eta,
        "alpha": alpha,
        "C_A": C_A,
        "C_V": C_V,
        "C_A_prime": C_A_prime,
        "C_V_prime": C_V_prime,
        "m_Z": mZ,
        "m_ZD": mZD,
    }





def _chiral_couplings_ZD(T3, Y_L, Y_R, alpha_mix, eta):
    """
    Chiral couplings of the physical Z_D to a SM fermion, from
        g^{Z_D} = (g2/c_W)[ sin(a)(Q s_W^2 - T^3) + eta s_W cos(a) Y ]
    evaluated per chirality with T^3_R = 0.  Returns (g_L, g_R).
    """
    sa, ca = np.sin(alpha_mix), np.cos(alpha_mix)
    pref = G_SU2 / COS_THETA_W
    Q = T3 + Y_L          # sign-preserving; Q2 in FERMIONS is squared

    g_L = pref * (sa * (Q * SIN2_THETA_W - T3) + eta * SIN_THETA_W * ca * Y_L)
    g_R = pref * (sa * (Q * SIN2_THETA_W)      + eta * SIN_THETA_W * ca * Y_R)
    return g_L, g_R


def A_ff_decay_m2(epsilon, mA, fermion, mf=None):
    """
    Polarization-averaged |M|^2 for A -> f fbar, including A-Z mixing.
    Obtained by inverting Curtin et al. Eq. (2.12) under the convention
        Gamma = |M|^2 beta_f / (16 pi mA).

    The fermion's electric charge is now carried by the couplings, so the
    external `Qf2 *` factor must be REMOVED at the call site.  Nc is still
    applied by the caller, as before.
    """
    fdata = FERMIONS[fermion]
    if mf is None:
        mf = fdata["mass_GeV"]

    mp = mixing_params(epsilon, mA)
    g_L, g_R = _chiral_couplings_ZD(
        fdata["T3_L"], fdata["Y_L"], fdata["Y_R"], mp["alpha"], mp["eta"]
    )

    sum_sq = g_L**2 + g_R**2
    return (2.0 / 3.0) * (mA**2 * sum_sq
                          - mf**2 * (-6.0 * g_L * g_R + sum_sq))



def xZD_xZD_m2(s, t, alphaD, mchi, mZD, average=True):
    """
    |M|^2 for chi A -> chi A as a function of mandelstam vars (s, t).
    vectorized over s, t where  u is fixed by  s + t + u = 2 mchi^2 + 2 mA^2.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)

    m2 = mchi * mchi
    M2 = mZD * mZD
    gD4 = (4.0 * np.pi * alphaD) ** 2
    Navg = 6.0 if average else 1.0

    u = 2.0 * m2 + 2.0 * M2 - s - t #mandelstam relation

    # spin/pol-summed traces
    Ts  = 8.0 * (m2*m2 + 3.0*m2*s + m2*u + 4.0*m2*M2 + M2*M2 - s*u)
    Tu  = 8.0 * (m2*m2 + m2*s + 3.0*m2*u + 4.0*m2*M2 + M2*M2 - s*u)
    Tsu = 8.0 * (4.0*m2*m2 + 2.0*m2*M2 - (m2 + 2.0*M2)*t)

    xi  = s - m2     # s-channel propagator factor
    eta = u - m2     # u-channel propagator factor (negative throughout physical range)

    return gD4 / Navg * (Ts/xi**2 + Tu/eta**2 + 2.0*Tsu/(xi*eta))


# ---------------------------------------------------------------------------
# Total cross section sigma(s) -- BUILT NUMERICALLY from |M|^2
# ---------------------------------------------------------------------------
def sigma_s_xZD_xZD(s, alphaD, mchi, mZD, average=True, Nt=64):
    """
    sigma(s) = (1 / (16 pi lambda(s,mchi^2,mZD^2)))  *  integral  |M|^2(s,t)  dt t in [t_min, 0]
    built by Gauss-Legendre quadrature over t.  Below threshold s <= (mchi+mZD)^2  the result is 0.

    Parameters
    ----------
    s        : scalar or ndarray
    Nt       : number of Gauss-Legendre nodes for the t integral
    average  : passes through to xZD_xZD_m2
    """
    s_in = np.asarray(s, dtype=float)
    scalar_input = (s_in.ndim == 0)
    s = np.atleast_1d(s_in)
    out = np.zeros_like(s)

    thr2 = (mchi + mZD) ** 2
    ok = s > thr2
    if not np.any(ok):
        return out.item() if scalar_input else out

    sx = s[ok]
    lam = (sx - (mchi + mZD)**2) * (sx - (mchi - mZD)**2)   # > 0 above threshold
    t_min = -lam / sx                                       # < 0
    # Gauss-Legendre nodes/weights on [-1, 1]; map to [t_min, 0]:
    #   t(x) = (t_min/2) (1 - x),   dt/dx = -t_min/2
    xn, wn = np.polynomial.legendre.leggauss(Nt)
    # broadcast: shape (Ns, Nt)
    t_grid = (t_min[:, None] / 2.0) * (1.0 - xn[None, :])
    s_grid = np.broadcast_to(sx[:, None], t_grid.shape)
    M2_grid = xZD_xZD_m2(s_grid, t_grid, alphaD, mchi, mZD, average=average)
    integral = (-t_min / 2.0) * (M2_grid * wn[None, :]).sum(axis=1)
    out[ok] = integral / (16.0 * np.pi * lam)
    return out.item() if scalar_input else out




def M2_chi_f_avg(s, t, mchi, mf, mA, gD, epsilon, Qf):
    """
    Spin-averaged |M|^2 for chi+f->chi+f via t-channel dark photon A.
    Derived from standard QFT trace calculation for t-channel massive
    vector exchange between two Dirac fermion species. Convention:
    averaged over 2 initial chi spins * 2 initial f spins, summed over final spins.

    Param
    ----------
    s, t    : Mandelstam variables [GeV^2], t <= 0 for physical scattering
    mchi    : chi mass [GeV]
    mf      : SM fermion mass [GeV]
    mA      : dark photon mass [GeV]
    gD      : dark gauge coupling
    epsilon : kinetic mixing parameter
    Qf      : fermion electric charge

    Returns
    -------
    |M|^2 [dimensionless in natural units]
    """
    g_f = epsilon * E_EM * Qf
    A   = s - mchi**2 - mf**2          # = s - mchi^2 - mf^2
    # u = 2mchi^2 + 2mf^2 - s - t => mchi^2+mf^2-u = s+t-mchi^2-mf^2 = A+t
    numerator   = A**2 + (A + t)**2 + 2.0*(mchi**2 + mf**2)*t
    propagator  = (t - mA**2)**2
    return 2.0 * gD**2 * g_f**2 * numerator / propagator

def M2_chi_t_averaged(s, mchi, mf, mA, gD, epsilon, Qf):
    """
    momentum-transfer averaged |M|^2, took from Binder et al. Eq.(7):
        <|M|^2>_t = 1/(8*k_cm^4) * int_0^{4*k_cm^2} d(-t) * (-t) * |M|^2(s,t)

    integration is over the physical t range [t_min, 0] with
    t_min = -(4*k_cm^2), i.e. tau = -t in [0, 4*k_cm^2].

    Parameters
    ----------
    s    : GeV^2]
    mchi : chi mass [GeV]
    mf   : SM fermion mass [GeV]
    mA   : dark photon mass [GeV]
    gD   : dark gauge coupling
    epsilon : kinetic mixing
    Qf   : fermion charge

    Returns
    -------
    <|M|^2>_t [dimensionless]
    """
    lam   = (s - (mchi + mf)**2) * (s - (mchi - mf)**2)
    if lam <= 0.0:
        return 0.0
    k2_cm = lam / (4.0 * s)
    t_max = 4.0 * k2_cm     # maximum |t|

    def integrand(tau):
        # tau = -t, physical range tau in [0, t_max]
        return tau * M2_chi_f_avg(s, -tau, mchi, mf, mA, gD, epsilon, Qf)

    # break at propagator pole tau = mA^2 if inside range
    points = [mA**2] if (0.0 < mA**2 < t_max) else []

    result, _ = quad(integrand, 0.0, t_max,
                     limit=150, points=points,
                     epsabs=0.0, epsrel=1e-5)

    return result / (8.0 * k2_cm**2)


################################
################################
# New integrated <sigma v> objects
def sigma_v_xxAA(T, alphaD, mchi, mA, s_max_factor=50.0):
    """
    <sigma v> for chi chibar -> AA at temperature T, with sigma(s) from
    sigma_xx_to_AA_full. Incoming particles have mass mchi (equal-mass).

    Returns the standard Gondolo-Gelmini thermal average (per-dof-averaged sigma
    convention, same as the kernel). Multiply by g_chi factors at the
    rate level, NOT here.
    """
    s_thr = max(4.0*mchi**2, 4.0*mA**2)
    x = mchi / T
    K2 = kn(2, x)
    if K2 <= 0.0 or not np.isfinite(K2):
        return 0.0
    norm = 1.0 / (8.0 * mchi**4 * T * K2*K2)

    # substitute u = sqrt(s) for a smoother integrand; integrate in sqrt(s)
    rs_thr = np.sqrt(s_thr)
    rs_max = rs_thr + s_max_factor * T          # exp tail in K1(sqrt(s)/T)

    def integrand(rs):                          # rs = sqrt(s)
        s = rs*rs
        sig = sigma_s_xxAA(alphaD, mchi, mA, np.array([s]))[0]
        if sig <= 0.0 or not np.isfinite(sig):
            return 0.0
        # ds = 2 rs d(rs); kernel = sigma*(s-4mchi^2)*sqrt(s)*K1(sqrt(s)/T)
        return sig * (s - 4.0*mchi**2) * rs * kn(1, rs/T) * (2.0*rs)

    val, _ = quad(integrand, rs_thr, rs_max, limit=200)
    return norm * val


def sigma_v_xxff(T, mchi, epsilon, alphaD, fermions_info, mA,
                s_max_factor=60.0):
    """
    <sigma v> for chi chibar -> sum_f f fbar at temperature T, using
    sigma_s_xxff_with_Zmixing for each fermion. Incoming mass = mchi.

    Returns the Gondolo-Gelmini thermal average, summed over fermions
    (Nc folded in). Per-dof-averaged convention (same as the sigma),
    substitute DIRECTLY into the number-density equation with no extra g_chi.
    """
    x = mchi / T
    K2 = kn(2, x)
    if K2 <= 0.0 or not np.isfinite(K2):
        return 0.0
    norm = 1.0 / (8.0 * mchi**4 * T * K2*K2)

    total = 0.0
    for name, fdata in fermions_info.items():
        mf = fdata["mass_GeV"]
        Nc = fdata["Nc"]
        s_thr = max(4.0*mchi**2, 4.0*mf**2)
        if 2.0*mchi <= 2.0*mf:          # channel closed (mchi < mf)
            continue

        couplings = compute_couplings(
            epsilon=epsilon, fermion=name, mA=mA,
            fermions_info=fermions_info)
        params = {
            "couplings": {k: couplings[k] for k in
                          ("C_A","C_V","C_A_prime","C_V_prime",
                           "m_ZD","m_Z","alpha")},
            "mchi": mchi, "mf": mf, "alphaD": alphaD, "epsilon": epsilon,
        }

        rs_thr = np.sqrt(s_thr)
        rs_max = rs_thr + s_max_factor * T

        def integrand(rs, mf=mf, params=params):     # bind per-fermion
            s = rs*rs
            sig = sigma_s_xxff(
                np.array([s]), m_initial=mchi, m_final=mf,
                Msq_const=None, params=params)[0]
            if sig <= 0.0 or not np.isfinite(sig):
                return 0.0
            return sig * (s - 4.0*mchi**2) * rs * kn(1, rs/T) * (2.0*rs)

        val, _ = quad(integrand, rs_thr, rs_max, limit=200)
        total += Nc * norm * val

    return total
