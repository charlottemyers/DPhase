"""
Dark photon model: couplings, matrix elements, cross sections and widths
for a Dirac fermion chi coupled to the SM through a kinetically mixed dark
photon Z_D.

Contents (in order):
  - hidden-sector coupling g_D and the Z-Z_D mass mixing (`compute_couplings`)
  - chi chibar <-> Z_D Z_D: sigma(s), |M|^2(s, t) and the angular structure
    used by `kernels.annihilation_hidden`
  - chi chibar -> f fbar and W+ W-, through s-channel Z / Z_D
  - Z_D -> f fbar decay |M|^2 and total width
  - elastic chi Z_D -> chi Z_D, and chi B -> chi B off the SM bath (B = f, W)
  - Gondolo-Gelmini thermal averages <sigma v> for the three annihilation
    channels (used by `cbe`)

Conventions:
  - x = chi, Z_D = dark photon, f = SM fermion.
  - epsilon is the hypercharge kinetic mixing, so the Z_D couples to the EM
    current with strength e epsilon cos(theta_W) for m_ZD << m_Z.
  - Cross sections and <sigma v> are averaged over initial-state dof; the
    g_chi factors are applied at the rate level, not here. Some |M|^2
    helpers are instead spin-summed, as stated in their docstrings.

Unless a source is noted in the docstring, all formulas and conventions follow
from our own calculations.
"""

import numpy as np
from dphase.constants import (FERMIONS, SIN_THETA_W, COS_THETA_W, SIN2_THETA_W,
                              GAMMA_Z, G_SU2, MZ, MW)
from scipy.special import kve
from scipy.integrate import quad
from functools import lru_cache
import numba


######################
######## Hidden sector couplings
######################

def gD_of_alpha(alphaD):
    return np.sqrt(4.0*np.pi*alphaD)


def alpha_of_gD(gD):
    return gD**2 / (4.0 * np.pi)


######################
######## Kinetic mixing
######################


def mixing_params_physical(epsilon, mZD, mZ_phys=MZ):
    """
    Mixing parameters with mZD and mZ_phys interpreted as the physical
    (post-mixing) mass eigenvalues rather than the unmixed inputs.
    """
    if abs(epsilon) >= 1.0:
        raise ValueError("|epsilon| must be < 1")

    eta = epsilon / np.sqrt(1.0 - epsilon**2)
    k = 1.0 + eta**2 * SIN2_THETA_W

    S = mZ_phys**2 + mZD**2
    P = mZ_phys**2 * mZD**2
    disc = S**2 - 4.0 * k * P

    mZ0_sq = (S + np.sqrt(disc)) / (2.0 * k)
    mX_sq = P / mZ0_sq
    delta = np.sqrt(mX_sq / mZ0_sq)

    alpha = 0.5 * np.arctan2(2.0 * eta * SIN_THETA_W,
                             1.0 - eta**2 * SIN2_THETA_W - delta**2)

    return {"eta": eta, "alpha": alpha, "delta": delta,
            "m_Z0": np.sqrt(mZ0_sq), "m_X": np.sqrt(mX_sq),
            "m_Z": mZ_phys, "m_ZD": mZD}


def compute_couplings(
    epsilon,
    fermion,
    mZD,
    alphaD,
    fermions_info=FERMIONS,
    T3=None,
    Y_L=None,
    Y_R=None,
    Gamma_ZD=None,
    Gamma_Z=GAMMA_Z,
):
    """
    Couplings, masses and widths of the Z / Z_D mass eigenstates.

    Gamma_ZD : float, optional
        Total Z_D width. If None (default) it is computed here from
        `gammaZD_allf_mixing(epsilon, m_ZD)`
    """
    if fermions_info is None:
        raise ValueError("You must provide fermions_info.")

    if T3 is not None and Y_L is not None and Y_R is not None:
        fname = "custom"
        mf = None
    elif fermion is not None:
        if fermion not in fermions_info:
            raise ValueError(
                f"Unknown fermion '{fermion}'. Choose from: {list(fermions_info.keys())}"
            )

        fdata = fermions_info[fermion]
        T3 = fdata["T3_L"]
        Y_L = fdata["Y_L"]
        Y_R = fdata["Y_R"]
        mf = fdata["mass_GeV"]
        fname = fermion
    else:
        raise ValueError("Provide either 'fermion' or explicit T3, Y_L, Y_R.")

    # --- mixing ---
    mp = mixing_params_physical(epsilon, mZD)
    eta = mp["eta"]
    alpha = mp["alpha"]
    delta = mp["delta"]

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ctW = COS_THETA_W
    stW = SIN_THETA_W

    prefactor = G_SU2 / (2.0 * ctW)

    # Z couplings
    iso_Z = ca * ctW**2 * T3
    mix_Z = eta * sa + ca * stW

    # Z_D couplings
    iso_ZD = sa * ctW**2 * T3
    mix_ZD = eta * ca - sa * stW

    C_V =  prefactor * (iso_Z - mix_Z * (Y_L + Y_R) * stW)
    C_A = -prefactor * (iso_Z - mix_Z * (Y_L - Y_R) * stW)

    C_V_prime =  prefactor * (iso_ZD + mix_ZD * (Y_L + Y_R) * stW)
    C_A_prime = -prefactor * (iso_ZD + mix_ZD * (Y_L - Y_R) * stW)

    gD = gD_of_alpha(alphaD)
    g_ZDchi = gD * np.cos(alpha)
    g_Zchi = -gD * np.sin(alpha)

    mZ, mZD = mp["m_Z"], mp["m_ZD"]

    # --- Widths -----------------------------------------------------------
    if Gamma_ZD is None:
        if fermions_info is FERMIONS:
            Gamma_ZD = _gammaZD_allf_mixing_cached(float(epsilon), float(mZD))
        else:
            Gamma_ZD = gammaZD_allf_mixing(epsilon, mZD,
                                          fermions_info=fermions_info)

    return {
        "fermion": fname,
        "T3_L": T3,
        "Y_L": Y_L,
        "Y_R": Y_R,
        "mf": mf,

        "epsilon": epsilon,
        "alphaD": alphaD,
        "delta": delta,
        "eta": eta,
        "alpha": alpha,
        "C_A": C_A,
        "C_V": C_V,
        "C_A_prime": C_A_prime,
        "C_V_prime": C_V_prime,
        "m_Z": mZ,
        "m_ZD": mZD,
        "Gamma_Z": Gamma_Z,
        "Gamma_ZD": Gamma_ZD,

        "gD": gD,
        "g_ZDchi": g_ZDchi,
        "g_Zchi": g_Zchi,
    }



#############################################################
###### Hidden sector annihilation
#############################################################

def sigma_s_xxZDZD(g_ZDchi, mchi, mZD, s):
    """
    XX -> Z_D Z_D cross section.
    The result is summed over final spins, and averaged over initial spins.

    Source: Feng, Nath, Wang 2103.15769 eq. D.8
    r -> 0 limit reproduces the QED result sigma*v_rel = pi alpha^2/mchi^2
    (spin-averaged e+e- -> 2gamma).

    Parameters
    ----------
    g_ZDchi : physical chi-chi-Z_D coupling, gD cos(alpha) (see compute_couplings)

    Returns
    -------
    out : ndarray
        s-dependent cross section for XX -> Z_D Z_D.
    """
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    valid = (s > 4*mchi**2) & (s > 4*mZD**2)
    if not np.any(valid):
        return out
    sv = s[valid]

    m2, M2 = mchi**2, mZD**2
    beta_chi2 = sv - 4*m2
    beta_ZD2   = sv - 4*M2

    num_B = (sv - 2*M2) + np.sqrt(beta_ZD2 * beta_chi2)
    den_B = (sv - 2*M2) - np.sqrt(beta_ZD2 * beta_chi2)
    logB = np.log(num_B / den_B)

    term1 = (sv**2 + 4*m2*(sv - 2*M2) + 4*M2**2 - 8*m2**2) \
            / ((sv - 2*M2) * beta_chi2) * logB
    term2 = np.sqrt(beta_ZD2 / beta_chi2) \
            * (m2*sv + 2*M2**2 + 4*m2**2) / ((sv - 4*M2)*m2 + M2**2)
    # g^4/(8 pi s) = 2 pi alpha^2/s: includes the 1/2 identical-final-state (Z_D Z_D) symmetry factor.
    out[valid] = (g_ZDchi**4 / (8*np.pi*sv)) * (term1 - term2)
    return out


@numba.njit(inline='always')
def _sigma_s_xxZDZD_scalar(s, mchi, mZD, g_ZDchi):
    """ Same as sigma_s_xxZDZD but for a single scalar s. Used in numba-compiled code.
    Returns 0 below threshold or whenever the bracketed expression is numerically unsafe.
    """
    m2 = mchi * mchi
    M2 = mZD * mZD
    if s <= 4.0 * m2 or s <= 4.0 * M2:
        return 0.0
    beta_chi2 = s - 4.0 * m2
    beta_ZD2 = s - 4.0 * M2
    sqrt_prod = np.sqrt(beta_ZD2 * beta_chi2)
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
    term2 = np.sqrt(beta_ZD2 / beta_chi2) \
            * (m2 * s + 2.0 * M2 * M2 + 4.0 * m2 * m2) / den_T2
    return (g_ZDchi**4 / (8.0 * np.pi * s)) * (term1 - term2)



#############################################################
###### Angular structure of hidden sector annihilation
#############################################################
# Not a new trace calculation: chi Z_D -> chi Z_D and chi chibar -> Z_D Z_D are the
# same amplitude, so the angular structure is obtained by crossing
#     k -> -k1,  p' -> -p2   ==>   s_C -> t,  t_C -> s,  u_C -> u
# plus the standard (-1) for crossing a single fermion line
#
# Both directions use the same function: |M|^2 summed over spins is invariant
# under time reversal, so |M|^2(chi chibar -> Z_D Z_D) = |M|^2(Z_D Z_D -> chi chibar)
# at the same (s, t, u). And t_of_cos_xxZDZD below is symmetric in the two masses.


def M2sum_xxZDZD(s, t, g_ZDchi, mchi, mZD):
    """
    Spin/polarization-summed |M|^2 for chi chibar -> Z_D Z_D at (s, t).
    No initial state averaging (no 1/4) and no 1/2! identical-final factor,
    (applied in the cross section)
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    m2, M2 = mchi*mchi, mZD*mZD
    u = 2.0*m2 + 2.0*M2 - s - t

    # compton traces with s_C -> t, u_C -> u, t_C -> s
    sC, uC, tC = t, u, s
    Ts  = 8.0*(m2*m2 + 3.0*m2*sC + m2*uC + 4.0*m2*M2 + M2*M2 - sC*uC)
    Tu  = 8.0*(m2*m2 + m2*sC + 3.0*m2*uC + 4.0*m2*M2 + M2*M2 - sC*uC)
    Tsu = 8.0*(4.0*m2*m2 + 2.0*m2*M2 - (m2 + 2.0*M2)*tC)

    xi, eta = sC - m2, uC - m2          # t - mchi^2, u - mchi^2 (both < 0)
    # leading minus sign = fermion-line crossing (see PS 5.68)
    return -(g_ZDchi**4) * (Ts/xi**2 + Tu/eta**2 + 2.0*Tsu/(xi*eta))


def t_of_cos_xxZDZD(s, cos_star, m_in, m_out):
    """
    t at CM scattering angle cos_star;  symmetric in (m_in, m_out)
    """
    mi2, mo2 = m_in*m_in, m_out*m_out
    p_in = np.sqrt(np.maximum(s/4.0 - mi2, 0.0))
    p_out = np.sqrt(np.maximum(s/4.0 - mo2, 0.0))
    return mi2 + mo2 - s/2.0 + 2.0*p_in*p_out*cos_star


def dsigma_dcos_xxZDZD(s, cos_star, g_ZDchi, mchi, mZD):
    """
    dsigma/dcos(theta*) for chi chibar -> Z_D Z_D.

    Includes the 1/2! identical-final factor and 1/4 initial-spin averaging,
    so integrating over cos_star in [-1, 1] gives the full cross section of sigma_s_xxZDZD(s).

    theta_* is the standard CM scattering angle: the angle between the incoming
    chi and the outgoing Z_D in the CM frame (collision axis defined by
    p_chi + p_chibar = 0), NOT relative to the lab boost axis.
    """
    s = np.asarray(s, dtype=float)
    m2, M2 = mchi*mchi, mZD*mZD
    p_chi = np.sqrt(np.maximum(s/4.0 - m2, 0.0))
    p_ZD = np.sqrt(np.maximum(s/4.0 - M2, 0.0))
    t = t_of_cos_xxZDZD(s, cos_star, mchi, mZD)
    M2s = M2sum_xxZDZD(s, t, g_ZDchi, mchi, mZD)
    lam = s*(s - 4.0*m2)                       # lambda(s, m^2, m^2)
    # dt/dcos = 2 p_chi p_ZD ;  dsigma/dt = |M|^2_avg / (16 pi lam)
    return 0.5 * 0.25 * M2s * (2.0*p_chi*p_ZD) / (16.0*np.pi*lam)


@numba.njit(inline='always')
def _dsdcos_shape_xxZDZD(s, cos_star, m_ferm, m_vec):
    """
    Unnormalized angular shape dsigma/dcos for chi chibar <-> Z_D Z_D.

    Couplings, the overall crossing sign and all s-only prefactors are
    dropped: only the cos_star dependence at fixed s matters, because the
    gain cache normalizes these weights.

    m_ferm / m_vec are the FERMION and VECTOR masses
    """
    m2 = m_ferm*m_ferm
    M2 = m_vec*m_vec
    p_f = np.sqrt(max(s/4.0 - m2, 0.0))
    p_v = np.sqrt(max(s/4.0 - M2, 0.0))
    t = m2 + M2 - s/2.0 + 2.0*p_f*p_v*cos_star
    u = 2.0*m2 + 2.0*M2 - s - t
    Ts  = 8.0*(m2*m2 + 3.0*m2*t + m2*u + 4.0*m2*M2 + M2*M2 - t*u)
    Tu  = 8.0*(m2*m2 + m2*t + 3.0*m2*u + 4.0*m2*M2 + M2*M2 - t*u)
    Tsu = 8.0*(4.0*m2*m2 + 2.0*m2*M2 - (m2 + 2.0*M2)*s)
    xi, eta = t - m2, u - m2
    if xi == 0.0 or eta == 0.0:
        return 0.0
    val = -(Ts/(xi*xi) + Tu/(eta*eta) + 2.0*Tsu/(xi*eta))
    return val if val > 0.0 else 0.0


@numba.njit(inline='always')
def _cos_star_axis(s, p3, E3, p4, mu, m_in, P_tot, beta, gamma):
    """
    Cos of the angle between the lab boost axis and the incoming
    particle's CM momentum direction.

    The gain cache deposits using cos(theta_b), the angle to the boost axis
    (fixes E_lab), but |M|^2 depends on cos(theta*), the angle to
    the incoming direction. These 2 frames are related by a rotation through
    this angle, so the deposit weight for a given cos(theta_b) is the
    azimuthal average of dsigma/dcos(theta*) around the boost axis.
    """
    p_star_in_sq = s/4.0 - m_in*m_in
    if p_star_in_sq <= 0.0 or P_tot <= 0.0:
        return 1.0
    p_star_in = np.sqrt(p_star_in_sq)
    # longitudinal (along boost axis) lab component of p3
    p3L = (p3*p3 + p3*p4*mu) / P_tot
    c = gamma * (p3L - beta * E3) / p_star_in
    if c > 1.0:
        return 1.0
    if c < -1.0:
        return -1.0
    return c


@numba.njit(inline='always')
def _fill_aniso_weights(out, s, c_axis, mu_out, w_out, cos_phi, w_phi,
                        m_ferm, m_vec):
    """
    Normalized deposit weights over the mu_out nodes.

    c_axis = cos of the angle between the boost axis and the incoming direction (cos(theta_b))
    mu_out = cos(theta_b) nodes for the outgoing particle relative to the boost axis.
    w_out = weights associated with the mu_out nodes.

    out[j] <- w_out[j] * <dsigma/dcos(theta*)>_phi ,  normalized to sum 1,
    where  cos(theta*) = mu_out[j]*c_axis + sin_b*sin_axis*cos(phi) (gives the rot. about the boost axis)
    """
    n = mu_out.shape[0]
    nphi = cos_phi.shape[0]
    s_axis = np.sqrt(max(1.0 - c_axis*c_axis, 0.0))
    tot = 0.0
    for j in range(n):
        cb = mu_out[j]
        sb = np.sqrt(max(1.0 - cb*cb, 0.0))
        acc = 0.0
        wsum = 0.0
        for k in range(nphi):
            # cs = cos(theta*) for this phi angle, where theta* is the angle to the incoming direction
            cs = cb*c_axis + sb*s_axis*cos_phi[k]
            if cs > 1.0:
                cs = 1.0
            elif cs < -1.0:
                cs = -1.0
            acc += w_phi[k] * _dsdcos_shape_xxZDZD(s, cs, m_ferm, m_vec)
            wsum += w_phi[k]
        val = w_out[j] * (acc / wsum if wsum > 0.0 else 0.0)
        out[j] = val
        tot += val
    if tot > 0.0:
        for j in range(n):
            out[j] /= tot
    else:                   # degenerate: fall back to isotropic
        wtot = 0.0
        for j in range(n):
            wtot += w_out[j]
        for j in range(n):
            out[j] = w_out[j] / wtot



#############################################################
###### Annihilation to SM fermions
#############################################################

def sigma_s_xxff(
    s,
    m_initial,
    m_final,
    params,
    Nc=1.0,
    return_M2=False
):
    """
    chi chibar -> f fbar through s-channel Z and Z_D exchange.
    Result is summed over final spins, averaged over initial spins, and integrated over angles
    """
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

    CV, CA = params["C_V"], params["C_A"]
    CVp, CAp = params["C_V_prime"], params["C_A_prime"]
    mZ, mZD = params["m_Z"], params["m_ZD"]

    GamZ = params.get("Gamma_Z", GAMMA_Z)

    # Widths from compute_couplings. Fall back to computing Gamma_ZD
    # here only for param dicts built without it.
    GamZD = params.get("Gamma_ZD")
    if GamZD is None:
        GamZD = _gammaZD_allf_mixing_cached(float(params["epsilon"]),
                                           float(mZD))

    m_chi = params["mchi"]
    mf = params.get("mf", m_final)

    if not np.isclose(mf, m_final) or not np.isclose(m_chi, m_initial):
        raise ValueError(
            f"mass mismatch: mf={mf} vs m_final={m_final}, "
            f"mchi={m_chi} vs m_initial={m_initial}"
        )

    g_ZDchi = params["g_ZDchi"]
    g_Zchi = params["g_Zchi"]

    # Fermion bilinear combinations
    sp, sm = sv + 2.0 * mf**2, sv - 4.0 * mf**2
    K_V = CV**2 * sp + CA**2 * sm
    K_Vp = CVp**2 * sp + CAp**2 * sm
    K_VpV = CVp * CV * sp + CAp * CA * sm

    # Propagator denominators
    D_Z2 = (sv - mZ**2) ** 2 + (mZ * GamZ) ** 2
    D_ZD2 = (sv - mZD**2) ** 2 + (mZD * GamZD) ** 2
    G = (sv - mZ**2) * (sv - mZD**2) + GamZ * GamZD * mZ * mZD

    term_ZD = g_ZDchi**2 * K_Vp / D_ZD2
    term_Z = g_Zchi**2 * K_V / D_Z2
    term_int = 2.0 * g_ZDchi * g_Zchi * K_VpV * G / (D_Z2 * D_ZD2)

    brackets = term_ZD + term_int + term_Z

    # Spin-averaged matrix element
    Msq = (4.0 / 3.0) * (sv + 2.0 * m_chi**2) * brackets

    if return_M2:
        return Msq

    sigma_out[valid] = (
        (Nc / (16.0 * np.pi * sv))
        * Msq
        * np.sqrt(sv - 4.0 * m_final**2)
        / np.sqrt(sv - 4.0 * m_initial**2)
    )
    return sigma_out


#############################################################
###### Z_D -> f fbar decay
#############################################################

def _chiral_couplings_ZD(T3, Y_L, Y_R, alpha_mix, eta):
    """
    Chiral couplings of the physical Z_D to a SM fermion, from
        g^{Z_D} = (g2/c_W)[ sin(a)(T^3 - Q s_W^2) + eta s_W cos(a) Y ]
    evaluated per chirality with T^3_R = 0.  Returns (g_L, g_R).
    """
    sa, ca = np.sin(alpha_mix), np.cos(alpha_mix)
    pref = G_SU2 / COS_THETA_W
    Q = T3 + Y_L

    g_L = pref * (sa * (T3 - Q * SIN2_THETA_W) + eta * SIN_THETA_W * ca * Y_L)
    g_R = pref * (-sa * (Q * SIN2_THETA_W)     + eta * SIN_THETA_W * ca * Y_R)
    return g_L, g_R


def ZD_ff_decay_m2(epsilon, mZD, fermion, mf=None):
    """
    Polarization-averaged |M|^2 for Z_D -> f fbar, including Z_D-Z mixing.
    Obtained by inverting Curtin et al. Eq. (2.12) under the convention
        Gamma = |M|^2 beta_f / (16 pi mZD).

    The fermion's electric charge is now carried by the couplings.  Nc is still
    applied by the caller, as before.
    """
    fdata = FERMIONS[fermion]
    if mf is None:
        mf = fdata["mass_GeV"]

    mp = mixing_params_physical(epsilon, mZD)
    g_L, g_R = _chiral_couplings_ZD(
        fdata["T3_L"], fdata["Y_L"], fdata["Y_R"], mp["alpha"], mp["eta"]
    )

    sum_sq = g_L**2 + g_R**2
    return (2.0 / 3.0) * (mZD**2 * sum_sq
                          - mf**2 * (-6.0 * g_L * g_R + sum_sq))


def gammaZD_allf_mixing(epsilon, mZD, fermions_info=FERMIONS):
    """
    Total Z_D -> f fbar width including Z_D-Z mixing, summed over kinematically
    accessible SM fermions.

    Built from ZD_ff_decay_m2 via Gamma = Nc |M|^2 beta_f / (16 pi mZD),
    which is equivalent to Curtin et al. Eq. (2.12):
        Gamma_f = Nc beta_f / (24 pi mZD)
                  [ mZD^2 (gL^2+gR^2) - mf^2 (-6 gL gR + gL^2 + gR^2) ].
    """
    total = 0.0
    for fname, fdata in fermions_info.items():
        mf, Nc = fdata["mass_GeV"], fdata["Nc"]
        if mZD <= 2.0 * mf:
            continue
        beta_f = np.sqrt(1.0 - 4.0 * mf**2 / mZD**2)
        Msq = ZD_ff_decay_m2(epsilon=epsilon, mZD=mZD, fermion=fname, mf=mf)
        total += Nc * Msq * beta_f / (16.0 * np.pi * mZD)
    return total


@lru_cache(maxsize=8192)
def _gammaZD_allf_mixing_cached(epsilon, mZD):
    """
    Memoized gammaZD_allf_mixing over the default FERMIONS table.

    Used by compute_couplings, which is typically called once per fermion
    with the same (epsilon, mZD); without this the width sum is redone
    len(FERMIONS) times per model point.
    """
    return gammaZD_allf_mixing(epsilon, mZD, fermions_info=FERMIONS)



#############################################################
###### Processes involving W bosons
#############################################################

def _ww_coupling_sq(x, params, include_widths=True):
    """
    g^2 c_W^2 | g_ZDchi sin(a) / D_ZD(x) + g_Zchi cos(a) / D_Z(x) |^2
    with D_V(x) = x - m_V^2 + i m_V Gamma_V  (x = s for annihilation,
    x = t for scattering).

    Evaluated over a common denominator,
        N = (g_ZDchi sa + g_Zchi ca) x
            - g_ZDchi sa (m_Z^2  - i m_Z  Gamma_Z)
            - g_Zchi  ca (m_ZD^2 - i m_ZD Gamma_ZD),
    because the two propagators cancel at leading order for
    x >> m_Z^2, m_ZD^2, and summing them separately would lose precision.
    """
    x = np.asarray(x, dtype=float)
    sa, ca = np.sin(params["alpha"]), np.cos(params["alpha"])
    g_ZDchi, g_Zchi = params["g_ZDchi"], params["g_Zchi"]
    mZ, mZD = params["m_Z"], params["m_ZD"]

    if include_widths:
        GamZ = params.get("Gamma_Z", GAMMA_Z)
        GamZD = params.get("Gamma_ZD")
        if GamZD is None:
            GamZD = _gammaZD_allf_mixing_cached(float(params["epsilon"]),
                                               float(mZD))
    else:
        GamZ = GamZD = 0.0

    cZD, cZ = g_ZDchi * sa, g_Zchi * ca
    m2Z_c = mZ**2 - 1j * mZ * GamZ
    m2ZD_c = mZD**2 - 1j * mZD * GamZD

    num = (cZD + cZ) * x - cZD * m2Z_c - cZ * m2ZD_c
    den = (x - m2Z_c) * (x - m2ZD_c)

    return G_SU2**2 * COS_THETA_W**2 * np.abs(num / den)**2


def M2_xxWW(s, t, mchi, params, mW=MW):
    """
    chi chibar -> W+ W- through s-channel Z / Z_D.
    |M|^2 averaged over the 4 initial spins, summed over final polarizations:

        |M|^2 = g^2 c_W^2 |...|^2 [ 2 s (s - 4mW^2)(s + 2mchi^2) / mW^2
                + (s^2 - 4 s mW^2 + 12 mW^4)(s (s - 4mW^2) - (u - t)^2) / (8 mW^4) ]

    with t = (p_chi - p_W+)^2 and u = 2mchi^2 + 2mW^2 - s - t.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    m2, mW2 = mchi**2, mW**2
    u = 2.0 * m2 + 2.0 * mW2 - s - t

    kin = (2.0 * s * (s - 4.0 * mW2) * (s + 2.0 * m2) / mW2
           + (s**2 - 4.0 * s * mW2 + 12.0 * mW2**2)
           * (s * (s - 4.0 * mW2) - (u - t)**2) / (8.0 * mW2**2))

    return _ww_coupling_sq(s, params) * kin


def sigma_s_xxWW(s, mchi, params, mW=MW):
    """
    chi chibar -> W+ W-, averaged over initial spins, summed over final
    polarizations:

        sigma = g^2 c_W^2 |...|^2 sqrt(1 - 4mW^2/s)
                / (192 pi s sqrt(1 - 4mchi^2/s))
                x (s - 4mW^2)(s + 2mchi^2)(12 mW^4 + 20 s mW^2 + s^2) / mW^4

    Zero below threshold s = max(4 mchi^2, 4 mW^2).
    """
    s = np.asarray(s, dtype=float)
    sigma_out = np.zeros_like(s, dtype=float)

    tol = 1e-14
    valid = ((s > 4.0 * mchi**2 + tol) & (s > 4.0 * mW**2 + tol)
             & np.isfinite(s))
    if not np.any(valid):
        return sigma_out

    sv = s[valid]
    m2, mW2 = mchi**2, mW**2

    sigma_out[valid] = (
        _ww_coupling_sq(sv, params)
        * np.sqrt(1.0 - 4.0 * mW2 / sv)
        / (192.0 * np.pi * sv * np.sqrt(1.0 - 4.0 * m2 / sv))
        * (sv - 4.0 * mW2) * (sv + 2.0 * m2)
        * (12.0 * mW2**2 + 20.0 * sv * mW2 + sv**2) / mW2**2
    )
    return sigma_out


def M2_xW_xW(s, t, mchi, params, mW=MW, include_widths=False):
    """
    chi W- -> chi W- through t-channel Z / Z_D, obtained from M2_xxWW by
    crossing (s <-> t, overall minus sign for the crossed fermion).
    Averaged over 2 x 3 = 6 initial spins/polarizations, summed over final spins/polarizations.

    The same expression holds for W+ and for chibar.
    """
    s = np.asarray(s, dtype=float)
    t = np.asarray(t, dtype=float)
    m2, mW2 = mchi**2, mW**2
    u = 2.0 * m2 + 2.0 * mW2 - s - t

    kin = (-4.0 * t * (t - 4.0 * mW2) * (t + 2.0 * m2) / mW2
           + (t**2 - 4.0 * t * mW2 + 12.0 * mW2**2)
           * ((u - s)**2 - t * (t - 4.0 * mW2)) / (4.0 * mW2**2))

    return (_ww_coupling_sq(t, params, include_widths=include_widths)
            * kin / 3.0)


#############################################################
###### chi B -> chi B elastic scattering off the SM bath
#############################################################
# B = SM fermion (t-channel Z / Z_D) or W (t-channel Z / Z_D, WWZ vertex).
#   PROC_FERMION : consts = (Nc, then 5 entries per xf_coupling_pairs pair)
#   PROC_W       : consts = (g^2 c_W^2, c_ZD, c_Z, mZ^2, mZD^2)

PROC_W = 0
PROC_FERMION = 1


def xf_coupling_pairs(params, mZD):
    """
    t-independent part of the (B, B') double sum in the chi f |M|^2,
    B, B' in {Z, Z_D}.

    Returns a tuple of (16*g_B*g_B', m_B^2, m_Bp^2, CV.CV+CA.CA, CV.CV-CA.CA)
    in {Z, Z_D} x {Z, Z_D} order. Depends only on the model (not s or t),
    so it is hoisted out of the integrands.
    """
    # Z mixings
    CV, CA = params["C_V"], params["C_A"]

    # Z_D mixing
    CVp, CAp = params["C_V_prime"], params["C_A_prime"]
    g_Zchi, g_ZDchi = params["g_Zchi"], params["g_ZDchi"]

    species = ((CV, CA, MZ, g_Zchi),
               (CVp, CAp, mZD, g_ZDchi))

    pairs = []
    for CV_B, CA_B, m_B, g_chiB in species:
        for CV_Bp, CA_Bp, m_Bp, g_chiBp in species:
            pairs.append((16 * g_chiB * g_chiBp,
                          m_B**2, m_Bp**2,
                          CV_B * CV_Bp + CA_B * CA_Bp,
                          CV_B * CV_Bp - CA_B * CA_Bp))
    return tuple(pairs)


def xf_coupling_consts(params, mZD, Nc):
    """PROC_FERMION consts array: Nc followed by the flattened xf_coupling_pairs."""
    pairs = xf_coupling_pairs(params, mZD)
    return np.array([Nc] + [x for pair in pairs for x in pair],
                    dtype=np.float64)


def xW_coupling_consts(params):
    """
    PROC_W consts array (g^2 c_W^2, c_ZD, c_Z, mZ^2, mZD^2) with
    c_ZD = g_ZDchi sin(a), c_Z = g_Zchi cos(a). Same couplings as
    _ww_coupling_sq, without widths (negligible for t < 0).
    """
    sa, ca = np.sin(params["alpha"]), np.cos(params["alpha"])
    return np.array([G_SU2**2 * COS_THETA_W**2,
                     params["g_ZDchi"] * sa, params["g_Zchi"] * ca,
                     params["m_Z"]**2, params["m_ZD"]**2], dtype=np.float64)


@numba.njit(inline='always')
def _M2_xB_xB_scalar(proc, s, t, m2_chi, m2_B, consts):
    """
    |M|^2 for chi B -> chi B, SUMMED over all spins/polarizations
    (Nc included for fermions). u is fixed by s + t + u = 2 m_chi^2 + 2 m_B^2.

    PROC_W       : g_chi * g_W = 6 times the initial-averaged M2_xW_xW
                   (without widths).
    PROC_FERMION : t-channel Z / Z_D exchange, summed over the
                   xf_coupling_pairs (B, B') double sum.
    """
    u = 2.0 * m2_chi + 2.0 * m2_B - s - t

    if proc == PROC_W:
        g2cW2 = consts[0]
        cZD = consts[1]
        cZ = consts[2]
        mZ2 = consts[3]
        mZD2 = consts[4]
        mW2 = m2_B

        num = (cZD + cZ) * t - cZD * mZ2 - cZ * mZD2
        amp = num / ((t - mZ2) * (t - mZD2))
        coup = g2cW2 * amp * amp

        kin = (-4.0 * t * (t - 4.0 * mW2) * (t + 2.0 * m2_chi) / mW2
               + (t * t - 4.0 * t * mW2 + 12.0 * mW2 * mW2)
               * ((u - s) * (u - s) - t * (t - 4.0 * mW2)) / (4.0 * mW2 * mW2))

        # kin/3 is the 1/6-averaged result, so the full sum is 6 * coup * kin/3
        return 2.0 * coup * kin

    # PROC_FERMION
    Nc = consts[0]
    n_pairs = (consts.shape[0] - 1) // 5
    a1 = s - m2_chi - m2_B
    a2 = u - m2_chi - m2_B
    K1 = 0.5 * (a1 * a1 + a2 * a2) + m2_chi * t - 2.0 * m2_chi * m2_B
    K2 = 2.0 * m2_chi + t

    total = 0.0
    for ip in range(n_pairs):
        j = 1 + 5 * ip
        g16 = consts[j]
        m2_Bv = consts[j + 1]
        m2_Bpv = consts[j + 2]
        C_plus = consts[j + 3]
        C_minus = consts[j + 4]
        pref = g16 / ((t - m2_Bv) * (t - m2_Bpv))
        total += pref * (C_plus * K1 + m2_B * C_minus * K2)
    return Nc * total


def M2_xB_t_averaged(s, mchi, mB, mZD, proc, consts, epsrel=1e-3, limit=100):
    """
    Momentum-transfer averaged |M|^2 for chi B -> chi B, Binder et al.
    1706.07433 Eq.(7):
        <|M|^2>_t = 1/(8*k_cm^4) * int_0^{4*k_cm^2} d(-t) * (-t) * |M|^2(s,t)

    This is the quantity in gamma(T) via Binder Eq.(6). Integration is over
    the physical t range [t_min, 0] with t_min = -(4*k_cm^2), i.e.
    tau = -t in [0, 4*k_cm^2].

    Parameters
    ----------
    s       : [GeV^2]
    mchi    : chi mass [GeV]
    mB      : bath partner mass (mf or mW) [GeV]
    mZD      : dark photon mass [GeV], used to split the quadrature at the
              t-channel pole
    proc, consts : PROC_FERMION / PROC_W and the matching
              xf_coupling_consts / xW_coupling_consts array
    epsrel, limit : inner quadrature settings.

    Returns
    -------
    <|M|^2>_t [dimensionless]
    """
    lam = (s - (mchi + mB)**2) * (s - (mchi - mB)**2)
    if lam <= 0.0:
        return 0.0
    k2_cm = lam / (4.0 * s)
    t_max = 4.0 * k2_cm     # maximum |t|
    m2_chi, m2_B = mchi**2, mB**2

    def integrand(tau):
        # tau = -t, physical range tau in [0, t_max]
        return tau * _M2_xB_xB_scalar(proc, s, -tau, m2_chi, m2_B, consts)

    # break at propagator pole tau = mZD^2 if inside range
    points = [mZD**2] if (0.0 < mZD**2 < t_max) else []

    result, _ = quad(integrand, 0.0, t_max,
                     limit=limit, points=points,
                     epsabs=0.0, epsrel=epsrel)

    return result / (8.0 * k2_cm**2)


#############################################################
###### Thermally averaged <sigma v> for all channels
#############################################################
# All use the Gondolo-Gelmini average with exponentially scaled Bessel
# functions (kve), so they stay finite at large x = mchi/T

def sigma_v_xxZDZD(T, alphaD, mchi, mZD, alpha_mix, s_max_factor=60.0):
    """
    <sigma v> for chi chibar -> Z_D Z_D at temperature T, with sigma(s) from
    sigma_s_xxZDZD. Incoming particles have mass mchi (equal-mass).

    Returns the standard Gondolo-Gelmini thermal average (per-dof-averaged sigma
    convention, same as the kernel). Multiply by g_chi factors at the
    rate level, NOT here.
    """
    T_in = np.asarray(T, dtype=float)
    if T_in.ndim > 0:                            # vectorize over T
        flat = T_in.reshape(-1)
        out = np.array([sigma_v_xxZDZD(t, alphaD, mchi, mZD, alpha_mix,
                                     s_max_factor=s_max_factor)
                        for t in flat])
        return out.reshape(T_in.shape)
    T = float(T_in)

    s_thr = max(4.0*mchi**2, 4.0*mZD**2)
    x = mchi / T

    K2e = kve(2, x)
    if K2e <= 0.0 or not np.isfinite(K2e):
        return 0.0
    norm = 1.0 / (8.0 * mchi**4 * T * K2e*K2e)

    # physical chi-chi-Z_D coupling
    g_ZDchi = gD_of_alpha(alphaD) * np.cos(alpha_mix)

    # substitute u = sqrt(s) for a smoother integrand
    rs_thr = np.sqrt(s_thr)
    rs_max = rs_thr + s_max_factor * T          # exp tail in K1(sqrt(s)/T)

    def integrand(rs):                          # rs = sqrt(s)
        s = rs*rs
        sig = sigma_s_xxZDZD(g_ZDchi, mchi, mZD, np.array([s]))[0]
        if sig <= 0.0 or not np.isfinite(sig):
            return 0.0
        # ds = 2 rs d(rs); kernel = sigma*(s-4mchi^2)*sqrt(s)*K1(sqrt(s)/T)
        return (sig * (s - 4.0*mchi**2) * rs * (2.0*rs)
                * kve(1, rs/T) * np.exp(-(rs - 2.0*mchi) / T))

    val, _ = quad(integrand, rs_thr, rs_max, limit=200)
    return norm * val


def sigma_v_xxff(T, mchi, epsilon, alphaD, fermions_info, mZD,
                 s_max_factor=60.0):
    """
    <sigma v> for chi chibar -> sum_f f fbar at temperature T, using
    sigma_s_xxff for each fermion. Incoming mass = mchi.

    Returns the Gondolo-Gelmini thermal average, summed over fermions.
    """
    T_in = np.asarray(T, dtype=float)
    if T_in.ndim > 0:                            # vectorize over T
        flat = T_in.reshape(-1)
        out = np.array([sigma_v_xxff(t, mchi, epsilon, alphaD, fermions_info,
                                     mZD, s_max_factor=s_max_factor)
                        for t in flat])
        return out.reshape(T_in.shape)
    T = float(T_in)

    x = mchi / T
    K2e = kve(2, x)
    if K2e <= 0.0 or not np.isfinite(K2e):
        return 0.0
    norm = 1.0 / (8.0 * mchi**4 * T * K2e*K2e)

    total = 0.0
    for name, fdata in fermions_info.items():
        mf = fdata["mass_GeV"]
        Nc = fdata["Nc"]
        s_thr = max(4.0*mchi**2, 4.0*mf**2)

        params = compute_couplings(
            epsilon=epsilon, fermion=name, mZD=mZD, alphaD=alphaD,
            fermions_info=fermions_info)
        params["mchi"] = mchi

        rs_thr = np.sqrt(s_thr)
        rs_max = rs_thr + s_max_factor * T

        def integrand(rs, mf=mf, params=params, Nc=Nc):
            s = rs*rs
            sig = sigma_s_xxff(
                np.array([s]), m_initial=mchi, m_final=mf,
                params=params, Nc=Nc)[0]
            if sig <= 0.0 or not np.isfinite(sig):
                return 0.0
            return (sig * (s - 4.0*mchi**2) * rs * (2.0*rs)
                    * kve(1, rs/T) * np.exp(-(rs - 2.0*mchi) / T))

        val, _ = quad(integrand, rs_thr, rs_max, limit=200)
        total += norm * val

    return total


def sigma_v_xxWW(T, mchi, epsilon, alphaD, mZD, mW=MW, s_max_factor=60.0):
    """
    <sigma v> for chi chibar -> W+ W- at temperature T, using
    sigma_s_xxWW. Same Gondolo-Gelmini average and per-dof convention as
    sigma_v_xxff: substitute directly into the number-density equation.
    """
    T_in = np.asarray(T, dtype=float)
    if T_in.ndim > 0:                            # vectorize over T
        flat = T_in.reshape(-1)
        out = np.array([sigma_v_xxWW(t, mchi, epsilon, alphaD, mZD, mW=mW,
                                     s_max_factor=s_max_factor)
                        for t in flat])
        return out.reshape(T_in.shape)
    T = float(T_in)

    x = mchi / T
    K2e = kve(2, x)
    if K2e <= 0.0 or not np.isfinite(K2e):
        return 0.0
    norm = 1.0 / (8.0 * mchi**4 * T * K2e*K2e)

    # WW couplings don't depend on the fermion; 'e' is used as a placeholder.
    params = compute_couplings(epsilon=epsilon, fermion="e", mZD=mZD,
                               alphaD=alphaD)

    rs_thr = 2.0 * mchi
    rs_max = rs_thr + s_max_factor * T

    def integrand(rs):
        s = rs*rs
        sig = sigma_s_xxWW(np.array([s]), mchi, params, mW=mW)[0]
        if sig <= 0.0 or not np.isfinite(sig):
            return 0.0
        return (sig * (s - 4.0*mchi**2) * rs * (2.0*rs)
                * kve(1, rs/T) * np.exp(-(rs - 2.0*mchi) / T))

    points = [mZD] if rs_thr < mZD < rs_max else None
    val, _ = quad(integrand, rs_thr, rs_max, limit=200, points=points)
    return norm * val
