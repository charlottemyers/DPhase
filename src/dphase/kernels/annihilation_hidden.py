"""
Collision kernels for chi chibar <-> Z_D Z_D, the number-changing process
internal to the dark sector.

Both directions are built, with the cross section provided by`model.sigma_s_xxZDZD`.
Each direction produces two distinct objects:

  loss  -- how fast the incoming pair is destroyed. Depends only on sigma(s),
           so it collapses to a (Np, Np) matrix K_ij, built by `_annih_loss_kernel`.
  gain  -- where in momentum the products land. Needs the outgoing kinematics
           boosted back from the CM frame, and the (k, l) -> i map is sparse
           and irregular, so it is stored as a flat event list rather than a
           matrix, built by `_build_gain_cache_inner_bilinear`.
"""

import numpy as np
import numba
from numba import jit

from dphase.grid import log_bin_edges
from dphase.model import (_sigma_s_xxZDZD_scalar, sigma_s_xxZDZD, gD_of_alpha,
                          _cos_star_axis, _fill_aniso_weights)

_ZDZDXX_DOF_RATIO = 8.0 / 9.0


def _annih_loss_kernel(p, m_in, m_out, alphaD, Nmu=16, reverse=False):
    r"""
    Loss kernel K(p1, p2) for 2 -> 2 annihilation within the dark sector,

        K_ij = Int d\mu/2  sigma(s_ij(\mu))  v_Moller(s_ij(\mu)),

    where the two incoming legs sit in bins i and j and \mu is the cosine of
    the angle between them. Both directions of chi chibar <-> Z_D Z_D are built
    here; they differ only in which particle is incoming and in one crossing
    factor on sigma.

    Parameters
    ----------
    p       : physical momentum grid [GeV], shape (Np,)
    m_in    : mass of the incoming particles [GeV] -- the pair being destroyed.
              Sets E(p), s, and the Moller velocity.
    m_out   : mass of the outgoing particles [GeV]. Enters only through the
              threshold and, for `reverse=True`, the crossing factor.
    alphaD  : dark coupling strength
    Nmu     : Gauss-Legendre (GL) nodes for the angular integral
    reverse : False for chi chibar -> Z_D Z_D, using sigma_xxZDZD(s) directly.
              True for Z_D Z_D -> chi chibar, obtained by crossing (see below).

    Returns
    -------
    K : shape (Np, Np) [GeV^-2 * dimensionless velocity]

    Notes
    -----
    `sigma_s_xxZDZD` is always called in the chi chibar -> Z_D Z_D direction, i.e.
    with argument order (gD, mchi, mZD), regardless of `reverse`. The
    reverse cross section is then obtained from it by the crossing relation

        sigma_ZDZDxx = (8/9) (lambda_out / lambda_in) sigma_xxZDZD,

    with lambda_in / lambda_out the flux factors of the Z_D Z_D and chi chibar
    states. The 8/9 is the ratio of internal degrees of freedom between the
    two initial states.
    """
    p = np.asarray(p, dtype=float)

    # Incoming legs both have mass m_in, so E, s and the flux factor below all use m_in .
    E = np.sqrt(p**2 + m_in**2)

    mu_nodes, mu_w = np.polynomial.legendre.leggauss(Nmu)
    p1 = p[:, None, None]
    p2 = p[None, :, None]
    E1 = E[:, None, None]
    E2 = E[None, :, None]
    s = 2.0 * m_in**2 + 2.0 * (E1 * E2 - p1 * p2 * mu_nodes[None, None, :])

    # Both the incoming pair and the outgoing pair must be on shell. With
    # forward, it is the incoming-pair bound (s >= 4 m_in^2),
    # reverse, it is the genuine production threshold
    threshold = max(4.0 * m_in**2, 4.0 * m_out**2)
    valid = s >= threshold

    s_v = s[valid]
    E1_full = np.broadcast_to(E1, s.shape)
    E2_full = np.broadcast_to(E2, s.shape)

    mchi, mZD = (m_out, m_in) if reverse else (m_in, m_out)
    sigma = np.zeros_like(s)
    sigma[valid] = sigma_s_xxZDZD(gD_of_alpha(alphaD), mchi, mZD, s_v)

    if reverse:
        # Cross to Z_D Z_D -> chi chibar. lambda(s, m^2, m^2) = s(s - 4m^2) is the
        # equal-mass Kallen function
        lam_in = s_v**2 - 4.0 * s_v * m_in**2       # Z_D Z_D
        lam_out = s_v**2 - 4.0 * s_v * m_out**2     # chi chibar
        sigma[valid] *= _ZDZDXX_DOF_RATIO * (lam_out / lam_in)

    # Moller velocity of the incoming pair,
    v_mol = (np.sqrt(s_v * s_v - 4.0 * s_v * m_in**2)
             / (2.0 * E1_full[valid] * E2_full[valid]))

    integrand = np.zeros_like(s)
    # The 0.5 is the d\mu/2 angular normalization, included here so the outer
    # reduction is just a tensordot against the GL weights.
    integrand[valid] = sigma[valid] * v_mol * 0.5

    return np.tensordot(integrand, mu_w, axes=([2], [0]))


# ===========================================================================
# Loss kernels -- dense (Np, Np) matrices
# ===========================================================================

def build_annihilation_loss_kernels(T_span, state, mchi, mZD, direction):
    """
    Tabulate an annihilation loss kernel on a temperature grid.
    One kernel per temperature, because the comoving grid maps to different
    physical momenta at each T.

    Parameters
    ----------
    T_span    : temperatures [GeV]
    state     : PhaseSpaceState, supplying `gstar_func`, `alphaD` and `Nmu`
    mchi, mZD  : chi and dark photon masses [GeV]
    direction : "xxZDZD" for chi chibar -> Z_D Z_D, "ZDZDxx" for the reverse

    Returns
    -------
    list of (Np, Np) kernels, one per entry in T_span, aligned with it.
    """
    if direction not in ("xxZDZD", "ZDZDxx"):
        raise ValueError(
            f"direction must be 'xxZDZD' or 'ZDZDxx', got {direction!r}")
    reverse = (direction == "ZDZDxx")
    # The incoming pair is chi chibar going forward, Z_D Z_D going in reverse.
    m_in, m_out = (mZD, mchi) if reverse else (mchi, mZD)

    out = []
    for T in T_span:
        p_phys = state.grid.p_phys(float(T), gstars_func=state.gstar_func)
        out.append(_annih_loss_kernel(
            p=p_phys, m_in=m_in, m_out=m_out,
            alphaD=state.alphaD, Nmu=state.Nmu, reverse=reverse,
        ))
    return out


# ===========================================================================
# Gain caches -- sparse event lists
#
# Built by `_build_gain_cache_inner_bilinear`, with nodes on mu_b, the CM
# emission angle to the lab boost axis (which fixes E_lab). The absolute rate
# per (k, l, mu_in) comes from sigma(s); only the split over the mu_b nodes
# depends on the angular model:
#
#   isotropic   -- flat in cos(theta_b), weights wP_out
#   anisotropic -- the true CM distribution. |M|^2 depends on cos(theta*),
#                  the angle to the INCOMING direction, not the boost axis;
#                  the two axes differ by a rotation, so the weight at fixed
#                  cos(theta_b) is the azimuthal average of dsigma/dcos(theta*):
#
#       w(cos th_b) ~ w_out (1/2pi) Int dphi dsigma/dcos(cos th*),
#       cos th* = cos th_b cos th_ax + sin th_b sin th_ax cos phi
#
#                  with cos th_ax from `model._cos_star_axis`. The weights are
#                  normalized to 1, so the total rate is identical to the
#                  isotropic cache.
# ===========================================================================

_ANGULAR_MODES = ("isotropic", "anisotropic")


@numba.njit
def _build_gain_cache_inner_bilinear(
    p_out, E_out, p_in, E_in, PS_in, invW_out,
    mu_in, w_in, mu_out, wP_out,
    m_out, m_in, alphaD, yield_out, reverse,
    anisotropic, w_out, cos_phi, w_phi,
):
    """
    Build the gain (deposition) event list for chi chibar <-> Z_D Z_D.
    This routine walks every quadrature point of

        (incoming bin k) x (incoming bin l) x (mu_in) x (mu_out)

    and, for each, boosts the outgoing particle from the CM frame back to the
    cosmological frame to get its momentum p1. Because p1 rarely lands on
    a grid point, the event's weight is split linearly in energy between the
    two bracketing bins ("bilinear"). Result is a flat event list rather
    than a matrix, since the (k, l) -> i mapping is sparse and irregular.

    Applied later by `apply_gain_cache` as
        df_out[idx_out[e]] += w_base[e] * f_in[k_idx[e]] * f_in[l_idx[e]],
    which is why the weights already absorb the 1/W_out bin measure.

    Both directions are built here; they differ only in the cross section, see
    `reverse` below and the same relation in `_annih_loss_kernel`.

    Parameters
    ----------
    p_out, E_out    : outgoing momentum grid and its energies [GeV]
    p_in, E_in      : incoming momentum grid and energies [GeV]
    PS_in           : incoming phase-space weight p^3 dlogp / (2 pi^2)
    invW_out        : 1 / (outgoing bin measure), pre-inverted
    mu_in, w_in     : Gauss-Legendre nodes/weights for the angle between the
                      two incoming momenta
    mu_out, wP_out  : nodes and weights for the CM emission
                      angle of the outgoing particle, relative to the boost axis
    m_out, m_in     : outgoing and incoming masses [GeV]
    alphaD          : dark coupling strength
    yield_out       : number of the tracked species produced per reaction
                      (2 for the two Z_D from chi chibar -> Z_D Z_D, 1 otherwise)
    reverse         : False for chi chibar -> Z_D Z_D; True for Z_D Z_D -> chi chibar,
                      whose cross section is obtained from the same
                      sigma_xxZDZD by crossing.
    anisotropic     : if True, replace wP_out by the per-s angular weights
                      from `model._fill_aniso_weights` (see section header)
    w_out           : un-halved GL weights for mu_out (anisotropic only)
    cos_phi, w_phi  : azimuthal nodes/weights for the anisotropic average

    Returns
    -------
    (idx_out, k_idx, l_idx, w_base), each shape (N_events,)
    """
    Np_out = p_out.shape[0]
    Np_in = p_in.shape[0]
    Nmu_in = mu_in.shape[0]
    Nmu_out = mu_out.shape[0]

    # Upper bound on the event count: two deposits (i_lo, i_hi) per quadrature point.
    max_events = 2 * Np_in * Np_in * Nmu_in * Nmu_out
    idx_out_buf = np.empty(max_events, dtype=np.int64)
    k_buf = np.empty(max_events, dtype=np.int64)
    l_buf = np.empty(max_events, dtype=np.int64)
    w_buf = np.empty(max_events, dtype=np.float64)
    n = 0

    m_in_sq = m_in * m_in
    m_out_sq = m_out * m_out
    thresh = 4.0 * m_out_sq          # outgoing pair on shell
    gD = np.sqrt(4.0 * np.pi * alphaD)

    # The |M|^2 shape always takes (fermion, vector) masses.
    m_ferm = m_out if reverse else m_in
    m_vec = m_in if reverse else m_out

    # per-s deposit weights over the mu_out nodes; fixed when isotropic
    aw = wP_out.copy()

    # iterate over magnitude of the incoming momenta
    for k in range(Np_in):
        p3 = p_in[k]
        E3 = E_in[k]
        l_start = 0
        for l in range(l_start, Np_in):
            p4 = p_in[l]
            E4 = E_in[l]
            pair_pref =  PS_in[k] * PS_in[l]

            E3E4 = E3 * E4
            p3p4 = p3 * p4
            p3sq_p4sq = p3 * p3 + p4 * p4

            # iterate over incoming angular nodes (mu_in); where mu = cos(angle between p3 and p4, in the lab frame)
            for mi in range(Nmu_in):
                mu = mu_in[mi]
                wmu = w_in[mi]

                s = 2.0 * m_in_sq + 2.0 * (E3E4 - p3p4 * mu)
                if s < thresh:
                    continue

                # Equal-mass Kallen factors, lambda(s, m^2, m^2) = s(s - 4m^2).
                lam_in = s * s - 4.0 * s * m_in_sq
                if lam_in <= 0.0:
                    continue
                vM = np.sqrt(lam_in) / (2.0 * E3 * E4)

                lam_out = s * s - 4.0 * s * m_out_sq
                if lam_out <= 0.0:
                    continue

                # sigma_s_xxZDZD always takes (s, mchi, mZD), so the two masses
                # swap roles between the directions.
                if reverse:
                    sig = _sigma_s_xxZDZD_scalar(s, m_out, m_in, gD)
                else:
                    sig = _sigma_s_xxZDZD_scalar(s, m_in, m_out, gD)
                if not np.isfinite(sig) or sig <= 0.0:
                    continue
                if reverse:
                    # Cross chi chibar -> Z_D Z_D into Z_D Z_D -> chi chibar.
                    sig = _ZDZDXX_DOF_RATIO * (lam_out / lam_in) * sig


                # note: the 0.5 factor for dmu_in/2 normalization is already included in wmu (see gain_cache_common_setup)
                rate_base = wmu * vM * sig * pair_pref

                # --- outgoing kinematics in the CM frame ---
                E_star = 0.5 * np.sqrt(s)
                p_star_sq = E_star * E_star - m_out_sq
                if p_star_sq <= 0.0:
                    continue
                p_star = np.sqrt(p_star_sq)

                # --- boost from CM back to the cosmological frame ---
                E_tot = E3 + E4
                if E_tot <= 0.0:
                    continue
                P_tot_sq = p3sq_p4sq + 2.0 * p3p4 * mu
                if P_tot_sq < 0.0:
                    continue
                P_tot = np.sqrt(P_tot_sq)

                beta = P_tot / E_tot
                gamma = 1.0 / np.sqrt(1.0 - beta * beta)

                if anisotropic:
                    # angle between the boost axis and the incoming direction,
                    # then the normalized dsigma/dcos(theta*) weights per mu_out
                    c_ax = _cos_star_axis(s, p3, E3, p4, mu, m_in,
                                          P_tot, beta, gamma)
                    _fill_aniso_weights(aw, s, c_ax, mu_out, w_out,
                                        cos_phi, w_phi, m_ferm, m_vec)

                for mj in range(Nmu_out):
                    # Boosted energy of one outgoing particle, emitted at cos(theta_b) = mu_out[mj].
                    E1 = gamma * (E_star + beta * p_star * mu_out[mj])
                    p1sq = E1 * E1 - m_out_sq
                    if p1sq <= 0.0:
                        continue
                    p1 = np.sqrt(p1sq)

                    # products landing outside the grid are dropped.
                    if p1 <= p_out[0] or p1 >= p_out[-1]:
                        continue

                    # Bracketing bins, then linear interpolation in energy.
                    # get i_lo and i_hi: the indices of the lower and upper bracketing bins for p1
                    i_lo = np.searchsorted(p_out, p1) - 1
                    if i_lo < 0:
                        i_lo = 0
                    if i_lo > Np_out - 2:
                        i_lo = Np_out - 2
                    i_hi = i_lo + 1

                    # weight split between the nearest 2 bins
                    alpha = (E1 - E_out[i_lo]) / (E_out[i_hi] - E_out[i_lo])
                    if alpha < 0.0:
                        alpha = 0.0
                    elif alpha > 1.0:
                        alpha = 1.0

                    # deposit the event, using yield_out = 2 for the two DPs.
                    # symmetric under mu_b -> -mu_b (t <-> u), so one
                    # deposit with yield_out covers both outgoing particles.
                    rate_event = rate_base * aw[mj] * yield_out

                    # Split the deposit between the two bracketing bins.
                    # invW_out factor converts a rate into a df/dt per bin.

                    # shape of idx_out_buf = (N_events,), where N_events is the total number of events accumulated so far
                    # populate buffers for the lower bracketing bin for the current event
                    # n tracks the total number of events accumulated so far;
                    # idx_out_buf tracks the index to the momentum grid of the outgoing particle for each event
                    idx_out_buf[n] = i_lo
                    k_buf[n] = k
                    l_buf[n] = l
                    w_buf[n] = (1.0 - alpha) * rate_event * invW_out[i_lo]

                    # populate the upper bracketing bin for the current event
                    n += 1
                    idx_out_buf[n] = i_hi
                    k_buf[n] = k
                    l_buf[n] = l
                    w_buf[n] = alpha * rate_event * invW_out[i_hi]
                    n += 1

    return idx_out_buf[:n], k_buf[:n], l_buf[:n], w_buf[:n]


def _aniso_phi_nodes(Nphi):
    """
    Gauss-Legendre nodes in phi over [0, pi] for the anisotropic average.
    The average only involves cos(phi), which is symmetric about pi, so
    integrating over [0, pi] and normalising is exact.
    """
    x, w = np.polynomial.legendre.leggauss(int(Nphi))
    phi = 0.5 * np.pi * (x + 1.0)                 # map [-1,1] -> [0,pi]
    return (np.ascontiguousarray(np.cos(phi), dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64))


def _build_gain_cache(
    p_out, p_in, dlogp_in,
    m_chi, m_ZD, alphaD,
    Nmu_in, Nmu_out, yield_out, reverse,
    angular, Nphi,
):
    """
    Shared wrapper behind `build_gain_cache_xxZDZD` / `build_gain_cache_ZDZDxx`:
    sets up the quadrature arrays, runs the event builder for the requested
    angular model, and packages the result.
    """
    if angular not in _ANGULAR_MODES:
        raise ValueError(
            f"angular must be one of {_ANGULAR_MODES}, got {angular!r}")

    # The incoming pair is chi chibar going forward, Z_D Z_D going in reverse.
    m_in, m_out = (m_ZD, m_chi) if reverse else (m_chi, m_ZD)

    p_out, p_in, E_in, PS_in, invW_out, mu_in_n, w_in_n, mu_out_n, wP_out = \
        _gain_cache_common_setup(p_out, p_in, dlogp_in, m_in, Nmu_in, Nmu_out)
    E_out = np.ascontiguousarray(np.sqrt(p_out**2 + m_out**2), dtype=np.float64)

    cos_phi, w_phi = _aniso_phi_nodes(Nphi)
    idx_out, k_idx, l_idx, w_base = _build_gain_cache_inner_bilinear(
        p_out, E_out, p_in, E_in, PS_in, invW_out,
        mu_in_n, w_in_n, mu_out_n, wP_out,
        float(m_out), float(m_in), float(alphaD),
        float(yield_out), bool(reverse),
        angular == "anisotropic", 2.0 * wP_out, cos_phi, w_phi,
    )

    if idx_out.size == 0:
        idx_out = np.zeros(0, dtype=np.int64)
        k_idx = np.zeros(0, dtype=np.int64)
        l_idx = np.zeros(0, dtype=np.int64)
        w_base = np.zeros(0, dtype=float)

    return dict(
        idx_out=idx_out, k_idx=k_idx, l_idx=l_idx, w_base=w_base,
        p_out=p_out, p_in=p_in, angular=angular,
    )


def build_gain_cache_xxZDZD(
    p_out, p_in, dlogp_in,
    m_chi, m_ZD, alphaD,
    Nmu_in=4, Nmu_out=4,
    yield_out=1.0,
    angular="isotropic", Nphi=8,
):
    """
    Gain cache for the Z_D produced by chi chibar -> Z_D Z_D.

    Wrapper: sets up the quadrature arrays, runs the event builder in the
    forward direction, and packages the result. All the physics is in
    `_build_gain_cache_inner_bilinear`.

    Parameters
    ----------
    p_out    : momentum grid the products are deposited onto [GeV]
    p_in     : momentum grid the incoming pair is read from [GeV]
    dlogp_in : log-momentum bin widths for `p_in`
    m_chi, m_ZD : masses [GeV]
    alphaD   : dark coupling strength
    Nmu_in, Nmu_out : Gauss-Legendre nodes for the incoming-pair angle and the
               CM emission angle
    yield_out : products of the tracked species per reaction -- 2 here, since
               chi chibar -> Z_D Z_D makes two dark photons.
    angular  : CM emission model, "isotropic" or "anisotropic"
    Nphi     : azimuthal nodes; ignored for "isotropic"

    Returns
    -------
    dict with keys idx_out, k_idx, l_idx, w_base (all shape (N_events,)) plus
    p_out, p_in and angular.
    Consume it with `apply_gain_cache`.
    """
    return _build_gain_cache(p_out, p_in, dlogp_in, m_chi, m_ZD, alphaD,
                             Nmu_in, Nmu_out, yield_out, False,
                             angular, Nphi)


def build_gain_cache_ZDZDxx(
    p_out, p_in, dlogp_in,
    m_chi, m_ZD, alphaD,
    Nmu_in=4, Nmu_out=4,
    yield_out=1.0,
    angular="isotropic", Nphi=8,
):
    """
    Gain cache for the chi produced by Z_D Z_D -> chi chibar.

    The reverse counterpart of `build_gain_cache_xxZDZD`, and identical to it
    except that the incoming pair is now the Z_D (so `m_ZD` sets the energies)
    and the cross section is crossed from sigma_xxZDZD. Same return shape. The
    same angular shape serves both directions: |M|^2 summed over spins is
    invariant under time reversal.

    `yield_out` is 1 here: one chi per reaction, with the chibar tracked as the
    same population (see `PhaseSpaceState.total_DM_number_density`).
    """
    return _build_gain_cache(p_out, p_in, dlogp_in, m_chi, m_ZD, alphaD,
                             Nmu_in, Nmu_out, yield_out, True,
                             angular, Nphi)


def build_fixed_grid_gain_caches_xxZDZD(
    p_cache,
    *,
    m_chi, m_ZD, alphaD,
    yield_out_chi=1.0,
    yield_out_ZD=2.0,
    Nmu_in=4, Nmu_out=4,
    angular="isotropic", Nphi=8,
):
    """
    Build both gain caches for one momentum grid, in one call.

    Single temperature needs two caches, because each species gains from the
    other one annihilating:

        cache_chi_from_ZDZD : chi deposited by Z_D Z_D -> chi chibar   (reverse)
        cache_ZD_from_xx   : Z_D deposited by chi chibar -> Z_D Z_D   (forward)

    Parameters
    ----------
    p_cache : shared momentum grid, used as both p_in and p_out [GeV]
    m_chi, m_ZD, alphaD : model parameters
    yield_out_chi, yield_out_ZD : products per reaction, 1 and 2 respectively.
    Nmu_in, Nmu_out, angular, Nphi : passed through to the builders

    Returns
    -------
    dict holding both caches, the grid, and the parameters they were built with.
    """
    p_cache = np.asarray(p_cache, float)
    dlogp_cache = log_bin_edges(p_cache)
    common = dict(p_out=p_cache, p_in=p_cache, dlogp_in=dlogp_cache,
                  m_chi=m_chi, m_ZD=m_ZD, alphaD=alphaD,
                  Nmu_in=Nmu_in, Nmu_out=Nmu_out,
                  angular=angular, Nphi=Nphi)

    # gain of chi
    cache_chi_from_ZDZD = build_gain_cache_ZDZDxx(yield_out=yield_out_chi, **common)

    # gain of Z_D
    cache_ZD_from_xx = build_gain_cache_xxZDZD(yield_out=yield_out_ZD, **common)

    return dict(
        p_cache=p_cache,
        dlogp_cache=dlogp_cache,
        cache_chi_from_ZDZD=cache_chi_from_ZDZD,
        cache_ZD_from_xx=cache_ZD_from_xx,
        _mchi=m_chi, _mZD=m_ZD,
        _alphaD=alphaD,
        _Nmu_in=Nmu_in, _Nmu_out=Nmu_out,
        _angular=angular, _Nphi=Nphi,
    )


def build_annihilation_gain_caches(T_span, state, angular="anisotropic", Nphi=8):
    """
    Tabulate both gain caches on a temperature grid.

    Mirrors `build_annihilation_loss_kernels`: one entry per temperature,
    because the comoving grid maps to different physical momenta at each T.

    Parameters
    ----------
    T_span  : temperatures [GeV]
    state   : PhaseSpaceState, supplying grid, gstar_func, species, alphaD, Nmu
    angular : CM emission model:
              "isotropic"   (default) flat in cos(theta_b) about the boost axis
              "anisotropic" true CM angular distribution, same total rate.
    Nphi    : azimuthal nodes; ignored for "isotropic"

    Returns
    -------
    list of dicts as returned by `build_fixed_grid_gain_caches_xxZDZD`, aligned
    with T_span. Assign to `state.gain_caches_xxZDZD`.

    """
    out = []
    for T in T_span:
        p_phys = state.grid.p_phys(float(T), gstars_func=state.gstar_func)
        cache_T = build_fixed_grid_gain_caches_xxZDZD(
            p_phys,
            m_chi=state.species["chi"].mass_GeV,
            m_ZD=state.species["ZD"].mass_GeV,
            alphaD=state.alphaD,
            Nmu_in=state.Nmu,
            Nmu_out=state.Nmu,
            yield_out_chi=1.0,
            yield_out_ZD=2.0,
            angular=angular,
            Nphi=Nphi,
        )
        out.append(cache_T)
    return out


def _gain_cache_common_setup(p_out, p_in, dlogp_in, m_in, Nmu_in, Nmu_out):
    """
    Build the quadrature arrays and phase-space weights both gain builders
    need, and force everything C-contiguous float64 for numba.

    PS_in is the incoming phase-space measure p^3 dlogp / (2 pi^2); invW_out is
    the reciprocal of the same measure on the outgoing grid, pre-inverted
    because the event loop needs it per deposit.
    """
    p_out = np.ascontiguousarray(p_out, dtype=np.float64)
    p_in = np.ascontiguousarray(p_in, dtype=np.float64)
    dlogp_in = np.ascontiguousarray(dlogp_in, dtype=np.float64)

    E_in = np.sqrt(p_in ** 2 + m_in ** 2)
    PS_in = (1.0 / (2.0 * np.pi ** 2)) * p_in ** 3  * dlogp_in
    W_out = (1.0 / (2.0 * np.pi ** 2)) * p_out ** 3 * dlogp_in
    invW_out = 1.0 / np.maximum(W_out, 1e-300)

    # --- GL quadrature nodes and weights ---
    mu_in_n, w_in_n = np.polynomial.legendre.leggauss(Nmu_in)
    mu_out_n, w_out_n = np.polynomial.legendre.leggauss(Nmu_out)
    mu_in_n  = np.ascontiguousarray(mu_in_n, dtype=np.float64)
    mu_out_n = np.ascontiguousarray(mu_out_n, dtype=np.float64)

    # --- fold in the 0.5 factor for dmu/2 normalization ---
    w_in_n = np.ascontiguousarray(0.5 * w_in_n, dtype=np.float64)
    wP_out = np.ascontiguousarray(0.5 * w_out_n, dtype=np.float64)

    return p_out, p_in, E_in, PS_in, invW_out, mu_in_n, w_in_n, mu_out_n, wP_out




# ===========================================================================
# Run-time helpers -- called every solver step
# ===========================================================================

@jit(nopython=True, cache=True)
def _apply_gain_numba(k_idx, l_idx, w_base, idx_out, f_in, Nout):
    """
    Scatter-add the event list into a df/dt array,

        df[idx_out[e]] += w_base[e] * f_in[k_idx[e]] * f_in[l_idx[e]].

    Bilinear in f because two particles annihilate, and both legs index the
    same distribution since the incoming pair is one species.
    """
    result = np.zeros(Nout)

    # for each event in the event list, add its contribution to the result array
    for i in range(len(k_idx)):
        # idx_out[i] is the index in the momentum grid of the outgoing particle,
        # so deposit the event into this corresponding momentum bin
        result[idx_out[i]] += w_base[i] * f_in[k_idx[i]] * f_in[l_idx[i]]
    return result

def apply_gain_cache(cache, f_in):
    """
    Contract a gain cache against the current distribution.

    Parameters
    ----------
    cache : dict from `build_gain_cache_xxZDZD` / `build_gain_cache_ZDZDxx`
    f_in  : distribution of the incoming species on the cache's p_in grid

    Returns
    -------
    df/dt on the cache's p_out grid, shape (Np_out,). Non-negative: this is
    the gain half only, and the caller adds the loss term separately.
    """
    f_in = np.asarray(f_in, dtype=np.float64)
    Nout = cache["p_out"].shape[0]
    if cache["w_base"].size == 0:
        return np.zeros(Nout, dtype=np.float64)

    # each cache has shape (N_events,)
    return _apply_gain_numba(
            cache["k_idx"], cache["l_idx"], cache["w_base"],
            cache["idx_out"], f_in, Nout
        )




def gain_cache_from_grid_nearest(T, T_grid, cache_grid):
    """
    Pick the tabulated gain cache closest to T.

    Nearest-neighbor rather than interpolation, so accuracy therefore
    comes from tabulating T_span finely enough.
    """
    i = int(np.argmin(np.abs(np.asarray(T_grid) - T)))
    return cache_grid[i]
