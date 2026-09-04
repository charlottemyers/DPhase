"""
Collision kernels for chi chibar <-> Z_D Z_D, the number-changing process
internal to the dark sector.

Both directions are built here, with the cross section provided by`model.sigma_s_xxAA`.
Each direction produces two distinct objects, because a Boltzmann collision
term splits into two pieces that need different treatment:

  loss  -- how fast the incoming pair is destroyed. Depends only on sigma(s),
           so it collapses to a dense (Np, Np) matrix K_ij, built by
           `_annih_loss_kernel`.
  gain  -- where in momentum the products land. Needs the outgoing kinematics
           boosted back from the CM frame, and the (k, l) -> i map is sparse
           and irregular, so it is stored as a flat event list rather than a
           matrix, built by `_build_gain_cache_inner_bilinear`.

Everything in this module runs once, before the solve, on a temperature grid;
`dphase.collisions` then looks up the nearest entry each step and contracts it
against the current distribution. Only the two `apply_*` / `*_from_grid_*` helpers
run every step.
"""

import numpy as np
import numba
from numba import jit

from dphase.grid import log_bin_edges
from dphase.model import _sigma_s_xxAA_scalar, sigma_s_xxAA

_AAXX_DOF_RATIO = 8.0 / 9.0


def _annih_loss_kernel(p, m_in, m_out, alphaD, Nmu=16, reverse=False):
    r"""
    Loss kernel K(p1, p2) for 2 -> 2 annihilation within the dark sector,

        K_ij = Int d\mu/2  sigma(s_ij(\mu))  v_Moller(s_ij(\mu)),

    where the two incoming legs sit in bins i and j and \mu is the cosine of
    the angle between them. Both directions of chi chibar <-> A A are built
    here; they differ only in which particle is incoming and in one crossing
    factor on sigma.

    Parameters
    ----------
    p       : physical momentum grid [GeV], shape (Np,)
    m_in    : mass of the INCOMING particles [GeV] -- the pair being destroyed.
              Sets E(p), s, and the Moller velocity.
    m_out   : mass of the outgoing particles [GeV]. Enters only through the
              threshold and, for `reverse=True`, the crossing factor.
    alphaD  : dark fine structure constant
    Nmu     : GL nodes for the angular integral
    reverse : False for chi chibar -> A A, using sigma_xxAA(s) directly.
              True for A A -> chi chibar, obtained by crossing (see below).

    Returns
    -------
    K : shape (Np, Np) [GeV^-2 * dimensionless velocity]

    Notes
    -----
    `sigma_s_xxAA` is always called in the chi chibar -> A A direction, i.e.
    with argument order (alphaD, mchi, mA), regardless of `reverse`. The
    reverse cross section is then obtained from it by the crossing relation

        sigma_AAxx = (8/9) (lambda_out / lambda_in) sigma_xxAA,

    with lambda_in / lambda_out the flux factors of the AA and chi chibar
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
    # forward it is the incoming-pair bound (s >= 4 m_in^2,automatic),
    # reverse it is the genuine chi chibar production threshold and the outgoing
    # constraint is the one doing the work.
    threshold = max(4.0 * m_in**2, 4.0 * m_out**2)
    valid = s >= threshold

    s_v = s[valid]
    E1_full = np.broadcast_to(E1, s.shape)
    E2_full = np.broadcast_to(E2, s.shape)

    mchi, mA = (m_out, m_in) if reverse else (m_in, m_out)
    sigma = np.zeros_like(s)
    sigma[valid] = sigma_s_xxAA(alphaD, mchi, mA, s_v)

    if reverse:
        # Cross to A A -> chi chibar. lambda(s, m^2, m^2) = s(s - 4m^2) is the
        # equal-mass Kallen function
        lam_in = s_v**2 - 4.0 * s_v * m_in**2       # A A
        lam_out = s_v**2 - 4.0 * s_v * m_out**2     # chi chibar
        sigma[valid] *= _AAXX_DOF_RATIO * (lam_out / lam_in)

    # Moller velocity of the incoming pair,
    v_mol = (np.sqrt(s_v * s_v - 4.0 * s_v * m_in**2)
             / (2.0 * E1_full[valid] * E2_full[valid]))

    integrand = np.zeros_like(s)
    # The 0.5 is the d\mu/2 angular normalisation, folded in here so the outer
    # reduction is just a tensordot against the GL weights.
    integrand[valid] = sigma[valid] * v_mol * 0.5

    return np.tensordot(integrand, mu_w, axes=([2], [0]))


# ===========================================================================
# Loss kernels -- dense (Np, Np) matrices
# ===========================================================================

def build_annihilation_loss_kernels(T_span, state, mchi, mA, direction):
    """
    Tabulate an annihilation loss kernel on a temperature grid.
    One kernel per temperature, because the comoving grid maps to different
    physical momenta at each T.

    Parameters
    ----------
    T_span    : temperatures [GeV]
    state     : PhaseSpaceState, supplying `gstar_func`, `alphaD` and `Nmu`
    mchi, mA  : chi and dark photon masses [GeV]
    direction : "xxAA" for chi chibar -> A A, "AAxx" for the reverse

    Returns
    -------
    list of (Np, Np) kernels, one per entry in T_span, aligned with it.
    """
    if direction not in ("xxAA", "AAxx"):
        raise ValueError(
            f"direction must be 'xxAA' or 'AAxx', got {direction!r}")
    reverse = (direction == "AAxx")
    # The incoming pair is chi chibar going forward, A A going in reverse.
    m_in, m_out = (mA, mchi) if reverse else (mchi, mA)

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
# ===========================================================================

@numba.njit
def _build_gain_cache_inner_bilinear(
    p_out, logp_out, p_in, E_in, PS_in, invW_out,
    mu_in, w_in, mu_out, wP_out,
    m_out, m_in, alphaD, yield_out, reverse,
):
    """
    Build the gain (deposition) event list for chi chibar <-> A A.

    Where a loss term only needs sigma(s), a gain term has to know where in
    momentum the products land. This routine walks every quadrature point of

        (incoming bin k) x (incoming bin l) x (mu_in) x (mu_out)

    and, for each, boosts the outgoing particle from the CM frame back to the
    cosmological frame to get its momentum p1. Because p1 rarely lands on
    a grid point, the event's weight is split linearly in log p between the two
    bracketing bins (hence "bilinear"). The result is a flat event list rather
    than a matrix, since the (k, l) -> i mapping is sparse and irregular.

    Applied later by `apply_gain_cache` as
        df_out[idx_out[e]] += w_base[e] * f_in[k_idx[e]] * f_in[l_idx[e]],
    which is why the weights already absorb the 1/W_out bin measure.

    Both directions are built here; they differ only in the cross section, see
    `reverse` below and the same relation in `_annih_loss_kernel`.

    Parameters
    ----------
    p_out, logp_out : outgoing momentum grid and its log [GeV]
    p_in, E_in      : incoming momentum grid and energies [GeV]
    PS_in           : incoming phase-space weight p^3 dlogp / (2 pi^2)
    invW_out        : 1 / (outgoing bin measure), pre-inverted
    mu_in, w_in     : Gauss-Legendre nodes/weights for the angle between the
                      two incoming momenta
    mu_out, wP_out  : nodes and (already halved) weights for the CM emission
                      angle of the outgoing particle
    m_out, m_in     : outgoing and incoming masses [GeV]
    alphaD          : dark fine structure constant
    yield_out       : number of the tracked species produced per reaction
                      (2 for the two A' from chi chibar -> A A, 1 otherwise)
    reverse         : False for chi chibar -> A A; True for A A -> chi chibar,
                      whose cross section is obtained from the same
                      sigma_xxAA by crossing.

    Returns
    -------
    (idx_out, k_idx, l_idx, w_base), each shape (N_events,)
    """
    Np_out = p_out.shape[0]
    Np_in = p_in.shape[0]
    Nmu_in = mu_in.shape[0]
    Nmu_out = mu_out.shape[0]

    # Upper bound on the event count: two deposits (i_lo, i_hi) per quadrature point.
    # Allocated once and sliced down at the end
    max_events = 2 * Np_in * Np_in * Nmu_in * Nmu_out
    idx_out_buf = np.empty(max_events, dtype=np.int64)
    k_buf = np.empty(max_events, dtype=np.int64)
    l_buf = np.empty(max_events, dtype=np.int64)
    w_buf = np.empty(max_events, dtype=np.float64)
    n = 0

    m_in_sq = m_in * m_in
    m_out_sq = m_out * m_out
    thresh = 4.0 * m_out_sq          # outgoing pair on shell

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

                # sigma_s_xxAA always takes (s, mchi, mA), so the two masses
                # swap roles between the directions.
                if reverse:
                    sig = _sigma_s_xxAA_scalar(s, m_out, m_in, alphaD)
                else:
                    sig = _sigma_s_xxAA_scalar(s, m_in, m_out, alphaD)
                if not np.isfinite(sig) or sig <= 0.0:
                    continue
                if reverse:
                    # Cross chi chibar -> A A into A A -> chi chibar. Safe to
                    # divide by lam_in: it was checked > 0 above.
                    sig = _AAXX_DOF_RATIO * (lam_out / lam_in) * sig


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

                for mj in range(Nmu_out):
                    # Boosted energy of one outgoing particle, emitted at cos(theta_CM) = mu_out[mj].
                    E1 = gamma * (E_star + beta * p_star * mu_out[mj])
                    p1sq = E1 * E1 - m_out_sq
                    if p1sq <= 0.0:
                        continue
                    p1 = np.sqrt(p1sq)

                    # products landing outside the grid are dropped.
                    if p1 <= p_out[0] or p1 >= p_out[-1]:
                        continue

                    # Bracketing bins, then linear interpolation in log p.
                    # get i_lo and i_hi: the indices of the lower and upper bracketing bins for p1
                    i_lo = np.searchsorted(p_out, p1) - 1
                    if i_lo < 0:
                        i_lo = 0
                    if i_lo > Np_out - 2:
                        i_lo = Np_out - 2
                    i_hi = i_lo + 1

                    # weight split between the nearest 2 bins
                    alpha = (np.log(p1) - logp_out[i_lo]) / \
                            (logp_out[i_hi] - logp_out[i_lo])
                    if alpha < 0.0:
                        alpha = 0.0
                    elif alpha > 1.0:
                        alpha = 1.0

                    # deposit the event, using yield_out = 2 for the two DPs
                    # note: wP_out already includes the 0.5 factor for dmu_out/2 normalization
                    rate_event = rate_base * wP_out[mj] * yield_out

                    # Split the deposit between the two bracketing bins. The
                    # invW_out factor converts a rate into a df/dt per bin.

                    # shape of idx_out_buf = (N_events,), where N_events is the total number of events accumulated so far
                    # populate buffers for the lower bracketing bin for the current event
                    # n tracks the total number of events accumulated so far; idx_out_buf tracks the index to the momentum grid of the outgoing particle for each event
                    idx_out_buf[n] = i_lo
                    k_buf[n] = k
                    l_buf[n] = l
                    w_buf[n] = (1.0 - alpha) * rate_event * invW_out[i_lo]

                    ### now populate the upper bracketing bin for the current event
                    n += 1
                    idx_out_buf[n] = i_hi
                    k_buf[n] = k
                    l_buf[n] = l
                    w_buf[n] = alpha * rate_event * invW_out[i_hi]
                    n += 1

    return idx_out_buf[:n], k_buf[:n], l_buf[:n], w_buf[:n]


def build_gain_cache_xxAA(
    p_out, p_in, dlogp_in,
    m_chi, m_A, alphaD,
    Nmu_in=4, Nmu_out=4,
    yield_out=1.0,
):
    """
    Gain cache for the A' produced by chi chibar -> A' A'.

    Thin wrapper: sets up the quadrature arrays, runs the event builder in the
    forward direction, and packages the result. All the physics is in
    `_build_gain_cache_inner_bilinear`.

    Parameters
    ----------
    p_out    : momentum grid the products are deposited onto [GeV]
    p_in     : momentum grid the incoming pair is read from [GeV]
    dlogp_in : log-momentum bin widths for `p_in`
    m_chi, m_A : masses [GeV]
    alphaD   : dark fine structure constant
    Nmu_in, Nmu_out : Gauss-Legendre nodes for the incoming-pair angle and the
               CM emission angle
    yield_out : products of the tracked species per reaction -- 2 here, since
               chi chibar -> A' A' makes two dark photons.

    Returns
    -------
    dict with keys idx_out, k_idx, l_idx, w_base (all shape (N_events,)) plus
    p_out and p_in. Consume it with `apply_gain_cache`.
    """
    p_out, p_in, E_in, PS_in, invW_out, mu_in_n, w_in_n, mu_out_n, wP_out = \
        _gain_cache_common_setup(p_out, p_in, dlogp_in, m_chi, Nmu_in, Nmu_out)

    logp_out = np.ascontiguousarray(np.log(p_out), dtype=np.float64)
    idx_out, k_idx, l_idx, w_base = _build_gain_cache_inner_bilinear(
        p_out, logp_out, p_in, E_in, PS_in, invW_out,
        mu_in_n, w_in_n, mu_out_n, wP_out,
        float(m_A), float(m_chi), float(alphaD),
        float(yield_out), False,
    )
    if idx_out.size == 0:
        return dict(
            idx_out=np.zeros(0, dtype=np.int64),
            k_idx=np.zeros(0, dtype=np.int64),
            l_idx=np.zeros(0, dtype=np.int64),
            w_base=np.zeros(0, dtype=float),
            p_out=p_out, p_in=p_in,
        )

    return dict(
        idx_out=idx_out, k_idx=k_idx, l_idx=l_idx, w_base=w_base,
        p_out=p_out, p_in=p_in,
    )


def build_gain_cache_AAxx(
    p_out, p_in, dlogp_in,
    m_chi, m_A, alphaD,
    Nmu_in=4, Nmu_out=4,
    yield_out=1.0,
):
    """
    Gain cache for the chi produced by A' A' -> chi chibar.

    The reverse counterpart of `build_gain_cache_xxAA`, and identical to it
    except that the incoming pair is now the A' (so `m_A` sets the energies)
    and the cross section is crossed from sigma_xxAA. Same return shape.

    `yield_out` is 1 here: one chi per reaction, with the chibar tracked as the
    same population (see `PhaseSpaceState.total_DM_number_density`).
    """
    p_out, p_in, E_in, PS_in, invW_out, mu_in_n, w_in_n, mu_out_n, wP_out = \
        _gain_cache_common_setup(p_out, p_in, dlogp_in, m_A, Nmu_in, Nmu_out)


    logp_out = np.ascontiguousarray(np.log(p_out), dtype=np.float64)
    idx_out, k_idx, l_idx, w_base = _build_gain_cache_inner_bilinear(
        p_out, logp_out, p_in, E_in, PS_in, invW_out,
        mu_in_n, w_in_n, mu_out_n, wP_out,
        float(m_chi), float(m_A), float(alphaD),
        float(yield_out), True,
    )


    if idx_out.size == 0:
        return dict(
            idx_out=np.zeros(0, dtype=np.int64),
            k_idx=np.zeros(0, dtype=np.int64),
            l_idx=np.zeros(0, dtype=np.int64),
            w_base=np.zeros(0, dtype=float),
            p_out=p_out, p_in=p_in,
        )

    # each array is shape (N_events,), with N_events = 2 * Np_in^2 * Nmu_in * Nmu_out
    return dict(
        idx_out=idx_out, k_idx=k_idx, l_idx=l_idx, w_base=w_base,
        p_out=p_out, p_in=p_in,
    )


def build_fixed_grid_gain_caches_xxAA(
    p_cache,
    *,
    m_chi, m_A, alphaD,
    yield_out_chi=1.0,
    yield_out_A=2.0,
    Nmu_in=4, Nmu_out=4,
):
    """
    Build both gain caches for one momentum grid, in one call.

    A single temperature needs two caches, because each species gains from the
    *other* one annihilating:

        cache_chi_from_AA : chi deposited by Z_D Z_D -> chi chibar   (reverse)
        cache_A_from_xx   : Z_D deposited by chi chibar -> Z_D Z_D   (forward)

    Note the crossing in the names: the chi cache is built by the AAxx builder.

    Parameters
    ----------
    p_cache : shared momentum grid, used as both p_in and p_out [GeV]
    m_chi, m_A, alphaD : model parameters
    yield_out_chi, yield_out_A : products per reaction, 1 and 2 respectively.
        These must stay consistent with the degeneracy prefactors in
        `collisions.AA_chichi_collision_rhs`
    Nmu_in, Nmu_out: passed through to the builders

    Returns
    -------
    dict holding both caches, the grid, and the parameters they were built
    with. The underscore-prefixed entries are provenance only -- nothing reads
    them, but they make a pickled cache self-describing.
    """
    p_cache = np.asarray(p_cache, float)
    dlogp_cache = log_bin_edges(p_cache)

    # gain of chi
    cache_chi_from_AA = build_gain_cache_AAxx(
        p_out=p_cache, p_in=p_cache, dlogp_in=dlogp_cache,
        m_chi=m_chi, m_A=m_A, alphaD=alphaD,
        Nmu_in=Nmu_in, Nmu_out=Nmu_out,
        yield_out=yield_out_chi,
    )

    # gain of A
    cache_A_from_xx = build_gain_cache_xxAA(
        p_out=p_cache, p_in=p_cache, dlogp_in=dlogp_cache,
        m_chi=m_chi, m_A=m_A, alphaD=alphaD,
        Nmu_in=Nmu_in, Nmu_out=Nmu_out,
        yield_out=yield_out_A,
    )
    return dict(
        p_cache=p_cache,
        dlogp_cache=dlogp_cache,
        cache_chi_from_AA=cache_chi_from_AA,
        cache_A_from_xx=cache_A_from_xx,
        _mchi=m_chi, _mA=m_A,
        _alphaD=alphaD,
        _Nmu_in=Nmu_in, _Nmu_out=Nmu_out,
    )


def build_annihilation_gain_caches(T_span, state):
    """
    Tabulate both gain caches on a temperature grid.

    The loop mirrors `build_annihilation_loss_kernels`: one entry per
    temperature, because the comoving grid maps to different physical momenta
    at each T.

    Speed: builder is O(Np^2 * Nmu_in * Nmu_out) per temperature.

    Parameters
    ----------
    T_span : temperatures [GeV]
    state  : PhaseSpaceState, supplying grid, gstar_func, species, alphaD, Nmu

    Returns
    -------
    list of dicts as returned by `build_fixed_grid_gain_caches_xxAA`, aligned
    with T_span. Assign to `state.gain_cache_grid`.

    """
    out = []
    for T in T_span:
        p_phys = state.grid.p_phys(float(T), gstars_func=state.gstar_func)
        cache_T = build_fixed_grid_gain_caches_xxAA(
            p_phys,
            m_chi=state.species["chi"].mass_GeV,
            m_A=state.species["A"].mass_GeV,
            alphaD=state.alphaD,
            Nmu_in=state.Nmu,
            Nmu_out=state.Nmu,
            yield_out_chi=1.0,
            yield_out_A=2.0,
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

    # --- Gauss-Legendre quadrature nodes and weights ---
    mu_in_n, w_in_n = np.polynomial.legendre.leggauss(Nmu_in)
    mu_out_n, w_out_n = np.polynomial.legendre.leggauss(Nmu_out)
    mu_in_n  = np.ascontiguousarray(mu_in_n, dtype=np.float64)
    mu_out_n = np.ascontiguousarray(mu_out_n, dtype=np.float64)

    # --- fold in the 0.5 factor for dmu/2 normalization ---
    w_in_n = np.ascontiguousarray(0.5 * w_in_n, dtype=np.float64)
    wP_out = np.ascontiguousarray(0.5 * w_out_n, dtype=np.float64) # 0.5 for the dmu/2 normalisation folded in

    return p_out, p_in, E_in, PS_in, invW_out, mu_in_n, w_in_n, mu_out_n, wP_out




# ===========================================================================
# Run-time helpers -- these are called every solver step
# ===========================================================================

@jit(nopython=True, cache=True)
def _apply_gain_numba(k_idx, l_idx, w_base, idx_out, f_in, Nout):
    """
    Scatter-add the event list into a df/dt array,

        df[idx_out[e]] += w_base[e] * f_in[k_idx[e]] * f_in[l_idx[e]].

    Bilinear in f because two particles annihilate, and both legs index the
    same distribution since the incoming pair is one species. Compiled: this
    runs once per species per Newton iteration
    """
    result = np.zeros(Nout)

    # for each event in the event list, add its contribution to the result array
    # recall: k_idx has shape (N_events,), same as l_idx, idx_out, w_base
    for i in range(len(k_idx)):
        # idx_out[i] is the index in the momentum grid of the outgoing particle,
        # so we deposit the event into this corresponding momentum bin
        result[idx_out[i]] += w_base[i] * f_in[k_idx[i]] * f_in[l_idx[i]]
    return result

def apply_gain_cache(cache, f_in):
    """
    Contract a gain cache against the current distribution.

    Parameters
    ----------
    cache : dict from `build_gain_cache_xxAA` / `build_gain_cache_AAxx`
    f_in  : distribution of the *incoming* species on the cache's p_in grid

    Returns
    -------
    df/dt on the cache's p_out grid, shape (Np_out,). Non-negative: this is
    the gain half only, and the caller adds the loss term separately.
    """
    f_in = np.asarray(f_in, dtype=np.float64)
    Nout = cache["p_out"].shape[0] # Nout = number of outgoing momentum grid points
    if cache["w_base"].size == 0:
        return np.zeros(Nout, dtype=np.float64)

    # each cache["..."] has shape (N_events,)
    return _apply_gain_numba(
            cache["k_idx"], cache["l_idx"], cache["w_base"],
            cache["idx_out"], f_in, Nout
        )




def gain_cache_from_grid_nearest(T, T_grid, cache_grid):
    """
    Pick the tabulated gain cache closest to T.

    Nearest-neighbor rather than interpolation, matching
    `annihilation_sm.kernel_from_grid_nearest`: the caches are sparse event
    lists with different lengths at different temperatures, so they cannot be
    blended. Accuracy therefore comes from tabulating T_span finely enough.
    """
    i = int(np.argmin(np.abs(np.asarray(T_grid) - T)))
    return cache_grid[i]
