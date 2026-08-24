"""
Elastic chi A' -> chi A' scattering -- "dark Compton".

This is the number-preserving / dark-sector cell of the taxonomy in
`dphase.kernels`. It moves chi and A' between momentum bins without changing
either population.

Computational cost
---------------------------------------------------------
The SM elastic channel (`elastic_sm`) uses a Fokker-Planck
approximation, valid because the bath particles are much lighter than chi so
each collision transfers little momentum. That fails here: both partners are
dark-sector particles of comparable mass. The operator therefore has to be a full 2 -> 2
redistribution, and the outgoing state depends on t as well as s -- which means
the quadrature needs BOTH CM angles, not just one:

    (k, l) x mu_in x mu_CM x phi_CM

against `annihilation_hidden`'s (k, l) x mu_in x mu_out. That extra axis, plus
the bilinear split over two outgoing grids instead of one, is what makes this
the slowest cache to build.

Detailed balance
----------------
The event cache stores an unnormalised rate R per (k, l, i_chi, i_A) tuple
rather than a pre-divided deposit weight. That choice is what lets the apply
step form

    delta = R * (f_chi[k] f_A[l] - f_chi[i_chi] f_A[i_A])

which vanishes exactly when the distribution is balanced between the incoming
and outgoing bins. Gain and loss are therefore handled together, per event, and
the operator has the correct fixed point by construction rather than by
cancellation between two separately-computed terms.
"""

import numpy as np
from numba import jit


# ===========================================================================
# Cache construction
# ===========================================================================

@jit(nopython=True, cache=True)
def _build_xA_elastic_cache_inner(
    p_grid, logp_grid, E_chi, E_A, PS,
    mu_in_n, w_in_n, mu_cm_n, w_cm_n, cos_phi_n, w_phi_n,
    m_chi, m_A, gD4, Navg,
):
    """
    Compiled event-list builder for the dark Compton operator.

    Walks every quadrature point

        (k = incoming chi bin) x (l = incoming A bin) x mu_in x mu_CM x phi_CM

    and, for each, boosts the outgoing pair from the CM frame back to the
    cosmological frame. The outgoing momenta (p3, p4) are continuous and land
    between grid points, so each quadrature point is split BILINEARLY in log p
    over the four combinations of bracketing bins -- (lo,lo), (lo,hi), (hi,lo),
    (hi,hi) -- with fractional weights that sum to 1.

    Each emitted sub-event is a tuple (k, l, i_chi, i_A, R). R is an
    unnormalized rate, NOT pre-divided by the bin measure W.

    Also accumulates K_loss[k, l] = (sum of R) / (W[k] W[l]), which is used
    only by the solver's rate-capping diagnostic, never during evolution.

    Parameters
    ----------
    p_grid, logp_grid : shared physical momentum grid for chi and A, and its log
    E_chi, E_A        : on-shell energies of each species on that grid [GeV]
    PS                : phase-space measure p^3 dlogp / (2 pi^2)
    mu_in_n, w_in_n   : Gauss-Legendre nodes/weights for the lab-frame angle
                        between the two incoming momenta
    mu_cm_n, w_cm_n   : nodes/weights for cos(theta_CM) of the outgoing chi
    cos_phi_n, w_phi_n : nodes/weights for the azimuth about the boost axis
    m_chi, m_A        : masses [GeV]
    gD4               : (4 pi alphaD)^2, the coupling factor of |M|^2
    Navg              : spin/polarisation averaging divisor

    Returns
    -------
    (ev_k, ev_l, ev_i_chi, ev_i_A, ev_R, K_loss), the first five all shape
    (N_subevents,) and K_loss shape (Np, Np).
    """
    Np = p_grid.shape[0]
    Nmu_in = mu_in_n.shape[0]
    Nmu_cm = mu_cm_n.shape[0]
    Nphi = cos_phi_n.shape[0]

    # 4 sub-events per quadrature event (bilinear interp into (lo,lo)..(hi,hi))
    max_subev = 4 * Np * Np * Nmu_in * Nmu_cm * Nphi
    ev_k_buf    = np.empty(max_subev, dtype=np.int64)
    ev_l_buf    = np.empty(max_subev, dtype=np.int64)
    ev_ichi_buf = np.empty(max_subev, dtype=np.int64)
    ev_iA_buf   = np.empty(max_subev, dtype=np.int64)
    ev_R_buf    = np.empty(max_subev, dtype=np.float64)
    K_loss_disc = np.zeros((Np, Np), dtype=np.float64)
    n_ev = 0

    m2_chi = m_chi * m_chi
    m2_A   = m_A   * m_A
    thresh_sq = (m_chi + m_A) * (m_chi + m_A)
    diff_sq   = (m_chi - m_A) * (m_chi - m_A)
    inv_64pi2 = 1.0 / (64.0 * np.pi * np.pi)

    p_lo = p_grid[0]
    p_hi = p_grid[-1]

    # k = incoming chi bin, l = incoming A bin, mu_in = cos(theta) between them in lab
    for k in range(Np):
        p_chi_k = p_grid[k]
        E_chi_k = E_chi[k]
        for l in range(Np):
            p_A_l = p_grid[l]
            E_A_l = E_A[l]
            pair_pref = PS[k] * PS[l]
            E_tot     = E_chi_k + E_A_l

            for mi in range(Nmu_in):
                # mu_i fixes s, hence the CM energy, hence the outgoing momenta
                # once the CM angles and the boost are applied below.
                mu_i  = mu_in_n[mi]
                wmu_i = w_in_n[mi]

                s = m2_chi + m2_A + 2.0 * (E_chi_k * E_A_l - p_chi_k * p_A_l * mu_i)
                if s <= thresh_sq:
                    continue

                lam_in = (s - thresh_sq) * (s - diff_sq)
                if lam_in <= 0.0:
                    continue
                sqrt_s     = np.sqrt(s)
                p_cm       = np.sqrt(lam_in) / (2.0 * sqrt_s)

                # Starred quantities are CM-frame, fixed entirely by s and the
                # two masses.
                E_chi_star = (s + m2_chi - m2_A) / (2.0 * sqrt_s)
                E_A_star   = (s + m2_A   - m2_chi) / (2.0 * sqrt_s)
                v_Mol      = np.sqrt(lam_in) / (2.0 * E_chi_k * E_A_l)

                P_tot_sq = p_chi_k * p_chi_k + p_A_l * p_A_l + 2.0 * p_chi_k * p_A_l * mu_i
                if P_tot_sq <= 0.0:
                    # Anti-aligned, near-equal momenta: the CM frame coincides
                    # with the lab, so the boost is the identity. Guarded with
                    # <= because roundoff can push a vanishing P_tot_sq
                    # negative.
                    gamma   = 1.0
                    cos_psi = 1.0
                    sin_psi = 0.0
                else:
                    P_tot = np.sqrt(P_tot_sq)
                    P_tot = np.sqrt(P_tot_sq)
                    beta  = P_tot / E_tot            # CM velocity in the lab
                    gamma = E_tot / sqrt_s           # = 1/sqrt(1 - beta^2)
                    # psi is the CM-frame angle between the boost axis and the
                    # incoming chi. It is what ties the CM emission angles to
                    # lab-frame momenta, via mu_b below.
                    p_chi_dot_b  = p_chi_k * (p_chi_k + p_A_l * mu_i) / P_tot
                    p_chi_par_cm = gamma * (p_chi_dot_b - beta * E_chi_k)
                    cos_psi      = p_chi_par_cm / p_cm
                    if cos_psi > 1.0:                # clamp against roundoff
                        cos_psi = 1.0
                    elif cos_psi < -1.0:
                        cos_psi = -1.0
                    sin_psi_sq = 1.0 - cos_psi * cos_psi
                    if sin_psi_sq < 0.0:
                        sin_psi_sq = 0.0
                    sin_psi = np.sqrt(sin_psi_sq)

                # Everything in the rate that depends only on s, hoisted out of
                # the two angular loops:
                #   PS[k] PS[l] * v_Mol * 1/(64 pi^2 s) * (dmu_in/2 weight).
                # The per-sub-event part is |M|^2 and the CM quadrature weights.
                rate_base = pair_pref * v_Mol * 0.5 * wmu_i * inv_64pi2 / s

                # now iterate over the CM angles to build the outgoing momenta and the per-event rate for each (k, l, mu_in, mu_CM, phi_CM) tuple
                for mc in range(Nmu_cm):
                    # mc = index for cos(theta_CM) quadrature; mu_c = cos(theta_CM)
                    mu_c = mu_cm_n[mc]
                    w_c  = w_cm_n[mc]

                    # |M|^2 inlined rather than calling model.xZD_xZD_m2, to keep
                    # this loop numba-compilable and allocation-free. For elastic
                    # scattering of equal CM momenta, t = -2 p_cm^2 (1 - cos).
                    t_val = -2.0 * p_cm * p_cm * (1.0 - mu_c)

                    u_val = 2.0 * m2_chi + 2.0 * m2_A - s - t_val
                    Ts  = 8.0 * (m2_chi*m2_chi + 3.0*m2_chi*s + m2_chi*u_val
                                 + 4.0*m2_chi*m2_A + m2_A*m2_A - s*u_val)
                    Tu  = 8.0 * (m2_chi*m2_chi + m2_chi*s + 3.0*m2_chi*u_val
                                 + 4.0*m2_chi*m2_A + m2_A*m2_A - s*u_val)
                    Tsu = 8.0 * (4.0*m2_chi*m2_chi + 2.0*m2_chi*m2_A
                                 - (m2_chi + 2.0*m2_A) * t_val)
                    xi_  = s - m2_chi     #s-channel propagator factor
                    eta_ = u_val - m2_chi #u-channel propagator factor (negative in phys. range)
                    if xi_ == 0.0 or eta_ == 0.0:
                        continue
                    M2 = gD4 / Navg * (Ts/(xi_*xi_) + Tu/(eta_*eta_)
                                       + 2.0*Tsu/(xi_*eta_))
                    if not np.isfinite(M2) or M2 <= 0.0:
                        continue


                    sin_th_sq = 1.0 - mu_c * mu_c
                    if sin_th_sq < 0.0:
                        sin_th_sq = 0.0
                    sin_th = np.sqrt(sin_th_sq)

                    # phi is the azimuth of the outgoing pair about the boost
                    # axis. |M|^2 depends on t, hence on theta_CM but NOT on
                    # phi -- which is why it is computed one level up and
                    # reused across this loop. phi still matters, because it
                    # changes how the CM angles project onto the lab frame.
                    for mp in range(Nphi):
                        cos_phi = cos_phi_n[mp]
                        w_p     = w_phi_n[mp]

                        # Angle between the boost axis and the outgoing chi, in
                        # the CM frame (PDG kinematics review).
                        mu_b = sin_th * cos_phi * sin_psi + mu_c * cos_psi

                        # Lab-frame energy of the outgoing chi. Events whose
                        # products fall off either end of the grid are dropped
                        # -- a real if usually small leak, so widen the grid if
                        # it matters.
                        E3 = gamma * (E_chi_star + beta * p_cm * mu_b)
                        p3_sq = E3 * E3 - m2_chi
                        if p3_sq <= 0.0:
                            continue
                        p3 = np.sqrt(p3_sq)
                        if p3 <= p_lo or p3 >= p_hi:
                            continue

                        # The outgoing A, fixed by energy-momentum conservation.
                        E4 = gamma * (E_A_star - beta * p_cm * mu_b)
                        p4_sq = E4 * E4 - m2_A
                        if p4_sq <= 0.0:
                            continue
                        p4 = np.sqrt(p4_sq)
                        if p4 <= p_lo or p4 >= p_hi:
                            continue

                        # One chi lands at p3 and one A at p4.
                        rate_event = rate_base * M2 * w_c * w_p

                        # Discretised loss rate, diagnostic only -- accumulated
                        # over the events actually kept, so it is consistent
                        # with the cache rather than with the exact integral.
                        K_loss_disc[k, l] += rate_event / pair_pref

                        # Bracketing bins for each product, then a bilinear
                        # split in log p over the four combinations.
                        i_chi_lo = np.searchsorted(p_grid, p3) - 1
                        if i_chi_lo < 0:
                            i_chi_lo = 0
                        if i_chi_lo > Np - 2:
                            i_chi_lo = Np - 2
                        i_chi_hi = i_chi_lo + 1
                        alpha_chi = (np.log(p3) - logp_grid[i_chi_lo]) \
                                    / (logp_grid[i_chi_hi] - logp_grid[i_chi_lo])
                        if alpha_chi < 0.0:
                            alpha_chi = 0.0
                        elif alpha_chi > 1.0:
                            alpha_chi = 1.0

                        i_A_lo = np.searchsorted(p_grid, p4) - 1
                        if i_A_lo < 0:
                            i_A_lo = 0
                        if i_A_lo > Np - 2:
                            i_A_lo = Np - 2
                        i_A_hi = i_A_lo + 1
                        alpha_A = (np.log(p4) - logp_grid[i_A_lo]) \
                                  / (logp_grid[i_A_hi] - logp_grid[i_A_lo])
                        if alpha_A < 0.0:
                            alpha_A = 0.0
                        elif alpha_A > 1.0:
                            alpha_A = 1.0

                        # 4 sub-events with bilinear weights summing to 1
                        w_ll = (1.0 - alpha_chi) * (1.0 - alpha_A)
                        w_lh = (1.0 - alpha_chi) * alpha_A
                        w_hl = alpha_chi         * (1.0 - alpha_A)
                        w_hh = alpha_chi         * alpha_A

                        # (i_chi_lo, i_A_lo)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_lo
                        ev_iA_buf[n_ev]   = i_A_lo
                        ev_R_buf[n_ev]    = w_ll * rate_event
                        n_ev += 1
                        # (i_chi_lo, i_A_hi)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_lo
                        ev_iA_buf[n_ev]   = i_A_hi
                        ev_R_buf[n_ev]    = w_lh * rate_event
                        n_ev += 1
                        # (i_chi_hi, i_A_lo)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_hi
                        ev_iA_buf[n_ev]   = i_A_lo
                        ev_R_buf[n_ev]    = w_hl * rate_event
                        n_ev += 1
                        # (i_chi_hi, i_A_hi)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_hi
                        ev_iA_buf[n_ev]   = i_A_hi
                        ev_R_buf[n_ev]    = w_hh * rate_event
                        n_ev += 1

    return (ev_k_buf[:n_ev], ev_l_buf[:n_ev],
            ev_ichi_buf[:n_ev], ev_iA_buf[:n_ev],
            ev_R_buf[:n_ev], K_loss_disc)


def build_xA_elastic_gain_cache(
    p_grid, dlogp, m_chi, m_A, alphaD,
    Nmu_in=8, Nmu_cm=8, Nphi_cm=4, average=True,
):
    """
    Build the dark Compton event cache for one physical momentum grid.

    Sets up the quadrature nodes and phase-space weights, runs the compiled
    builder, then deduplicates the resulting event list. Consume the result
    with `apply_xA_cache`.

    Parameters
    ----------
    p_grid, dlogp : shared physical momentum grid for chi and A, and its
        log-bin widths. Both species use the same grid, which is what lets a
        single (k, l, i_chi, i_A) index tuple describe an event.
    m_chi, m_A : masses [GeV]
    alphaD     : dark fine structure constant
    Nmu_in     : nodes for the lab-frame incoming angle
    Nmu_cm     : nodes for cos(theta_CM)
    Nphi_cm    : nodes for the azimuth. Uniform rather than Gauss-Legendre,
        since the integrand is periodic in phi.
    average    : divide |M|^2 by the initial-state multiplicity
        2 (chi spins) x 3 (A polarisations) = 6. Set False for the
        polarisation-summed matrix element.

    Returns
    -------
    dict with keys:
        p_grid, dlogp : the grid it was built for
        invW          : 1 / bin measure, pre-inverted for the apply step
        events        : dict of k_idx, l_idx, i_chi, i_A, R
        K_loss        : (Np, Np) discretised loss rate, DIAGNOSTIC ONLY --
                        the evolution operator does not use it
    plus underscore-prefixed provenance entries recording the parameters used.
    """
    p_grid = np.ascontiguousarray(p_grid, dtype=np.float64)
    dlogp  = np.ascontiguousarray(dlogp,  dtype=np.float64)

    E_chi = np.sqrt(p_grid * p_grid + m_chi * m_chi)
    E_A   = np.sqrt(p_grid * p_grid + m_A   * m_A)
    # Incoming and outgoing measures coincide here, because chi and A share a
    # single grid; kept as two names to mirror the annihilation caches.
    PS    = (1.0 / (2.0 * np.pi * np.pi)) * p_grid * p_grid * p_grid * dlogp
    W     = PS

    mu_in_n, w_in_n = np.polynomial.legendre.leggauss(Nmu_in)
    mu_cm_n, w_cm_n = np.polynomial.legendre.leggauss(Nmu_cm)
    # Uniform in phi on [0, 2 pi): the integrand is periodic, so the midpoint
    # rule converges faster here than Gauss-Legendre on a finite interval.
    phi_nodes = 2.0 * np.pi * (np.arange(Nphi_cm) + 0.5) / Nphi_cm
    cos_phi_n = np.cos(phi_nodes)
    w_phi_n   = np.full(Nphi_cm, 2.0 * np.pi / Nphi_cm)

    gD4  = (4.0 * np.pi * alphaD) ** 2
    Navg = 6.0 if average else 1.0   # 2 chi spins x 3 A polarisations

    logp_grid = np.log(p_grid)

    (ev_k, ev_l, ev_ichi, ev_iA, ev_R,
     K_loss_disc) = _build_xA_elastic_cache_inner(
        p_grid,
        np.ascontiguousarray(logp_grid, dtype=np.float64),
        np.ascontiguousarray(E_chi, dtype=np.float64),
        np.ascontiguousarray(E_A,   dtype=np.float64),
        np.ascontiguousarray(PS,    dtype=np.float64),
        np.ascontiguousarray(mu_in_n, dtype=np.float64),
        np.ascontiguousarray(w_in_n,  dtype=np.float64),
        np.ascontiguousarray(mu_cm_n, dtype=np.float64),
        np.ascontiguousarray(w_cm_n,  dtype=np.float64),
        np.ascontiguousarray(cos_phi_n, dtype=np.float64),
        np.ascontiguousarray(w_phi_n,   dtype=np.float64),
        float(m_chi), float(m_A), float(gD4), float(Navg),
    )
    # The raw list has one entry per sub-event, so it grows as
    # 4 * Np^2 * Nmu_in * Nmu_cm * Nphi_cm -- far more entries than there are
    # distinct (k, l, i_chi, i_A) tuples, since many angle combinations
    # interpolate into the same four bins. Collapsing them is a large win.
    Np_grid = p_grid.shape[0]
    ev_k_c, ev_l_c, ev_ichi_c, ev_iA_c, ev_R_c = _dedup_event_entries(
        ev_k, ev_l, ev_ichi, ev_iA, ev_R, Np_grid)

    # Pre-inverted so the compiled apply step multiplies instead of divides.
    invW_arr = (1.0 / np.maximum(W, 1e-300)).astype(np.float64)

    return dict(
        p_grid=p_grid,
        dlogp=dlogp,
        invW=invW_arr,
        events=dict(k_idx=ev_k_c, l_idx=ev_l_c,
                    i_chi=ev_ichi_c, i_A=ev_iA_c, R=ev_R_c),
        K_loss=np.ascontiguousarray(K_loss_disc).copy(),
        _m_chi=m_chi, _m_A=m_A,
        _alphaD=alphaD, _average=average,
        _Nmu_in=Nmu_in, _Nmu_cm=Nmu_cm, _Nphi_cm=Nphi_cm,
    )




def _dedup_event_entries(k_idx, l_idx, i_chi, i_A, R, Np):
    """
    Collapse cache entries sharing the same (k, l, i_chi, i_A) tuple, summing
    their rates.

    Because bilinear split lands many different angular quadrature points
    in the same bins, the raw event list is highly redundant --
    deduplicating typically shrinks it by ~5x, which speeds up every
    subsequent apply call.

    Summing R is exact, not an approximation: the apply step is linear in R for
    a fixed (k, l, i_chi, i_A), so merging entries changes nothing.

    Parameters
    ----------
    k_idx, l_idx, i_chi, i_A, R : the raw event arrays
    Np : grid size, used to pack the four indices into one key

    Returns
    -------
    The same five arrays, deduplicated and with zero-rate entries dropped.
    """
    if k_idx.size == 0:
        z_i = np.zeros(0, dtype=np.int64)
        return z_i, z_i.copy(), z_i.copy(), z_i.copy(), np.zeros(0, dtype=np.float64)

    # Pack the four indices into one int64 key so np.unique can do the grouping
    k64 = k_idx.astype(np.int64)
    l64 = l_idx.astype(np.int64)
    ic64 = i_chi.astype(np.int64)
    iA64 = i_A.astype(np.int64)
    flat = ((k64 * Np + l64) * Np + ic64) * Np + iA64

    uniq, inv = np.unique(flat, return_inverse=True)
    R_summed = np.bincount(inv, weights=R, minlength=uniq.size)

    nz = R_summed != 0.0
    uniq = uniq[nz]
    R_summed = R_summed[nz]

    iA_out  = (uniq % Np).astype(np.int64)
    ichi_out = ((uniq // Np) % Np).astype(np.int64)
    l_out   = ((uniq // (Np * Np)) % Np).astype(np.int64)
    k_out   = (uniq // (Np * Np * Np)).astype(np.int64)
    return k_out, l_out, ichi_out, iA_out, R_summed.astype(np.float64)


# ===========================================================================
# Run-time helpers -- called every solver step, not once up front
# ===========================================================================

@jit(nopython=True, cache=True)
def _apply_xA_delta_numba(k_idx, l_idx, i_chi, i_A, R, invW, f_chi, f_A, Np):
    """
    Apply the per-event delta operator.

    For each event (k, l, i_chi, i_A, R):

        delta          = R * (f_chi[k] f_A[l] - f_chi[i_chi] f_A[i_A])
        df_chi[k]     -= delta / W[k]        # chi leaves the incoming bin
        df_chi[i_chi] += delta / W[i_chi]    # ...and arrives in the outgoing one
        df_A[l]       -= delta / W[l]
        df_A[i_A]     += delta / W[i_A]

    Two properties follow directly from this form. Delta vanishes whenever the
    distribution is balanced across the event, so discrete detailed balance
    holds per event rather than as a cancellation between separately computed
    gain and loss terms. And every event removes exactly as much as it adds, so
    particle number is conserved bin-to-bin.

    Compiled, and the hot loop of the dark Compton operator: it runs once per
    Newton iteration over the full deduplicated event list.
    """
    df_chi = np.zeros(Np)
    df_A   = np.zeros(Np)
    for i in range(k_idx.shape[0]):
        k  = k_idx[i]
        l  = l_idx[i]
        ic = i_chi[i]
        iA = i_A[i]
        delta = R[i] * (f_chi[k] * f_A[l] - f_chi[ic] * f_A[iA])
        df_chi[k]  -= delta * invW[k] # remove from incoming chi bin
        df_chi[ic] += delta * invW[ic] # add to outgoing chi bin
        df_A[l]    -= delta * invW[l] # remove from incoming A bin
        df_A[iA]   += delta * invW[iA] # add to outgoing A bin
    return df_chi, df_A


def apply_xA_cache(cache, f_chi, f_A):
    """
    Compute the full dark Compton collision term for both species.

    Parameters
    ----------
    cache        : dict from `build_xA_elastic_gain_cache`
    f_chi, f_A   : current distributions on the cache's `p_grid`

    Returns
    -------
    (df_chi, df_A), each shape (Np,). Sums to zero over each species up to the
    grid-edge leakage noted in the builder.
    """
    Np = cache["p_grid"].shape[0]
    ev = cache["events"]
    if ev["R"].size == 0:
        return np.zeros(Np), np.zeros(Np)
    f_chi = np.ascontiguousarray(f_chi, dtype=np.float64)
    f_A   = np.ascontiguousarray(f_A,   dtype=np.float64)
    return _apply_xA_delta_numba(
        ev["k_idx"], ev["l_idx"], ev["i_chi"], ev["i_A"], ev["R"],
        cache["invW"], f_chi, f_A, Np,
    )




def apply_xA_elastic_loss(p_grid, dlogp, K_loss, f_chi, f_A):
    """
    Loss-only rate, for diagnostics.

        df_chi[i]/dt|loss = -f_chi[i] sum_j W[j] K_loss[i,j] f_A[j]
        df_A[j]/dt|loss   = -f_A[j]   sum_i W[i] K_loss[i,j] f_chi[i]

    `apply_xA_cache` already includes the loss inside its per-event delta.
    The only caller is `solver.get_dark_compton_loss_term`, which needs a loss-only rate to
    estimate Gamma/H for the stiffness cap.
    """
    W = (1.0 / (2.0 * np.pi * np.pi)) * p_grid * p_grid * p_grid * dlogp
    df_chi = -f_chi * (K_loss * (W * f_A)[None, :]).sum(axis=1)
    df_A   = -f_A   * (K_loss * (W * f_chi)[:, None]).sum(axis=0)
    return df_chi, df_A


# ===========================================================================
# Temperature-grid builder
# ===========================================================================

def build_xAxA_gain_caches(T_span, state, Nmu_in=8, Nmu_cm=8, Nphi_cm=4, average=True):
    """
    Tabulate the dark Compton cache on a temperature grid.

    A separate cache is genuinely needed at every temperature even though the
    grid structure never changes: the stored momenta are comoving, while the
    scattering rate depends on physical momenta, which redshift with T.

    Parameters
    ----------
    T_span : temperatures [GeV]
    state  : PhaseSpaceState, supplying grid, gstar_func, species and alphaD
    Nmu_in, Nmu_cm, Nphi_cm, average : passed through to the builder

    Returns
    -------
    list of cache dicts aligned with T_span. Assign to
    `state.gain_caches_xAxA`.

    """
    out = []
    n_T = len(T_span)
    m_chi = state.species["chi"].mass_GeV
    m_A   = state.species["A"].mass_GeV
    for i, T in enumerate(T_span):
        T = float(T)
        p = state.grid.p_phys(T, gstars_func=state.gstar_func)
        cache_T = build_xA_elastic_gain_cache(
            p_grid=p, dlogp=state.grid.dlogp,
            m_chi=m_chi, m_A=m_A, alphaD=state.alphaD,
            Nmu_in=Nmu_in, Nmu_cm=Nmu_cm, Nphi_cm=Nphi_cm,
            average=average,
        )
        out.append(cache_T)
        if i % max(1, n_T // 10) == 0:
            print(f"  xA-gain [{i+1}/{n_T}]: T={T:.3e} GeV  "
                  f"N_events={cache_T['events']['R'].size}")
    return out



def _xA_kernel_index(T, T_grid):
    """
    Index of the tabulated cache nearest to T.

    Nearest-neighbour, matching every other lookup in `dphase.kernels`.
    """
    return int(np.argmin(np.abs(np.asarray(T_grid) - float(T))))
