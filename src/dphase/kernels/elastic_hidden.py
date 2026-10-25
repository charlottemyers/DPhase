"""
Dark Compton: elastic chi Z_D -> chi Z_D scattering.

Computational cost
---------------------------------------------------------
Here, both scattering partners are dark-sector particles of comparable mass. The operator
therefore has to be a full 2 -> 2 redistribution, and the outgoing state depends on t as well
as s, so the quadrature needs BOTH CM angles: (k, l) x mu_in x mu_CM x phi_CM

Detailed balance
----------------
The event cache stores an unnormalized rate R per (k, l, i_chi, i_ZD) tuple
rather than a pre-divided deposit weight. Gain and loss are handled together,
per event, and the operator has the correct fixed point by construction rather than by
cancellation between two separately-computed terms.
"""

import numpy as np
from numba import jit

from dphase.model import compute_couplings


# ===========================================================================
# Cache construction
# ===========================================================================

@jit(nopython=True, cache=True)
def _build_xZD_elastic_cache_inner(
    p_grid, E_chi, E_ZD, PS,
    mu_in_n, w_in_n, mu_cm_n, w_cm_n,
    cos_phi_n, w_phi_n,
    m_chi, m_ZD, g_ZDchi4, Navg,
):
    """
    Compiled event-list builder. Walks every quadrature point

        (k = incoming chi bin) x (l = incoming Z_D bin) x mu_in x mu_CM x phi_CM

    and, for each, boosts the outgoing pair from the CM frame back to the
    cosmological frame. The outgoing momenta (p3, p4) are continuous and land
    between grid points, so each quadrature point is split BILINEARLY, linear
    in energy, over the four combinations of bracketing bins with weights that sum to 1.

    Each emitted sub-event is a tuple (k, l, i_chi, i_ZD, R). R is an
    unnormalized rate. Also accumulates K_loss[k, l] = (sum of R) / (W[k] W[l]),
    which is used only by the solver's rate-capping diagnostic, never during evolution.

    Parameters
    ----------
    p_grid            : shared physical momentum grid for chi and Z_D
    E_chi, E_ZD        : on-shell energies of each species on that grid [GeV]
    PS                : phase-space measure p^3 dlogp / (2 pi^2)
    mu_in_n, w_in_n   : GL nodes/weights for the lab-frame angle
                        between the two incoming momenta
    mu_cm_n, w_cm_n   : nodes/weights for cos(theta_CM) of the outgoing chi
    cos_phi_n, w_phi_n : nodes/weights for the azimuth about the boost axis
    m_chi, m_ZD        : masses [GeV]
    g_ZDchi4          : g_ZDchi^4, the physical chi-chi-Z_D coupling to the 4th
    Navg              : spin/polarisation averaging divisor

    Returns
    -------
    (ev_k, ev_l, ev_i_chi, ev_i_ZD, ev_R, K_loss), the first five all shape
    (N_subevents,) and K_loss shape (Np, Np).
    """
    Np = p_grid.shape[0]
    Nmu_in = mu_in_n.shape[0]
    Nmu_cm = mu_cm_n.shape[0]
    Nphi = cos_phi_n.shape[0]

    # 4 sub-events per quadrature event
    max_subev = 4 * Np * Np * Nmu_in * Nmu_cm * Nphi
    ev_k_buf    = np.empty(max_subev, dtype=np.int64)
    ev_l_buf    = np.empty(max_subev, dtype=np.int64)
    ev_ichi_buf = np.empty(max_subev, dtype=np.int64)
    ev_iZD_buf   = np.empty(max_subev, dtype=np.int64)
    ev_R_buf    = np.empty(max_subev, dtype=np.float64)
    K_loss_disc = np.zeros((Np, Np), dtype=np.float64)
    n_ev = 0

    m2_chi = m_chi * m_chi
    m2_ZD   = m_ZD   * m_ZD
    thresh_sq = (m_chi + m_ZD) * (m_chi + m_ZD)
    diff_sq   = (m_chi - m_ZD) * (m_chi - m_ZD)
    inv_64pi2 = 1.0 / (64.0 * np.pi * np.pi)

    p_lo = p_grid[0]
    p_hi = p_grid[-1]

    # k = incoming chi bin, l = incoming Z_D bin, mu_in = cos(theta) between them in lab
    for k in range(Np):
        p_chi_k = p_grid[k]
        E_chi_k = E_chi[k]
        for l in range(Np):
            p_ZD_l = p_grid[l]
            E_ZD_l = E_ZD[l]
            pair_pref = PS[k] * PS[l]
            E_tot     = E_chi_k + E_ZD_l

            for mi in range(Nmu_in):
                # mu = cos(theta) between the incoming chi and Z_D 3-momenta in the lab frame
                # mu_i fixes s, hence the CM energy s
                mu_i  = mu_in_n[mi]
                wmu_i = w_in_n[mi]

                s = m2_chi + m2_ZD + 2.0 * (E_chi_k * E_ZD_l - p_chi_k * p_ZD_l * mu_i)
                if s <= thresh_sq:
                    continue

                lam_in = (s - thresh_sq) * (s - diff_sq)
                if lam_in <= 0.0:
                    continue
                sqrt_s     = np.sqrt(s)
                p_cm       = np.sqrt(lam_in) / (2.0 * sqrt_s)

                # Starred quantities are CM-frame, fixed entirely by s.
                E_chi_star = (s + m2_chi - m2_ZD) / (2.0 * sqrt_s)
                E_ZD_star   = (s + m2_ZD   - m2_chi) / (2.0 * sqrt_s)
                v_Mol      = np.sqrt(lam_in) / (2.0 * E_chi_k * E_ZD_l)

                P_tot_sq = p_chi_k * p_chi_k + p_ZD_l * p_ZD_l + 2.0 * p_chi_k * p_ZD_l * mu_i
                if P_tot_sq <= 0.0:
                    # Anti-aligned, near-equal momenta: the CM frame coincides
                    # with the lab, so the boost does nothing
                    gamma   = 1.0
                    beta    = 0.0
                    cos_psi = 1.0
                    sin_psi = 0.0
                else:
                    P_tot = np.sqrt(P_tot_sq)
                    beta  = P_tot / E_tot            # CM velocity in the lab
                    gamma = E_tot / sqrt_s           # = 1/sqrt(1 - beta^2)
                    # psi is the CM-frame angle between the boost axis and the incoming chi.
                    # Ties the CM emission angles to lab-frame momenta, via mu_b below.
                    p_chi_dot_b  = p_chi_k * (p_chi_k + p_ZD_l * mu_i) / P_tot
                    p_chi_par_cm = gamma * (p_chi_dot_b - beta * E_chi_k)
                    cos_psi      = p_chi_par_cm / p_cm
                    if cos_psi > 1.0:                # clamp
                        cos_psi = 1.0
                    elif cos_psi < -1.0:             # clamp
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

                # Iterate over the CM angles to build the outgoing momenta
                for mc in range(Nmu_cm):
                    # mc = index for cos(theta_CM) quadrature
                    # mu_c = cos(theta_CM), where theta_CM is the CM-frame scattering angle
                    mu_c = mu_cm_n[mc]
                    w_c  = w_cm_n[mc]

                    # |M|^2 inlined to keepthis loop numba-compilable.
                    # For elastic scattering of equal CM momenta, t = -2 p_cm^2 (1 - cos).
                    t_val = -2.0 * p_cm * p_cm * (1.0 - mu_c)

                    u_val = 2.0 * m2_chi + 2.0 * m2_ZD - s - t_val
                    Ts  = 8.0 * (m2_chi*m2_chi + 3.0*m2_chi*s + m2_chi*u_val
                                 + 4.0*m2_chi*m2_ZD + m2_ZD*m2_ZD - s*u_val)
                    Tu  = 8.0 * (m2_chi*m2_chi + m2_chi*s + 3.0*m2_chi*u_val
                                 + 4.0*m2_chi*m2_ZD + m2_ZD*m2_ZD - s*u_val)
                    Tsu = 8.0 * (4.0*m2_chi*m2_chi + 2.0*m2_chi*m2_ZD
                                 - (m2_chi + 2.0*m2_ZD) * t_val)
                    xi_  = s - m2_chi     #s-channel propagator factor
                    eta_ = u_val - m2_chi #u-channel propagator factor
                    if xi_ == 0.0 or eta_ == 0.0:
                        continue
                    M2 = g_ZDchi4 / Navg * (Ts/(xi_*xi_) + Tu/(eta_*eta_)
                                       + 2.0*Tsu/(xi_*eta_))
                    if not np.isfinite(M2) or M2 <= 0.0:
                        continue


                    sin_th_sq = 1.0 - mu_c * mu_c
                    if sin_th_sq < 0.0:
                        sin_th_sq = 0.0
                    sin_th = np.sqrt(sin_th_sq)

                    # phi is the azimuth of the outgoing pair about the boost
                    # axis. |M|^2 depends on t, hence on theta_CM but NOT on phi
                    # phi sets how the CM angles project onto the lab frame.
                    for mp in range(Nphi):
                        cos_phi = cos_phi_n[mp]
                        w_p     = w_phi_n[mp]

                        # Angle between the boost axis and the outgoing chi, in CM frame
                        mu_b = sin_th * cos_phi * sin_psi + mu_c * cos_psi

                        # Lab-frame energy of the outgoing chi. Events whose
                        # products fall off either end of the grid are dropped
                        E3 = gamma * (E_chi_star + beta * p_cm * mu_b)
                        p3_sq = E3 * E3 - m2_chi
                        if p3_sq <= 0.0:
                            continue

                        # p3 = lab-frame momentum of the outgoing chi particle
                        p3 = np.sqrt(p3_sq)
                        if p3 <= p_lo or p3 >= p_hi:
                            continue

                        # The outgoing Z_D, fixed by energy-momentum conservation.
                        E4 = gamma * (E_ZD_star - beta * p_cm * mu_b)

                        p4_sq = E4 * E4 - m2_ZD
                        if p4_sq <= 0.0:
                            continue

                        # p4 = lab-frame momentum of the outgoing Z_D particle
                        p4 = np.sqrt(p4_sq)
                        if p4 <= p_lo or p4 >= p_hi:
                            continue

                        rate_event = rate_base * M2 * w_c * w_p

                        # Discretized loss rate, diagnostic only
                        K_loss_disc[k, l] += rate_event / pair_pref

                        # Bracketing bins for each product, then a bilinear
                        # split, linear in energy, over the four combinations.
                        # Energy-linear so the split conserves energy exactly.
                        i_chi_lo = np.searchsorted(p_grid, p3) - 1
                        if i_chi_lo < 0:
                            i_chi_lo = 0
                        if i_chi_lo > Np - 2:
                            i_chi_lo = Np - 2
                        i_chi_hi = i_chi_lo + 1

                        # Interpolation fraction for the chi particle between its bracketing bins
                        alpha_chi = (E3 - E_chi[i_chi_lo]) \
                                    / (E_chi[i_chi_hi] - E_chi[i_chi_lo])
                        if alpha_chi < 0.0:
                            alpha_chi = 0.0
                        elif alpha_chi > 1.0:
                            alpha_chi = 1.0

                        i_ZD_lo = np.searchsorted(p_grid, p4) - 1
                        if i_ZD_lo < 0:
                            i_ZD_lo = 0
                        if i_ZD_lo > Np - 2:
                            i_ZD_lo = Np - 2
                        i_ZD_hi = i_ZD_lo + 1
                        alpha_ZD = (E4 - E_ZD[i_ZD_lo]) \
                                  / (E_ZD[i_ZD_hi] - E_ZD[i_ZD_lo])
                        if alpha_ZD < 0.0:
                            alpha_ZD = 0.0
                        elif alpha_ZD > 1.0:
                            alpha_ZD = 1.0

                        # because we're depositing 4 sub-events corresponding to the four combinations of the bracketing bins
                        # the weights for these sub-events are given by the bilinear interpolation fractions
                        w_ll = (1.0 - alpha_chi) * (1.0 - alpha_ZD)
                        w_lh = (1.0 - alpha_chi) * alpha_ZD
                        w_hl = alpha_chi         * (1.0 - alpha_ZD)
                        w_hh = alpha_chi         * alpha_ZD

                        # (i_chi_lo, i_ZD_lo)
                        # k, l = incoming chi and Z_D momenta
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        # i_chi_lo, i_ZD_lo = indices for the outgoing chi and Z_D particles in the momentum grid
                        ev_ichi_buf[n_ev] = i_chi_lo
                        ev_iZD_buf[n_ev]   = i_ZD_lo
                        # R = rate weight for this sub-event
                        ev_R_buf[n_ev]    = w_ll * rate_event
                        n_ev += 1
                        # (i_chi_lo, i_ZD_hi)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_lo
                        ev_iZD_buf[n_ev]   = i_ZD_hi
                        ev_R_buf[n_ev]    = w_lh * rate_event
                        n_ev += 1
                        # (i_chi_hi, i_ZD_lo)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_hi
                        ev_iZD_buf[n_ev]   = i_ZD_lo
                        ev_R_buf[n_ev]    = w_hl * rate_event
                        n_ev += 1
                        # (i_chi_hi, i_ZD_hi)
                        ev_k_buf[n_ev]    = k
                        ev_l_buf[n_ev]    = l
                        ev_ichi_buf[n_ev] = i_chi_hi
                        ev_iZD_buf[n_ev]   = i_ZD_hi
                        ev_R_buf[n_ev]    = w_hh * rate_event
                        n_ev += 1

    # for each event we save: the incoming momenta (k, l), the outgoing momentum bin indices (i_chi, i_ZD), and the rate weight (R);

    return (ev_k_buf[:n_ev], ev_l_buf[:n_ev],
            ev_ichi_buf[:n_ev], ev_iZD_buf[:n_ev],
            ev_R_buf[:n_ev], K_loss_disc)


def build_xZD_elastic_gain_cache(
    p_grid, dlogp, m_chi, m_ZD, g_ZDchi,
    Nmu_in=8, Nmu_cm=8, Nphi_cm=4
):
    """
    Build the dark Compton event cache for one physical momentum grid.

    Sets up the quadrature nodes and phase-space weights, runs the compiled
    builder, deduplicates the resulting event list.

    Parameters
    ----------
    p_grid, dlogp : shared physical momentum grid for chi and Z_D, and its
        log-bin widths. Both species use the same grid, which is what lets a
        single (k, l, i_chi, i_ZD) index tuple describe an event.
    m_chi, m_ZD : masses [GeV]
    g_ZDchi    : physical chi-chi-Z_D coupling gD cos(alpha), from
                 `model.compute_couplings`
    Nmu_in     : nodes for the lab-frame incoming angle
    Nmu_cm     : nodes for cos(theta_CM)
    Nphi_cm    : nodes for the azimuth. Uniform rather than GL.

    Returns
    -------
    dict with keys:
        p_grid, dlogp : momentum grid
        invW          : 1 / bin measure, pre-inverted for the apply step
        events        : dict of k_idx, l_idx, i_chi, i_ZD, R
        K_loss        : (Np, Np) discretized loss rate, DIAGNOSTIC ONLY

    """
    p_grid = np.ascontiguousarray(p_grid, dtype=np.float64)
    dlogp  = np.ascontiguousarray(dlogp,  dtype=np.float64)

    E_chi = np.sqrt(p_grid * p_grid + m_chi * m_chi)
    E_ZD   = np.sqrt(p_grid * p_grid + m_ZD   * m_ZD)
    # Incoming and outgoing measures coincide here, because chi and Z_D share a single grid.
    PS    = (1.0 / (2.0 * np.pi * np.pi)) * p_grid * p_grid * p_grid * dlogp
    W     = PS

    mu_in_n, w_in_n = np.polynomial.legendre.leggauss(Nmu_in)
    mu_cm_n, w_cm_n = np.polynomial.legendre.leggauss(Nmu_cm)
    # Uniform in phi on [0, 2 pi): the integrand is periodic, so the midpoint
    # rule converges faster here than GL.
    phi_nodes = 2.0 * np.pi * (np.arange(Nphi_cm) + 0.5) / Nphi_cm
    cos_phi_n = np.cos(phi_nodes)
    w_phi_n   = np.full(Nphi_cm, 2.0 * np.pi / Nphi_cm)

    g_ZDchi4 = g_ZDchi ** 4
    Navg = 6.0  # 2 chi spins x 3 Z_D polarisations

    (ev_k, ev_l, ev_ichi, ev_iZD, ev_R,
     K_loss_disc) = _build_xZD_elastic_cache_inner(
        p_grid,
        np.ascontiguousarray(E_chi, dtype=np.float64),
        np.ascontiguousarray(E_ZD,   dtype=np.float64),
        np.ascontiguousarray(PS,    dtype=np.float64),
        np.ascontiguousarray(mu_in_n, dtype=np.float64),
        np.ascontiguousarray(w_in_n,  dtype=np.float64),
        np.ascontiguousarray(mu_cm_n, dtype=np.float64),
        np.ascontiguousarray(w_cm_n,  dtype=np.float64),
        np.ascontiguousarray(cos_phi_n, dtype=np.float64),
        np.ascontiguousarray(w_phi_n,   dtype=np.float64),
        float(m_chi), float(m_ZD), float(g_ZDchi4), float(Navg),
    )
    # Raw list has one entry per sub-event, so it grows as
    # 4 * Np^2 * Nmu_in * Nmu_cm * Nphi_cm: more entries than there are
    # distinct (k, l, i_chi, i_ZD) tuples, since many angle combinations
    # interpolate into the same four bins. Collapsing them speeds up the apply step.
    # (recall the factor of 4 comes from the four sub-events generated per scattering
    # event due to bilinear interpolation)
    Np_grid = p_grid.shape[0]
    ev_k_c, ev_l_c, ev_ichi_c, ev_iZD_c, ev_R_c = _dedup_event_entries(
        ev_k, ev_l, ev_ichi, ev_iZD, ev_R, Np_grid)

    invW_arr = (1.0 / np.maximum(W, 1e-300)).astype(np.float64)

    return dict(
        p_grid=p_grid,
        dlogp=dlogp,
        invW=invW_arr,
        events=dict(k_idx=ev_k_c, l_idx=ev_l_c,
                    i_chi=ev_ichi_c, i_ZD=ev_iZD_c, R=ev_R_c),
        K_loss=np.ascontiguousarray(K_loss_disc).copy(),
        _m_chi=m_chi, _m_ZD=m_ZD,
        _g_ZDchi=g_ZDchi,
        _Nmu_in=Nmu_in, _Nmu_cm=Nmu_cm, _Nphi_cm=Nphi_cm,
    )




def _dedup_event_entries(k_idx, l_idx, i_chi, i_ZD, R, Np):
    """
    Collapse cache entries sharing the same (k, l, i_chi, i_ZD) tuple, summing
    their rates.

    Because bilinear split lands many different angular quadrature points
    in the same bins, the raw event list is highly redundant

    Summing R is exact, not an approximation: apply step is linear in R for
    a fixed (k, l, i_chi, i_ZD), so merging entries changes nothing.

    Parameters
    ----------
    k_idx, l_idx, i_chi, i_ZD, R : the raw event arrays
    Np : grid size, used to pack the four indices into one key

    Returns
    -------
    The same arrays, deduplicated and with zero-rate entries dropped.
    """
    if k_idx.size == 0:
        z_i = np.zeros(0, dtype=np.int64)
        return z_i, z_i.copy(), z_i.copy(), z_i.copy(), np.zeros(0, dtype=np.float64)

    k64 = k_idx.astype(np.int64)
    l64 = l_idx.astype(np.int64)
    ic64 = i_chi.astype(np.int64)
    iZD64 = i_ZD.astype(np.int64)
    flat = ((k64 * Np + l64) * Np + ic64) * Np + iZD64

    uniq, inv = np.unique(flat, return_inverse=True)

    # Sum the rates for events that share the same (k, l, i_chi, i_ZD) tuple.
    R_summed = np.bincount(inv, weights=R, minlength=uniq.size)

    nz = R_summed != 0.0
    uniq = uniq[nz]
    R_summed = R_summed[nz]

    iZD_out  = (uniq % Np).astype(np.int64)
    ichi_out = ((uniq // Np) % Np).astype(np.int64)
    l_out   = ((uniq // (Np * Np)) % Np).astype(np.int64)
    k_out   = (uniq // (Np * Np * Np)).astype(np.int64)
    return k_out, l_out, ichi_out, iZD_out, R_summed.astype(np.float64)


# ===========================================================================
# Run-time helpers -- called every solver step
# ===========================================================================

@jit(nopython=True, cache=True)
def _apply_xZD_delta_numba(k_idx, l_idx, i_chi, i_ZD, R, invW, f_chi, f_ZD, Np):
    """
    Apply the per-event delta operator. For each event (k, l, i_chi, i_ZD, R):

        delta          = (R/2) * (f_chi[k] f_ZD[l] - f_chi[i_chi] f_ZD[i_ZD])
        df_chi[k]     -= delta / W[k]        # chi leaves the incoming bin
        df_chi[i_chi] += delta / W[i_chi]    # and arrives in the outgoing one
        df_ZD[l]       -= delta / W[l]
        df_ZD[i_ZD]     += delta / W[i_ZD]

    Two properties follow: Delta vanishes whenever the
    distribution is balanced across the event, so discrete detailed balance
    holds per event rather. And every event removes exactly as much as it adds, so
    particle number is conserved bin-to-bin.

    The factor 1/2: delta includes the reverse process explicitly, but the
    cache also contains every reverse configuration as its own event, so
    without it each process is counted twice.

    The result is per dof, with no partner-dof factor; see
    `collisions.xZD_elastic_collision_rhs` for  prefactors.

    Runs once per Newton iteration over the full deduplicated event list.
    """
    df_chi = np.zeros(Np)
    df_ZD   = np.zeros(Np)

    # loop over each event, recalling that len(k_idx) == number of events
    for i in range(k_idx.shape[0]):
        # k, l = indices of the incoming momentum bins for chi and Z_D, respectively
        k  = k_idx[i]
        l  = l_idx[i]

        # ic, iZD = indices of the outgoing momentum bins for chi and Z_D, respectively
        ic = i_chi[i]
        iZD = i_ZD[i]

        # index into the current distributions to compute the delta for this event
        delta = 0.5 * R[i] * (f_chi[k] * f_ZD[l] - f_chi[ic] * f_ZD[iZD])
        df_chi[k]  -= delta * invW[k]  # remove from incoming chi bin
        df_chi[ic] += delta * invW[ic] # add to outgoing chi bin
        df_ZD[l]    -= delta * invW[l]  # remove from incoming Z_D bin
        df_ZD[iZD]   += delta * invW[iZD] # add to outgoing Z_D bin
    return df_chi, df_ZD


def apply_xZD_cache(cache, f_chi, f_ZD):
    """
    Compute the full dark Compton collision term for both species.

    Parameters
    ----------
    cache        : dict from `build_xZD_elastic_gain_cache`
    f_chi, f_ZD   : current distributions on the cache's `p_grid`

    Returns
    -------
    (df_chi, df_ZD), each shape (Np,). Sums to zero over each species.
    """
    Np = cache["p_grid"].shape[0]
    ev = cache["events"]
    if ev["R"].size == 0:
        return np.zeros(Np), np.zeros(Np)
    f_chi = np.ascontiguousarray(f_chi, dtype=np.float64)
    f_ZD   = np.ascontiguousarray(f_ZD,   dtype=np.float64)
    return _apply_xZD_delta_numba(
        ev["k_idx"], ev["l_idx"], ev["i_chi"], ev["i_ZD"], ev["R"],
        cache["invW"], f_chi, f_ZD, Np,
    )




def apply_xZD_elastic_loss(p_grid, dlogp, K_loss, f_chi, f_ZD):
    """
    Loss-only rate, for diagnostics.

        df_chi[i]/dt|loss = -f_chi[i] sum_j W[j] K_loss[i,j] f_ZD[j]
        df_ZD[j]/dt|loss   = -f_ZD[j]   sum_i W[i] K_loss[i,j] f_chi[i]

    `apply_xZD_cache` already includes the loss inside its per-event delta.
    The only caller is `solver.get_dark_compton_loss_term`.
    """
    W = (1.0 / (2.0 * np.pi * np.pi)) * p_grid * p_grid * p_grid * dlogp
    df_chi = -f_chi * (K_loss * (W * f_ZD)[None, :]).sum(axis=1)
    df_ZD   = -f_ZD   * (K_loss * (W * f_chi)[:, None]).sum(axis=0)
    return df_chi, df_ZD


# ===========================================================================
# Temperature-grid builder
# ===========================================================================

def build_xZDxZD_gain_caches(T_span, state, Nmu_in=8, Nmu_cm=8, Nphi_cm=4):
    """
    Tabulate the dark Compton cache on a temperature grid.

    Parameters
    ----------
    T_span : temperatures [GeV]
    state  : PhaseSpaceState, supplying grid, gstar_func, species, epsilon
             and alphaD (for g_ZDchi = gD cos(alpha))
    Nmu_in, Nmu_cm, Nphi_cm : passed through to the builder

    Returns
    -------
    list of cache dicts aligned with T_span. Assign to
    `state.gain_caches_xZDxZD`.

    """
    out = []
    n_T = len(T_span)
    m_chi = state.species["chi"].mass_GeV
    m_ZD   = state.species["ZD"].mass_GeV
    # chi-chi-Z_D coupling is fermion-independent; "e" is just a placeholder key
    g_ZDchi = compute_couplings(epsilon=state.epsilon, fermion="e", mZD=m_ZD,
                                alphaD=state.alphaD)["g_ZDchi"]
    for i, T in enumerate(T_span):
        T = float(T)
        p = state.grid.p_phys(T, gstars_func=state.gstar_func)
        cache_T = build_xZD_elastic_gain_cache(
            p_grid=p, dlogp=state.grid.dlogp,
            m_chi=m_chi, m_ZD=m_ZD, g_ZDchi=g_ZDchi,
            Nmu_in=Nmu_in, Nmu_cm=Nmu_cm, Nphi_cm=Nphi_cm,
        )
        out.append(cache_T)
        if i % max(1, n_T // 10) == 0:
            print(f"  xZD-gain [{i+1}/{n_T}]: T={T:.3e} GeV  "
                  f"N_events={cache_T['events']['R'].size}")
    return out



def _xZD_kernel_index(T, T_grid):
    """
    Index of the tabulated cache nearest to T.

    Nearest-neighbor, matching every other lookup in `dphase.kernels`.
    """
    return int(np.argmin(np.abs(np.asarray(T_grid) - float(T))))
