"""
Elastic scattering off heavy SM partners, chi B -> chi B, with the full 2 -> 2
operator instead of Fokker-Planck. B = SM fermion or W.

Fokker-Planck (kernels.elastic_sm_light) assumes m_B << m_chi, so each
collision gives chi only a small kick. That fails for mchi ~ m_B.
Here the bath partner is held in equilibrium at T, so the
operator is linear in f_chi and is tabulated as a matrix:

    df_chi/dt = C_B(T) @ f_chi

Each partner is described by a small dict (make_partner_W /
make_partner_fermion) carrying its mass, statistics and the coupling
constants for `model._M2_xB_xB_scalar`.
"""

import numpy as np
from numba import njit

from dphase.constants import MW, FERMIONS
from dphase.model import (compute_couplings, _M2_xB_xB_scalar,
                          xf_coupling_consts, xW_coupling_consts,
                          PROC_W, PROC_FERMION)


# ---------------------------------------------------------------------------
# Partner specifications
# ---------------------------------------------------------------------------

def make_partner_W(epsilon, alphaD, mZD, mW=MW):
    """
    Partner spec for W+ and W- (Bose-Einstein), for the full heavy-partner
    operator.

    Parameters
    ----------
    epsilon : kinetic mixing
    alphaD  : dark coupling strength
    mZD     : dark photon mass [GeV]
    mW      : W mass [GeV]

    Returns
    -------
    dict with keys:
        name      : "W"
        m         : W mass [GeV]
        stat      : +1.0 (BE)
        n_species : 2.0, for W+ and W-
        proc      : model.PROC_W, selects the |M|^2 in `model._M2_xB_xB_scalar`
        consts    : coupling constants from `model.xW_coupling_consts`
        m_map2    : m_ZD^2 [GeV^2], the t-channel pole used for t-mapping
    """
    # chi-W couplings are fermion-independent; "e" is a placeholder key
    params = compute_couplings(epsilon=epsilon, fermion="e", mZD=mZD,
                               alphaD=alphaD)
    return dict(name="W", m=float(mW), stat=1.0, n_species=2.0,
                proc=PROC_W,
                consts=xW_coupling_consts(params),
                m_map2=float(params["m_ZD"]**2))


def make_partner_fermion(fermion, epsilon, alphaD, mZD):
    """
    Partner spec for an SM fermion f and its antiparticle fbar
    (Fermi-Dirac), for the full heavy-partner operator.

    Parameters
    ----------
    fermion : key into `constants.FERMIONS`, e.g. "t"
    epsilon : kinetic mixing
    alphaD  : dark coupling strength
    mZD     : dark photon mass [GeV]

    Returns
    -------
    dict with keys:
        name      : the fermion key
        m         : fermion mass [GeV]
        stat      : -1.0 (Fermi-Dirac)
        n_species : 2.0, for f and fbar
        proc      : model.PROC_FERMION, selects the |M|^2 in `model._M2_xB_xB_scalar`
        consts    : coupling constants from `model.xf_coupling_consts`
                    (Nc included)
        m_map2    : m_ZD^2 [GeV^2], the t-channel pole used for t-mapping
    """
    params = compute_couplings(epsilon=epsilon, fermion=fermion, mZD=mZD,
                               alphaD=alphaD)
    Nc = float(FERMIONS[fermion]["Nc"])
    return dict(name=fermion, m=float(params["mf"]), stat=-1.0,
                n_species=2.0, proc=PROC_FERMION,
                consts=xf_coupling_consts(params, mZD, Nc),
                m_map2=float(mZD**2))


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

@njit(inline='always')
def _deposit(C, loss, k, ic, iB, l, R, fB, stat, invW, E_chi, T):
    """
    Add one sub-event (k, l) -> (ic, iB) with rate R to the matrix.

    Note: The delta form R (a f_k - b f_ic) includes the reverse process
    explicitly, but the event loop runs over ALL initial (k, l), so the
    reverse configuration is also visited as its own event. Each direction
    is therefore weighted by R/2 here.

    The reverse weight b = a exp((E_ic - E_k)/T) is fixed by detailed
    balance on the binned equilibrium f, so e^{-E/T} is an exact zero of C.

    Parameters
    ----------
    C      : (Np, Np) operator being accumulated, modified in place
    loss   : (Np,) scattering-out rate being accumulated, modified in place
    k, ic  : incoming and outgoing chi bins
    l, iB  : incoming and outgoing B bins
    R      : rate of this sub-event (quadrature weight x bilinear weight)
    fB     : (Np,) equilibrium occupancy of B on the grid
    stat   : B statistics, +1 BE, -1 FD, 0 MB (sets the Bose enhancement /
             Pauli blocking factor 1 + stat f_B)
    invW   : (Np,) 1 / bin phase-space measure
    E_chi  : (Np,) chi energies on the grid [GeV]
    T      : bath temperature [GeV]

    Returns
    -------
    None; C and loss are updated in place.
    """
    a = fB[l] * (1.0 + stat * fB[iB])     # forward:  chi_k B_l -> chi_ic B_iB
    x = (E_chi[ic] - E_chi[k]) / T
    if x > 700.0:
        x = 700.0
    b = a * np.exp(x)                      # reverse, balanced on binned f_eq
    Rh = 0.5 * R
    C[k, k]   -= Rh * a * invW[k]
    C[k, ic]  += Rh * b * invW[k]
    C[ic, k]  += Rh * a * invW[ic]
    C[ic, ic] -= Rh * b * invW[ic]
    loss[k]   += R * a * invW[k]           # physical scattering rate


@njit(cache=True)
def _build_heavy_matrix_inner(
    p_grid, E_chi, E_B, PS, invW, fB, stat,
    mu_in_n, w_in_n, x_cm_n, w_cm_n, cos_phi_n, w_phi_n,
    m_chi, m_B, proc, consts, m2_scale, t_mapped, m_map2, T_bath,
):
    """
    Same event loop as elastic_hidden._build_xZD_elastic_cache_inner, but
    accumulating directly into C (Np, Np) since f_B is fixed.

    Parameters
    ----------
    p_grid             : shared physical momentum grid for chi and B [GeV]
    E_chi, E_B         : on-shell energies of each species on that grid [GeV]
    PS                 : phase-space measure p^3 dlogp / (2 pi^2)
    invW               : 1 / PS
    fB                 : equilibrium occupancy of B on the grid
    stat               : B statistics, +1 BE, -1 FD, 0 MB
    mu_in_n, w_in_n    : GL nodes/weights for the lab-frame angle
                         between the two incoming momenta
    x_cm_n, w_cm_n     : GL nodes/weights on [-1, 1] for the CM angle, used
                         directly as cos(theta_CM) or mapped (see t_mapped)
    cos_phi_n, w_phi_n : nodes/weights for the azimuth about the boost axis
    m_chi, m_B         : masses [GeV]
    proc, consts       : process and coupling constants
    m2_scale           : n_species / g_chi, the number of B species divided
                         by the chi dof
    t_mapped           : sample the CM angle uniformly in w = 1/(m_map2 - t)
                         instead of cos(theta_CM), which resolves the forward peak
    m_map2             : mass^2 of the t-channel pole used for the mapping [GeV^2]
    T_bath             : bath temperature [GeV]

    Returns
    -------
    (C, loss): C shape (Np, Np), with df_chi/dt = C @ f_chi [GeV];
    loss shape (Np,), with loss[k] the scattering-out rate of a chi in bin k [GeV].
    """
    Np = p_grid.shape[0]
    Nmu_in = mu_in_n.shape[0]
    Nmu_cm = x_cm_n.shape[0]
    Nphi = cos_phi_n.shape[0]

    C = np.zeros((Np, Np))
    loss = np.zeros(Np)

    m2_chi = m_chi * m_chi
    m2_B = m_B * m_B
    thresh_sq = (m_chi + m_B) * (m_chi + m_B)
    diff_sq = (m_chi - m_B) * (m_chi - m_B)
    inv_64pi2 = 1.0 / (64.0 * np.pi * np.pi)

    p_lo = p_grid[0]
    p_hi = p_grid[-1]

    mu_c_arr = np.empty(Nmu_cm)
    w_c_arr = np.empty(Nmu_cm)

    for k in range(Np):
        p_chi_k = p_grid[k]
        E_chi_k = E_chi[k]
        for l in range(Np):
            p_B_l = p_grid[l]
            E_B_l = E_B[l]
            pair_pref = PS[k] * PS[l]
            E_tot = E_chi_k + E_B_l

            for mi in range(Nmu_in):
                mu_i = mu_in_n[mi]
                wmu_i = w_in_n[mi]

                s = m2_chi + m2_B + 2.0 * (E_chi_k * E_B_l - p_chi_k * p_B_l * mu_i)
                if s <= thresh_sq:
                    continue
                lam_in = (s - thresh_sq) * (s - diff_sq)
                if lam_in <= 0.0:
                    continue
                sqrt_s = np.sqrt(s)
                p_cm = np.sqrt(lam_in) / (2.0 * sqrt_s)
                p_cm2 = p_cm * p_cm

                E_chi_star = (s + m2_chi - m2_B) / (2.0 * sqrt_s)
                E_B_star = (s + m2_B - m2_chi) / (2.0 * sqrt_s)
                v_Mol = np.sqrt(lam_in) / (2.0 * E_chi_k * E_B_l)

                # boost from CM to lab
                P_tot_sq = p_chi_k * p_chi_k + p_B_l * p_B_l + 2.0 * p_chi_k * p_B_l * mu_i
                if P_tot_sq <= 0.0:
                    gamma = 1.0
                    beta = 0.0
                    cos_psi = 1.0
                    sin_psi = 0.0
                else:
                    P_tot = np.sqrt(P_tot_sq)
                    beta = P_tot / E_tot
                    gamma = E_tot / sqrt_s
                    p_chi_dot_b = p_chi_k * (p_chi_k + p_B_l * mu_i) / P_tot
                    p_chi_par_cm = gamma * (p_chi_dot_b - beta * E_chi_k)
                    cos_psi = p_chi_par_cm / p_cm
                    if cos_psi > 1.0:
                        cos_psi = 1.0
                    elif cos_psi < -1.0:
                        cos_psi = -1.0
                    sin_psi_sq = 1.0 - cos_psi * cos_psi
                    if sin_psi_sq < 0.0:
                        sin_psi_sq = 0.0
                    sin_psi = np.sqrt(sin_psi_sq)

                rate_base = pair_pref * v_Mol * 0.5 * wmu_i * inv_64pi2 / s * m2_scale

                # CM-angle nodes for this s: cos(theta_CM) or t-mapped
                if t_mapped and m_map2 > 0.0:
                    w_lo = 1.0 / (m_map2 + 4.0 * p_cm2)
                    w_hi = 1.0 / m_map2
                    w_mid = 0.5 * (w_hi + w_lo)
                    w_half = 0.5 * (w_hi - w_lo)
                    for mc in range(Nmu_cm):
                        wv = w_mid + w_half * x_cm_n[mc]
                        t_n = m_map2 - 1.0 / wv
                        mu_c_arr[mc] = 1.0 + t_n / (2.0 * p_cm2)
                        # dmu = dt / (2 p_cm^2),  dt = dw / w^2
                        w_c_arr[mc] = w_half * w_cm_n[mc] / (wv * wv * 2.0 * p_cm2)
                else:
                    for mc in range(Nmu_cm):
                        mu_c_arr[mc] = x_cm_n[mc]
                        w_c_arr[mc] = w_cm_n[mc]

                for mc in range(Nmu_cm):
                    mu_c = mu_c_arr[mc]
                    w_c = w_c_arr[mc]

                    t_val = -2.0 * p_cm2 * (1.0 - mu_c)
                    M2 = _M2_xB_xB_scalar(proc, s, t_val, m2_chi, m2_B, consts)
                    if not np.isfinite(M2) or M2 <= 0.0:
                        continue

                    sin_th_sq = 1.0 - mu_c * mu_c
                    if sin_th_sq < 0.0:
                        sin_th_sq = 0.0
                    sin_th = np.sqrt(sin_th_sq)

                    for mp in range(Nphi):
                        cos_phi = cos_phi_n[mp]
                        w_p = w_phi_n[mp]

                        mu_b = sin_th * cos_phi * sin_psi + mu_c * cos_psi

                        E3 = gamma * (E_chi_star + beta * p_cm * mu_b)
                        p3_sq = E3 * E3 - m2_chi
                        if p3_sq <= 0.0:
                            continue
                        p3 = np.sqrt(p3_sq)
                        if p3 <= p_lo or p3 >= p_hi:
                            continue

                        E4 = gamma * (E_B_star - beta * p_cm * mu_b)
                        p4_sq = E4 * E4 - m2_B
                        if p4_sq <= 0.0:
                            continue
                        p4 = np.sqrt(p4_sq)
                        if p4 <= p_lo or p4 >= p_hi:
                            continue

                        rate_event = rate_base * M2 * w_c * w_p

                        # bilinear deposit over (chi bin) x (B bin), linear in energy
                        ic_lo = np.searchsorted(p_grid, p3) - 1
                        if ic_lo < 0:
                            ic_lo = 0
                        if ic_lo > Np - 2:
                            ic_lo = Np - 2
                        al_c = (E3 - E_chi[ic_lo]) / (E_chi[ic_lo + 1] - E_chi[ic_lo])
                        if al_c < 0.0:
                            al_c = 0.0
                        elif al_c > 1.0:
                            al_c = 1.0

                        iB_lo = np.searchsorted(p_grid, p4) - 1
                        if iB_lo < 0:
                            iB_lo = 0
                        if iB_lo > Np - 2:
                            iB_lo = Np - 2
                        al_B = (E4 - E_B[iB_lo]) / (E_B[iB_lo + 1] - E_B[iB_lo])
                        if al_B < 0.0:
                            al_B = 0.0
                        elif al_B > 1.0:
                            al_B = 1.0

                        _deposit(C, loss, k, ic_lo, iB_lo, l,
                                 (1.0 - al_c) * (1.0 - al_B) * rate_event,
                                 fB, stat, invW, E_chi, T_bath)
                        _deposit(C, loss, k, ic_lo, iB_lo + 1, l,
                                 (1.0 - al_c) * al_B * rate_event,
                                 fB, stat, invW, E_chi, T_bath)
                        _deposit(C, loss, k, ic_lo + 1, iB_lo, l,
                                 al_c * (1.0 - al_B) * rate_event,
                                 fB, stat, invW, E_chi, T_bath)
                        _deposit(C, loss, k, ic_lo + 1, iB_lo + 1, l,
                                 al_c * al_B * rate_event,
                                 fB, stat, invW, E_chi, T_bath)

    return C, loss


def bath_occupancy(E, T, stat):
    """
    Per-state equilibrium occupancy of a bath species with zero chemical potential.

    Parameters
    ----------
    E    : energies [GeV], scalar or array
    T    : temperature [GeV]
    stat : statistics, +1 Bose-Einstein, -1 Fermi-Dirac, 0 Maxwell-Boltzmann

    Returns
    -------
    occupancy f(E), same shape as E. E/T is clipped at 700 to avoid overflow.
    """
    x = np.minimum(E / T, 700.0)
    if stat == 0.0:
        return np.exp(-x)
    return 1.0 / (np.exp(x) - stat)


def build_heavy_partner_matrix(p_grid, dlogp, T, m_chi, partner, g_chi=2.0,
                               Nmu_in=8, Nmu_cm=8, Nphi_cm=4, t_mapped=True,
                               stat=None):
    """
    C_B(T) such that df_chi/dt = C_B @ f_chi for chi B -> chi B elastic
    scattering, B in equilibrium at T.

    Parameters
    ----------
    p_grid, dlogp : physical momentum grid at T [GeV] and its log bin widths
    T             : bath temperature [GeV]
    m_chi         : chi mass [GeV]
    partner       : dict from make_partner_W / make_partner_fermion
    g_chi         : chi internal dof
    Nmu_in        : GL nodes for the lab-frame incoming angle
    Nmu_cm        : GL nodes for the CM scattering angle
    Nphi_cm       : nodes for the azimuth. Uniform rather than GL.
    t_mapped      : sample the CM angle uniformly in 1/(m_map2 - t) to resolve
                    the forward peak, rather than uniformly in cos(theta_CM)
    stat          : override the partner's statistics (+1 BE, -1 FD, 0 MB);
                    None uses partner["stat"]

    Returns
    -------
    dict with keys:
        C      : (Np, Np) operator, df_chi/dt = C @ f_chi [GeV]
        loss   : (Np,) rate at which a chi in bin k scatters out [GeV],
                 DIAGNOSTIC ONLY
        p_grid : the momentum grid used
        T      : the temperature used
        name   : partner["name"]
    """
    p_grid = np.ascontiguousarray(p_grid, dtype=np.float64)
    dlogp = np.ascontiguousarray(dlogp, dtype=np.float64)
    m_B = partner["m"]
    stat = float(partner["stat"] if stat is None else stat)

    E_chi = np.sqrt(p_grid**2 + m_chi**2)
    E_B = np.sqrt(p_grid**2 + m_B**2)
    PS = (1.0 / (2.0 * np.pi**2)) * p_grid**3 * dlogp
    invW = 1.0 / np.maximum(PS, 1e-300)
    fB = bath_occupancy(E_B, T, stat)

    mu_in_n, w_in_n = np.polynomial.legendre.leggauss(Nmu_in)
    x_cm_n, w_cm_n = np.polynomial.legendre.leggauss(Nmu_cm)
    phi_nodes = 2.0 * np.pi * (np.arange(Nphi_cm) + 0.5) / Nphi_cm
    cos_phi_n = np.cos(phi_nodes)
    w_phi_n = np.full(Nphi_cm, 2.0 * np.pi / Nphi_cm)

    m2_scale = partner["n_species"] / g_chi

    C, loss = _build_heavy_matrix_inner(
        p_grid, E_chi, E_B, PS, invW, fB, stat,
        mu_in_n, w_in_n, x_cm_n, w_cm_n, cos_phi_n, w_phi_n,
        float(m_chi), float(m_B), int(partner["proc"]),
        np.ascontiguousarray(partner["consts"], dtype=np.float64),
        float(m2_scale), bool(t_mapped), float(partner["m_map2"]), float(T),
    )
    return dict(C=C, loss=loss, p_grid=p_grid, T=float(T),
                name=partner["name"])


# ---------------------------------------------------------------------------
# T-grid builder and run-time lookup
# ---------------------------------------------------------------------------

def build_heavy_partner_matrices(T_grid, state, partners, **kwargs):
    """
    Precompute the heavy-partner matrices C_B(T) on a temperature grid, for
    each partner. Build on state.T_grid and assign
        state.C_grid_heavy = build_heavy_partner_matrices(state.T_grid, state, partners)

    Parameters
    ----------
    T_grid   : array of temperatures [GeV]
    state    : PhaseSpaceState; uses state.grid, state.gstar_func and
               state.species['chi'] (mass and dof)
    partners : list of dicts from make_partner_W / make_partner_fermion
    **kwargs : forwarded to build_heavy_partner_matrix (Nmu_in, Nmu_cm,
               Nphi_cm, t_mapped, stat)

    Returns
    -------
    dict {partner name: array (NT, Np, Np)} of C_B(T) [GeV]
    """
    m_chi = state.species["chi"].mass_GeV
    g_chi = state.species["chi"].dof
    out = {pt["name"]: [] for pt in partners}
    n_T = len(T_grid)
    for i, T in enumerate(T_grid):
        T = float(T)
        p = state.grid.p_phys(T, gstars_func=state.gstar_func)
        for pt in partners:
            res = build_heavy_partner_matrix(p, state.grid.dlogp, T, m_chi, pt,
                                             g_chi=g_chi, **kwargs)
            out[pt["name"]].append(res["C"])
        if i % max(1, n_T // 10) == 0:
            print(f"  heavy-partner [{i+1}/{n_T}]: T={T:.3e} GeV")
    return {name: np.array(Cs) for name, Cs in out.items()}


def heavy_partner_matrix_from_grid_nearest(T, state, partners=None):
    """
    Summed chi matrix C(T) = sum_B C_B(T) at the nearest grid temperature.

    Parameters
    ----------
    T        : temperature to evaluate at [GeV]
    state    : PhaseSpaceState; uses state.C_grid_heavy and state.T_grid
    partners : iterable of names to include (default: all in C_grid_heavy)

    Returns
    -------
    (Np, Np) array C(T) [GeV], or None if state.C_grid_heavy is not set.
    """
    C_grid = state.C_grid_heavy
    if not C_grid:
        return None

    i_T = int(np.argmin(np.abs(np.asarray(state.T_grid) - float(T))))

    if partners is None:
        partners = C_grid.keys()
    return sum(C_grid[name][i_T] for name in partners)
