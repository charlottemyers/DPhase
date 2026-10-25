"""
Collision operators: the right-hand sides that the solver integrates.

Calling convention
------------------
Every `*_collision_rhs*` function has the signature

    f(T, y_flat, state, names, Np) -> ndarray of the same shape as y_flat

where `y_flat` is the flattened multi-species state vector described in
`dphase.solver`. Each operator writes df/dt into the slices of the species it
affects and exact zeros elsewhere, so the solver can simply add them.

A few conventions:

  * Kernels are looked up by nearest temperature, not interpolated.
  * The kernels are built with per-d.o.f. cross sections, so the
    degeneracy factors (g_chi, g_ZD) are applied here, at the rate level;
    see `ZDZD_chichi_collision_rhs`.
"""

import numpy as np

from dphase.constants import FERMIONS
from dphase.model import ZD_ff_decay_m2
from dphase.kernels import (
    gain_cache_from_grid_nearest, apply_gain_cache, kernel_from_grid_nearest,
    _xZD_kernel_index, apply_xZD_cache,
    fp_matrices_from_grid_nearest,
    heavy_partner_matrix_from_grid_nearest,
)

def loss_self_annih_dfdt(p, f, K, dlogp):
    """
    Loss part of a two-body annihilation term, for an isotropic distribution
    on a log-p grid:

        df_i/dt = -sum_j W_j K_ij f_i f_j,     W_j = p_j^3 dlogp_j / (2 pi^2)

    Loss ONLY -- there is no equilibrium term here. Use this where the gain is
    supplied separately (as a deposition cache), which is the case for
    chi chibar <-> Z_D Z_D; use `collision_annihilation_dfdt_from_kernel` where
    detailed balance against a thermal bath supplies the gain instead.

    W_j is the phase-space measure of the partner bin, so the sum over j
    is the integral over the annihilation partner's momentum.

    Parameters
    ----------
    p     : physical momenta [GeV], shape (Np,)
    f     : distribution on the grid, shape (Np,)
    K     : kernel <sigma v> matrix K_ij [GeV^-2], shape (Np, Np)
    dlogp : log-momentum bin widths, shape (Np,)

    Returns
    -------
    df/dt, shape (Np,), everywhere <= 0.
    """
    W = (1.0 / (2.0 * np.pi**2)) * p**3 * dlogp
    GL =  - (f[:, None] * f[None, :])  # (Np,Np)
    dfdt = np.sum((K * GL) * W[None, :], axis=1)
    return dfdt # shape (Np,)


# ===========================================================================
# Annihilation: chi chibar -> f fbar (into the SM bath) and chi chibar <-> Z_D Z_D
# ===========================================================================


def collision_annihilation_dfdt_from_kernel(p, f, feq, K, dlogp):
    """
    Gain and loss together for annihilation into a thermal bath:

        df_i/dt = sum_j W_j K_ij (feq_i feq_j - f_i f_j)

    The `feq_i feq_j` term is the inverse reaction, fixed by detailed balance:
    because the final state (SM fermions) stays in equilibrium at T, the gain
    is completely determined by the same kernel and the equilibrium
    distribution. The operator vanishes identically when f = feq.

    Parameters
    ----------
    p     : physical momenta [GeV], shape (Np,)
    f     : current distribution, shape (Np,)
    feq   : equilibrium distribution at T on the same grid, shape (Np,)
    K     : kernel <sigma v> matrix K_ij [GeV^-2], shape (Np, Np)
    dlogp : log-momentum bin widths, shape (Np,)

    Returns
    -------
    df/dt, shape (Np,). Negative where f exceeds equilibrium, positive below.
    """
    # phase-space weight for bin j in physical momentum:
    W = (1.0 / (2.0 * np.pi**2)) * p**3 * dlogp
    GL = (feq[:, None] * feq[None, :]) - (f[:, None] * f[None, :])  # (Np,Np)
    dfdt = np.sum((K * GL) * W[None, :], axis=1)
    return dfdt # shape (Np,)


def annihilation_collision_rhs_kernel(T, y_flat, state, names, Np,
                                      fermion_dist='MB', channels=("ff", "WW")):
    """
    Collision operator for chi chibar -> f fbar and chi chibar -> W+ W-,
    using the kernels tabulated on `state.T_grid`.

    Only chi is affected: the SM final states are assumed to stay in
    equilibrium at T, so their distributions are not evolved. Both channels
    relax chi toward the same feq, so their kernels simply add.

    Degeneracy factor: the kernels carry sigma averaged over the incoming
    chi chibar spins and f is per dof, so the rate picks up the dof of the
    chibar partner, g_chi (as `chi_loss_pref` in `ZDZD_chichi_collision_rhs`).

    Parameters
    ----------
    T, y_flat, state, names, Np : see the module docstring
    fermion_dist : {"FD", "MB"}
        Statistics used for the equilibrium chi distribution that the
        annihilation relaxes toward. "MB" (Maxwell-Boltzmann) is the standard choice
    channels : which kernels to include, any of "ff" and "WW". "WW" is used
        only if `state.K_grid_xxWW` has been set.

    Returns
    -------
    df/dt for the whole state vector, nonzero only in the chi slice.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.empty_like(y)

    p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
    chi = state.species["chi"]
    E_phys = np.sqrt(p_phys**2 + chi.mass_GeV**2)

    if fermion_dist == 'FD':
        feq = 1.0 / (np.exp(E_phys / T) + 1.0)
    elif fermion_dist == 'MB':
        feq = np.exp(-E_phys / T)
    else:
        raise ValueError(f"Unknown fermion distribution: {fermion_dist}")

    K = np.zeros((Np, Np))
    if "ff" in channels:
        K = K + kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxff)
    if "WW" in channels and state.K_grid_xxWW is not None:
        K = K + kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxWW)

    for i, name in enumerate(names):
        sl = slice(i*Np, (i+1)*Np)
        if name == "chi":
            f = y[sl]
            out[sl] = chi.dof * collision_annihilation_dfdt_from_kernel(
                p_phys, f, feq, K, state.grid.dlogp)
        else:
            out[sl] = 0.0 * y[sl]
    return out




def ZDZD_chichi_collision_rhs(T, y_flat, state, names, Np):
    """
    Collision operator for chi chibar <-> Z_D Z_D, both directions, both species.

    Unlike annihilation into the SM bath, neither side of this reaction is held
    in equilibrium; both chi and Z_D are being evolved. So detailed balance
    cannot supply the gain term, and each species needs two separate objects:

      loss : kernel, contracted by `loss_self_annih_dfdt`
      gain : deposition event cache, contracted by `apply_gain_cache`

    Degeneracy factors
    ------------------
    Both kernels carry cross sections that are averaged over the internal
    states of the incoming pair and summed over those of the outgoing pair,
    with the 1/2 for an identical outgoing pair already included. The
    distributions `f` are likewise per dof. Neither convention tracks how many
    internal states the collision partner has, so those factors
    are supplied here at the rate level.

    Loss. Writing the Boltzmann loss term for one internal state of species 1,

        df_1/dt|loss = - g_2 Int d^3p_2/(2 pi)^3 sigma v_mol f_1 f_2

    The momentum integral runs over the partner's momenta but not over its
    internal states, while sigma was averaged over them; hence one factor of
    the partner's dof:

        chi_loss_pref = gx   partner is chibar, g = gx           -> 2
        ZD_loss_pref   = gZD   partner is the other Z_D, g = gZD      -> 3

    Gain. The reaction rate per unit volume needs the dof of both incoming legs,
    and a 1/2 if they are identical to avoid double counting. The products are
    then spread over the internal states of the deposited species, so divide by its dof:

        chi_gain_pref = gZD^2 / (2 gx)   gZD^2 incoming states, 1/2 for the
                                        identical pair, / gx    -> 9/4
        ZD_gain_pref   = gx^2 / gZD       gx^2 incoming states, / gZD  -> 4/3

    Multiplicity of products is carried by the caches rather than by these
    prefactors, and assumes they were built with yield_out_chi = 1 (one chi per
    reaction, the chibar tracked by the same f) and yield_out_ZD = 2.

    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"]*Np, (idx["chi"]+1)*Np) # slice in the co-moving grid
    sl_ZD   = slice(idx["ZD"]  *Np, (idx["ZD"]  +1)*Np)

    floor = 1e-100
    fchi = np.maximum(y[sl_chi], floor)
    fZD   = np.maximum(y[sl_ZD],   floor)

    # Physical momenta at this temperature, accounting for redshift.
    p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    # --- loss: each species destroyed by annihilating with its own kind ---

    K_chi = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxZDZD)
    K_ZD   = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_ZDZDxx)

    dfchi_loss = loss_self_annih_dfdt(p_phys, fchi, K_chi, dlogp)
    dfZD_loss   = loss_self_annih_dfdt(p_phys, fZD,   K_ZD,   dlogp)

    # --- gain: each species deposited by the other one annihilating ---

    cache_T = gain_cache_from_grid_nearest(T, state.T_grid, state.gain_caches_xxZDZD)

    dfchi_gain = apply_gain_cache(cache_T["cache_chi_from_ZDZD"], fZD)
    dfZD_gain   = apply_gain_cache(cache_T["cache_ZD_from_xx"], fchi)

    gx, gZD = state.species["chi"].dof, state.species["ZD"].dof

    # See "Degeneracy factors" in the docstring
    chi_gain_pref = gZD**2 / (2*gx)
    chi_loss_pref = gx
    ZD_gain_pref   = gx**2 / gZD
    ZD_loss_pref   = gZD

    out[sl_chi] = chi_gain_pref*dfchi_gain + chi_loss_pref*dfchi_loss
    out[sl_ZD]   = ZD_gain_pref  *dfZD_gain   + ZD_loss_pref  *dfZD_loss
    return out


# ===========================================================================
# Decay: Z_D -> f fbar and its inverse
# ===========================================================================

def decay_affine_coeffs(T, state, Nmu=28, fermion_dist='MB'):
    """
    Affine coefficients of the Z_D -> f fbar collision term.
    The decay operator is affine in f_ZD and diagonal in momentum,

        df_ZD/dt[j] = S[j] - G[j] f_ZD[j],

    with S the inverse-decay source and G the decay rate (Pauli-blocked for
    FD). Both depend only on T, not on f, so they can be
    computed once per temperature and reused.

    For each momentum bin the decay products are integrated over their allowed
    lab-frame energy range, summed over every kinematically accessible SM
    fermion:

        MB:  bracket = ff*ffbar - f_ZD                  -> S ~ ff*ffbar, G ~ 1
        FD:  bracket = ff*ffbar - f_ZD (1-ff)(1-ffbar)  -> S ~ ff*ffbar, G ~ (1-ff)(1-ffbar)

    The per-fermion |M|^2 depends only on (epsilon, m_ZD), so it is computed
    once rather than per bin and node.

    Parameters
    ----------
    T     : SM temperature [GeV]
    state : PhaseSpaceState, supplying grid, gstar_func, epsilon and species
    Nmu   : Gauss-Legendre nodes for the fermion-energy integral
    fermion_dist : {"FD", "MB"}
        Statistics for the SM fermions. "FD" includes Pauli blocking in the
        inverse decay; "MB" drops it, which is consistent because
        exp(-Ef/T) exp(-Efbar/T) = exp(-E_ZD/T) already reproduces the correct
        equilibrium for f_ZD.

    Returns
    -------
    (S, G), each shape (Np,), [GeV] (S per dof of Z_D, G a rate).
    """
    T = float(T)
    mZD = state.species["ZD"].mass_GeV
    p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
    EZD = np.sqrt(p_phys**2 + mZD**2)
    Ef_nodes, weights = np.polynomial.legendre.leggauss(Nmu)

    # open channels
    channels = []
    for fname, fdata in FERMIONS.items():
        mf = fdata["mass_GeV"]
        # skip kinematically inaccessible fermions
        if mZD <= 2.0 * mf:
            continue
        Msq = ZD_ff_decay_m2(epsilon=state.epsilon, mZD=mZD, fermion=fname, mf=mf)
        channels.append((mf, fdata["Nc"], Msq))

    S = np.zeros_like(p_phys)
    G = np.zeros_like(p_phys)
    for j, (p_j, E_j) in enumerate(zip(p_phys, EZD)):
        gamma = E_j / mZD
        beta = p_j / E_j if E_j > 0 else 0.0
        # 1/(16 pi E p) from the 1->2 phase space after the angular integral
        prefactor = 1.0 / (16.0 * np.pi * E_j * max(p_j, 1e-30))
        S_j = 0.0
        G_j = 0.0
        for mf, Nc, Msq in channels:
            # p_star = momentum of the fermion in the Z_D rest frame
            p_star = 0.5 * np.sqrt(mZD**2 - 4.0*mf**2)
            # min and max fermion energies in the lab frame
            Ef_min = gamma * (mZD/2.0 - beta*p_star)
            Ef_max = gamma * (mZD/2.0 + beta*p_star)
            Ef_vals  = 0.5*(Ef_max - Ef_min)*Ef_nodes + 0.5*(Ef_max + Ef_min)
            jacobian = 0.5*(Ef_max - Ef_min)
            Efbar_vals = E_j - Ef_vals

            if fermion_dist == 'FD':
                ff    = 1.0 / (np.exp(Ef_vals    / T) + 1.0)
                ffbar = 1.0 / (np.exp(Efbar_vals / T) + 1.0)
                blk = (1.0 - ff) * (1.0 - ffbar)
            elif fermion_dist == 'MB':
                ff    = np.exp(-Ef_vals    / T)
                ffbar = np.exp(-Efbar_vals / T)
                blk = 1.0
            else:
                raise ValueError(f"Unknown fermion distribution: {fermion_dist}")

            w = prefactor * jacobian * Msq * Nc
            S_j += w * np.sum(weights * ff * ffbar)
            G_j += w * np.sum(weights * blk)
        S[j] = S_j
        G[j] = G_j
    return S, G


def _decay_affine_coeffs_cached(T, state, Nmu, fermion_dist):
    """
    decay_affine_coeffs, reused while T and the model are unchanged.

    Solver evaluates the RHS many times at the same T within one Newton
    solve, and (S, G) do not depend on f, so only the most recent entry is
    kept (on `state._decay_affine_cache`).
    """
    key = (float(T), int(Nmu), fermion_dist, state.epsilon,
           state.species["ZD"].mass_GeV, id(state.grid), id(state.gstar_func))
    cached = state._decay_affine_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    S, G = decay_affine_coeffs(T, state, Nmu=Nmu, fermion_dist=fermion_dist)
    state._decay_affine_cache = (key, (S, G))
    return S, G


def decay_collision_rhs_direct_allf(T, y_flat, state, names, Np, Nmu=28, fermion_dist = 'MB'):
    """
    1 -> 2 collision term for Z_D -> f fbar and its inverse, summed over every
    kinematically accessible SM fermion:

        df_ZD/dt = S - G f_ZD,

    with (S, G) from `decay_affine_coeffs`, computed once per temperature and
    reused across calls at the same T (see `_decay_affine_coeffs_cached`).
    The operator vanishes when f_ZD reaches equilibrium.

    Only the Z_D slice is nonzero: the SM fermions are held in equilibrium at T
    and are not evolved.

    Parameters
    ----------
    T, y_flat, state, names, Np : see module docstring
    Nmu, fermion_dist : see `decay_affine_coeffs`
    """
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    if "ZD" not in names:
        return out
    i = list(names).index("ZD")
    sl = slice(i * Np, (i + 1) * Np)

    S, G = _decay_affine_coeffs_cached(T, state, Nmu, fermion_dist)
    out[sl] = S - G * y[sl]
    return out


# ===========================================================================
# Elastic scattering: number-preserving, redistributes momentum only
# ===========================================================================

def elastic_collision_rhs(T, y_flat, state, names, Np):
    """
    Elastic scattering off light SM partners, chi B -> chi B (B = f, W), as a
    Fokker-Planck operator in conservative (Chang-Cooper) form:
    df/dt = M(T) @ f per species. Number-preserving: this term moves particles
    between momentum bins but changes no total

    Fokker-Planck form is a purely local operator in p -- all the integration
    has already been done to produce the gamma(T) coefficients (see
    `kernels.elastic_sm_light`).

    Requires `state.FP_matrix_grid` from `kernels.build_fp_matrices`.

    Returns
    -------
    df/dt for the whole state vector. Both chi and Z_D are affected, though
    `gamma_grid_ZD` is left at its zero default; Z_D f -> Z_D f is
    suppressed by eps^4 and is neglected in `kernels.elastic_sm_light`.
    """
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)
    M = fp_matrices_from_grid_nearest(T, state)
    if M is None:
        return out
    for i, name in enumerate(names):
        if name in M:
            sl = slice(i * Np, (i + 1) * Np)
            out[sl] = M[name] @ y[sl]
    return out


def heavy_partner_collision_rhs(T, y_flat, state, names, Np, partners=None):
    """
    Elastic scattering off heavy SM partners (e.g. tops, Ws), chi B -> chi B,
    with the full 2 -> 2 operator rather than Fokker-Planck. B is held in
    equilibrium at T, so df_chi/dt = C(T) @ f_chi.

    Requires `state.C_grid_heavy` from `kernels.build_heavy_partner_matrices`.

    partners : names to include (default: all in state.C_grid_heavy)
    """
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    C = heavy_partner_matrix_from_grid_nearest(T, state, partners)
    if C is None:
        return out

    i_chi = list(names).index("chi")
    sl = slice(i_chi * Np, (i_chi + 1) * Np)
    out[sl] = C @ y[sl]
    return out


def xZD_elastic_collision_rhs(T, y_flat, state, names, Np):
    """
    Elastic chi Z_D -> chi Z_D "dark Compton" scattering, for both species.

    Unlike `elastic_collision_rhs`, this is a full 2 -> 2 redistribution rather
    than a Fokker-Planck approximation, because both scattering partners are
    hidden-sector particles of comparable mass so the small-momentum-transfer
    assumption that justifies FP does not hold.

    Number-preserving for both species: chi and Z_D are only moved between
    momentum bins.

    Requires `state.gain_caches_xZDxZD`, a list of event caches per entry in
    `state.T_grid`, built by `kernels.elastic_hidden`.

    Degeneracy factors: the cache gives a per-dof rate with sigma averaged
    over the incoming internal states, so each species picks up the dof of
    its collision partner:

        df_chi/dt *= g_ZD   (3),     df_ZD/dt *= g_chi   (2)

    Returns
    -------
    df/dt for the whole state vector.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    idx = {nm: i for i, nm in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_ZD   = slice(idx["ZD"]   * Np, (idx["ZD"]   + 1) * Np)

    f_chi = y[sl_chi]
    f_ZD   = y[sl_ZD]

    T_grid =  state.T_grid
    i_T = _xZD_kernel_index(T, T_grid)
    cache  = state.gain_caches_xZDxZD[i_T]

    # The cache stores a per-event delta, so one pass yields gain AND loss
    # together; i.e. there is no separate K_loss contraction here
    df_chi_total, df_ZD_total = apply_xZD_cache(cache, f_chi, f_ZD)
    out[sl_chi] = state.species["ZD"].dof * df_chi_total
    out[sl_ZD]   = state.species["chi"].dof * df_ZD_total
    return out
