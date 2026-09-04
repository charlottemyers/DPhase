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

  * Kernels are looked up by nearest temperature, not interpolated. Accuracy
    comes from tabulating `state.T_grid` finely.
  * The kernels are built with per-d.o.f. cross sections, so the
    degeneracy factors (g_chi, g_A) are applied here, at the rate level --
    see `AA_chichi_collision_rhs`.
"""

import numpy as np

from dphase.constants import FERMIONS
from dphase.model import A_ff_decay_m2
from dphase.kernels import (
    gain_cache_from_grid_nearest, apply_gain_cache, kernel_from_grid_nearest,
    _xA_kernel_index, apply_xA_cache, gamma_from_grid
)

def loss_self_annih_dfdt(p, f, K, dlogp):
    """
    Loss half of a two-body annihilation term, for an isotropic distribution
    on a log-p grid:

        df_i/dt = -sum_j W_j K_ij f_i f_j,     W_j = p_j^3 dlogp_j / (2 pi^2)

    Loss ONLY -- there is no equilibrium term here. Use this where the gain is
    supplied separately (as a deposition cache), which is the case for
    chi chibar <-> A A; use `collision_annihilation_dfdt_from_kernel` where
    detailed balance against a thermal bath supplies the gain instead.

    Bilinear in f because two particles must meet. W_j is the phase-space
    measure of the partner bin, so the sum over j is the integral over the
    annihilation partner's momentum.

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
    ## sum over annihilation partner j
    dfdt = np.sum((K * GL) * W[None, :], axis=1)
    return dfdt # shape (Np,)


# ===========================================================================
# Annihilation: chi chibar -> f fbar (into the SM bath) and chi chibar <-> A A
# (internal to the dark sector)
# ===========================================================================


def collision_annihilation_dfdt_from_kernel(p, f, feq, K, dlogp):
    """
    Gain and loss together for annihilation into a thermal bath:

        df_i/dt = sum_j W_j K_ij (feq_i feq_j - f_i f_j)

    The `feq_i feq_j` term is the inverse reaction, fixed by detailed balance:
    because the final state (SM fermions) stays in equilibrium at T, the gain
    is completely determined by the same kernel and the equilibrium
    distribution, with no separate deposition cache needed. The operator
    vanishes identically when f = feq.

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


def annihilation_collision_rhs_kernel(T, y_flat, state, names, Np, fermion_dist = 'FD'):
    """
    Collision operator for chi chibar -> f fbar, using the kernel tabulated on
    `state.T_grid`.

    Only chi is affected: the SM fermions are assumed to stay in equilibrium at
    T, so their distributions are not evolved.

    Parameters
    ----------
    T, y_flat, state, names, Np : see the module docstring
    fermion_dist : {"FD", "MB"}
        Statistics used for the *equilibrium chi* distribution that the
        annihilation relaxes toward. "MB" (Maxwell-Boltzmann) is the usual choice

    Returns
    -------
    df/dt for the whole state vector, nonzero only in the chi slice.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.empty_like(y)

    p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
    sp = state.species["chi"]
    E_phys = np.sqrt(p_phys**2 + sp.mass_GeV**2)

    if fermion_dist == 'FD':
        feq = 1.0 / (np.exp(E_phys / T) + 1.0)
    elif fermion_dist == 'MB':
        feq = np.exp(-E_phys / T)
    else:
        raise ValueError(f"Unknown fermion distribution: {fermion_dist}")
    K = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxff)

    for i, name in enumerate(names):
        sl = slice(i*Np, (i+1)*Np)
        if name == "chi":
            f = y[sl]
            out[sl] = collision_annihilation_dfdt_from_kernel(
                p_phys, f, feq, K, state.grid.dlogp)
        else:
            out[sl] = 0.0 * y[sl]
    return out




def AA_chichi_collision_rhs(T, y_flat, state, names, Np):
    """
    Collision operator for chi chibar <-> A A, both directions, both species.

    Unlike annihilation into the SM bath, neither side of this reaction is held
    in equilibrium -- both chi and A are being evolved. So detailed balance
    cannot supply the gain term, and each species needs two separate objects:

      loss : dense kernel, contracted by `loss_self_annih_dfdt`
      gain : deposition event cache, contracted by `apply_gain_cache`

    Note the crossing in the gain terms: chi is *gained* from AA annihilating,
    so `dfchi_gain` is driven by `fA`, and vice versa.

    Degeneracy factors
    ------------------
    Both kernels carry cross sections that are averaged over the internal
    states of the incoming pair and summed over those of the outgoing pair,
    with the 1/2 for an identical outgoing pair already included. The
    distributions `f` are likewise per dof. Neither convention knows how many
    internal states the collision partner has, so those factors
    are supplied here at the rate level.

    Loss. Writing the Boltzmann loss term for one internal state of species 1,

        df_1/dt|loss = - g_2 Int d^3p_2/(2 pi)^3 sigma v_mol f_1 f_2

    The momentum integral runs over the partner's momenta but not over its
    internal states, while sigma was averaged over them -- hence one factor of
    the partner's dof:

        chi_loss_pref = gx   partner is chibar, g = gx           -> 2
        A_loss_pref   = gA   partner is the other A, g = gA      -> 3

    Gain. The natural object is the reaction rate per unit volume, which
    needs the dof of both incoming legs, and a 1/2 if they are identical to
    avoid counting each pair twice. The products are then spread over the
    internal states of the species being deposited into, so divide by its dof:

        chi_gain_pref = gA^2 / (2 gx)   gA^2 incoming states, 1/2 for the
                                        identical A A pair, / gx to spread the
                                        one chi over its states     -> 9/4
        A_gain_pref   = gx^2 / gA       gx^2 incoming states, / gA  -> 4/3

    Multiplicity of products is carried by the caches rather than by these
    prefactors, and assumes they were built with yield_out_chi = 1 (one chi per
    reaction, the chibar tracked by the same f) and yield_out_A = 2.

    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"]*Np, (idx["chi"]+1)*Np) # slice in the co-moving grid
    sl_A   = slice(idx["A"]  *Np, (idx["A"]  +1)*Np)

    floor = 1e-100
    fchi = np.maximum(y[sl_chi], floor)
    fA   = np.maximum(y[sl_A],   floor)

    # Physical momenta at this temperature, accounting for redshift.
    p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    # --- loss: each species destroyed by annihilating with its own kind ---

    K_chi = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxAA)
    K_A   = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_AAxx)

    dfchi_loss = loss_self_annih_dfdt(p_phys, fchi, K_chi, dlogp)
    dfA_loss   = loss_self_annih_dfdt(p_phys, fA,   K_A,   dlogp)

    # --- gain: each species deposited by the OTHER one annihilating ---

    cache_T = gain_cache_from_grid_nearest(T, state.T_grid, state.gain_caches_xxAA)

    dfchi_gain = apply_gain_cache(cache_T["cache_chi_from_AA"], fA)
    dfA_gain   = apply_gain_cache(cache_T["cache_A_from_xx"], fchi)

    gx, gA = state.species["chi"].dof, state.species["A"].dof

    # See "Degeneracy factors" in the docstring
    chi_gain_pref = gA**2 / (2*gx)   # 9/4  incoming AA states, /2 identical pair, /gx
    chi_loss_pref = gx               # 2    dof of the chibar partner
    A_gain_pref   = gx**2 / gA       # 4/3  incoming chi chibar states, /gA
    A_loss_pref   = gA               # 3    dof of the other A

    out[sl_chi] = chi_gain_pref*dfchi_gain + chi_loss_pref*dfchi_loss
    out[sl_A]   = A_gain_pref  *dfA_gain   + A_loss_pref  *dfA_loss
    return out


# ===========================================================================
# Decay: A -> f fbar and its inverse
# ===========================================================================

def decay_collision_rhs_direct_allf(T, y_flat, state, names, Np, Nmu=28, fermion_dist = 'MB'):
    """
    1 -> 2 collision term for A -> f fbar and its inverse, summed over every
    kinematically accessible SM fermion.

    For each momentum bin the decay products are integrated over their allowed
    lab-frame energy range. The `bracket` is the inverse decay minus the decay,
    so the operator vanishes when f_A reaches equilibrium.

    Only the A slice is nonzero: the SM fermions are held in equilibrium at T
    and are not evolved.

    Parameters
    ----------
    T, y_flat, state, names, Np : see module docstring
    Nmu : Gauss-Legendre nodes for the fermion-energy integral
    fermion_dist : {"FD", "MB"}
        Statistics for the SM fermions. "FD" includes Pauli blocking in the
        inverse decay; "MB" drops it, which is consistent because
        exp(-Ef/T) exp(-Efbar/T) = exp(-E_A/T) already reproduces the correct
        equilibrium for f_A.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.empty_like(y)

    for i, name in enumerate(names):
        sl = slice(i * Np, (i + 1) * Np)

        # skip non-A species
        if name != "A":
            out[sl] = 0.0 * y[sl]
            continue

        fA = y[sl]
        sp = state.species[name]
        mA = sp.mass_GeV
        p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
        EA = np.sqrt(p_phys**2 + mA**2)

        Ef_nodes, weights = np.polynomial.legendre.leggauss(Nmu)
        dfdt = np.zeros_like(fA)

        for j, (p_j, E_j) in enumerate(zip(p_phys, EA)):
            gamma = E_j / mA
            beta = p_j / E_j if E_j > 0 else 0.0
            dfdt_j = 0.0

            for fname, fdata in FERMIONS.items():
                mf = fdata["mass_GeV"]
                Nc = fdata["Nc"]

                # skip kinematically inaccessible fermions
                if mA <= 2.0 * mf:
                    continue
                Msq_decay_f = A_ff_decay_m2(epsilon=state.epsilon, mA=mA, fermion=fname, mf=mf)

                # p_star = momentum of the fermion in the A rest frame
                p_star = 0.5 * np.sqrt(mA**2 - 4.0*mf**2)

                # Ef_min and Ef_max are the minimum and maximum fermion energies in the lab frame
                Ef_min = gamma * (mA/2.0 - beta*p_star)
                Ef_max = gamma * (mA/2.0 + beta*p_star)

                Ef_vals  = 0.5*(Ef_max - Ef_min)*Ef_nodes + 0.5*(Ef_max + Ef_min)
                jacobian = 0.5*(Ef_max - Ef_min)
                Efbar_vals = E_j - Ef_vals


                if fermion_dist == 'FD':
                    ff    = 1.0 / (np.exp(Ef_vals    / T) + 1.0)
                    ffbar = 1.0 / (np.exp(Efbar_vals / T) + 1.0)
                    bracket = ff * ffbar - fA[j] * (1.0 - ff) * (1.0 - ffbar)
                elif fermion_dist == 'MB':
                    ff    = np.exp(-Ef_vals    / T)
                    ffbar = np.exp(-Efbar_vals / T)
                    # No blocking: ff*ffbar = exp(-E_A/T) already gives correct equilibrium for f_A.
                    bracket = ff * ffbar - fA[j]
                else:
                    raise ValueError(
                        f"Unknown fermion distribution: {fermion_dist}")

                integrand = Msq_decay_f * Nc * bracket
                # 1/(16 pi E p) from the 1->2 phase space after the angular
                # integral; floor on p_j guards the p -> 0 bin
                prefactor = 1.0 / (16.0 * np.pi * E_j * max(p_j, 1e-30))

                dfdt_j += prefactor * jacobian * np.sum(weights * integrand)

            dfdt[j] = dfdt_j
        out[sl] = dfdt
    return out


# ===========================================================================
# Elastic scattering: number-preserving, redistributes momentum only
# ===========================================================================

def elastic_collision_rhs(T, y_flat, state, names, Np):
    """
    Elastic scattering off the SM bath, chi f -> chi f, as a Fokker-Planck
    operator. Number-preserving: this term moves particles between momentum bins
    but changes no total

    Fokker-Planck form is a purely LOCAL differential operator-- all the integration
    has already been done to produce the gamma(T) coefficients (see
    `kernels.elastic_sm`).

    Requires on `state`: `T_grid` and `gstar_func`. The rate grids
    `gamma_grid_chi` and `gamma_grid_A` read as zeros when unassigned, so an
    unpopulated grid makes this operator a no-op for that species rather than
    an error.

    Returns
    -------
    df/dt for the whole state vector. Both chi and A are affected, though
    `gamma_grid_A` is left at its zero default -- A f -> A f is
    suppressed by eps^4 and is neglected in `kernels.elastic_sm`.
    """
    T   = float(T)
    y   = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    p_phys    = state.grid.p_phys(T, gstars_func=state.gstar_func)
    gamma_chi = gamma_from_grid(T, state.T_grid, state.gamma_grid_chi)
    gamma_A   = gamma_from_grid(T, state.T_grid, state.gamma_grid_A)

    for i, name in enumerate(names):
        sl = slice(i * Np, (i + 1) * Np)
        f  = y[sl]

        if name == "chi":
            out[sl] = fokker_planck_dfdt(
                p_phys, f, T, state.species["chi"].mass_GeV, gamma_chi)
        elif name == "A":
            out[sl] = fokker_planck_dfdt(
                p_phys, f, T, state.species["A"].mass_GeV, gamma_A)
        else:
            out[sl] = 0.0

    return out



def fokker_planck_dfdt(p, f, T, m, gamma_T):
    """
    Semi-relativistic Fokker-Planck elastic scattering collision term.
    Source: Binder et al. (1706.07433) Eq.(8):

    - Fixed point: f ~ exp(-E/T)  [relativistic equilibrium]
    - Valid for arbitrary f(p), no thermality assumption on chi
    - requires momentum transfer per collision << p

    Parameters
    ----------
    p       : physical momentum grid [GeV], log-spaced, shape (Np,)
    f       : distribution on grid, shape (Np,)
    T       : SM temperature [GeV]
    m       : mass of scattering particle [GeV]
    gamma_T : scalar momentum exchange rate [GeV] from gamma_from_grid

    Returns
    -------
    dfdt : shape (Np,)

    Notes
    -----
    Derivatives are taken in log p and converted back, since the grid is
    log-spaced; `np.gradient` gives second-order central differences in the
    interior and drops to first order at the two endpoints, so the operator is
    least accurate exactly at the grid edges. Keep the grid wide enough that
    f is negligible there.

    Non-finite results are zeroed rather than propagated.
    """
    if gamma_T == 0.0:
        return np.zeros_like(f)

    E     = np.sqrt(p**2 + m**2)
    dlogp = np.log(p[1] / p[0])

    dfdlogp   = np.gradient(f, dlogp)
    dfdp      = dfdlogp / np.maximum(p, 1e-30)

    d2fdlogp2 = np.gradient(dfdlogp, dlogp)
    d2fdp2    = (d2fdlogp2 - dfdlogp) / np.maximum(p**2, 1e-60)

    result = (E / 2.0) * gamma_T * (
        T * E * d2fdp2
        + (p + 2.0 * T * E / p + T * p / E) * dfdp
        + 3.0 * f
    )
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def xA_elastic_collision_rhs(T, y_flat, state, names, Np):
    """
    Elastic chi A -> chi A "dark Compton" scattering, for both species.

    Unlike `elastic_collision_rhs`, this is a full 2 -> 2 redistribution rather
    than a Fokker-Planck approximation, because both scattering partners are
    hidden-sector particles of comparable mass -- the small-momentum-transfer
    assumption that justifies Fokker-Planck does not hold.

    Number-preserving for both species: chi and A are only moved between
    momentum bins.

    Requires `state.gain_caches_xAxA`, a list of event caches per entry in
    `state.T_grid`, built by `kernels.elastic_hidden`.

    Returns
    -------
    df/dt for the whole state vector.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    out = np.zeros_like(y)

    idx = {nm: i for i, nm in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_A   = slice(idx["A"]   * Np, (idx["A"]   + 1) * Np)

    f_chi = y[sl_chi]
    f_A   = y[sl_A]

    T_grid =  state.T_grid
    i_T = _xA_kernel_index(T, T_grid)
    cache  = state.gain_caches_xAxA[i_T]

    # The cache stores a per-event delta, so one pass yields gain AND loss
    # together; there is no separate K_loss contraction here!
    df_chi_total, df_A_total = apply_xA_cache(cache, f_chi, f_A)
    out[sl_chi] = df_chi_total
    out[sl_A]   = df_A_total
    return out
