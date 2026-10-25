"""
Time integrator for the phase-space Boltzmann equation.
For each species s in {chi, Z_D} the Boltzmann equation on the comoving grid
reduces to

    df_s(ptilde_i) / dt = sum over collision operators,

with no drift term: because ptilde = a(t) p is comoving, free expansion moves
no probability between bins, so every RHS contribution is really a
collision effect.

State vector
-------------------
The solver flattens all species into one vector,

    y = [ f_chi(p_0) ... f_chi(p_{Np-1}),  f_ZD(p_0) ... f_ZD(p_{Np-1}) ]

of length n_species * Np, so that a single Newton solve couples the species.
Species order is `list(state.f.keys())`. Species are located by name; order is arbitrary.

Time stepping
-------------
The temperature grid is prescribed by the caller. Each step
T_i -> T_{i+1} is converted to a time step via `cosmology.t_of_T_RD` and then
taken as:

    f_next = f_current + dt * dfdt(T_next, f_next),

solved for f_next with `scipy.optimize.root`.

Operators fall into three types:

  linear     heavy-partner scattering (C @ f_chi) and light-partner
             scattering, a conservative Fokker-Planck matrix: exactly implicit.
  affine     Z_D <-> f fbar decay, df_ZD/dt = S - G f_ZD with (S, G) computed
             once per step: exactly implicit.
  nonlinear  annihilation, chi chibar <-> Z_D Z_D and dark Compton: implicit
             through the Newton solve, with rate caps.

Rate capping
------------
Some operators run so far above H that backward Euler struggles to
converge. Each step measures their per-particle rate at the peak of p^2 f and,
if it exceeds the ceiling, scales the operator down. The caps only engage where the
process is deep in equilibrium, so slowing it down to Gamma_H_max * H leaves it in equilibrium.
"""

import numpy as np
from scipy.optimize import root

from dphase.collisions import (
    ZDZD_chichi_collision_rhs,
    annihilation_collision_rhs_kernel,
    decay_affine_coeffs,
    kernel_from_grid_nearest,
    loss_self_annih_dfdt,
    xZD_elastic_collision_rhs,
)
from dphase.cosmology import H_RD, t_of_T_RD
from dphase.kernels import (
    apply_xZD_elastic_loss,
    gamma_chi_from_grid,
    gamma_from_grid,
    fp_matrices_from_grid_nearest,
    heavy_partner_matrix_from_grid_nearest,
)
from dphase.in_medium import make_omega_p2_interp, step_averaged_suppression

# Positive floor applied to f before forming loss rates
_F_FLOOR = 1e-100



def get_HS_loss_term(T, y_flat, state, names, Np):
    """
    Loss-only part of the chi chibar <-> Z_D Z_D operator, for both species.

    Used solely to measure how fast the hidden-sector operator is running, so
    that `solve_BE` can compare it against H and decide whether to cap it. The
    gain term is excluded: gain and loss cancel to high precision
    near equilibrium, so sum is a poor estimate of the underlying rate.

    Parameters
    ----------
    T      : SM temperature [GeV]
    y_flat : flattened state vector, see the module docstring
    state  : PhaseSpaceState, with K_grid_xxZDZD / K_grid_ZDZDxx precomputed
    names  : species names, in state-vector order
    Np     : bins per species

    Returns
    -------
    (df_chi_loss, df_ZD_loss), each shape (Np,) and non-positive.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)

    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_ZD = slice(idx["ZD"] * Np, (idx["ZD"] + 1) * Np)

    fchi = np.maximum(y[sl_chi], _F_FLOOR)
    fZD = np.maximum(y[sl_ZD], _F_FLOOR)

    p = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    # Nearest-neighbor lookup in T, not interpolation
    K_chi = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxZDZD)
    K_ZD = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_ZDZDxx)

    dfchi_loss = loss_self_annih_dfdt(p, fchi, K_chi, dlogp)
    dfZD_loss = loss_self_annih_dfdt(p, fZD, K_ZD, dlogp)
    return dfchi_loss, dfZD_loss


def get_dark_compton_loss_term(T, y_flat, state, names, Np):
    """
    Loss-only part of the chi Z_D -> chi Z_D dark Compton operator.

    Dark-Compton counterpart of `get_HS_loss_term`, and used the same way:
    to estimate the operator's rate at the peak of the distribution so it can
    be capped against Hubble.

    Returns
    -------
    (df_chi_loss, df_ZD_loss), each shape (Np,).
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_ZD = slice(idx["ZD"] * Np, (idx["ZD"] + 1) * Np)

    fchi = np.maximum(y[sl_chi], _F_FLOOR)
    fZD = np.maximum(y[sl_ZD], _F_FLOOR)

    p = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    T_grid = state.T_grid
    i_T = int(np.argmin(np.abs(np.asarray(T_grid) - T)))
    cache = state.gain_caches_xZDxZD[i_T]

    df_chi, df_ZD = apply_xZD_elastic_loss(p, dlogp, cache["K_loss"], fchi, fZD)
    return (state.species["ZD"].dof * df_chi,
            state.species["chi"].dof * df_ZD)


def _peak_rate_over_H(loss_chi, loss_ZD, f_chi, f_ZD, p, H):
    """
    Per-particle loss rate |df/dt| / f at the peak of p^2 f, in units of H,
    taking whichever species is running faster. 0 where f has underflowed.

    The peak of p^2 f is the most occupied bin,where the operator's stiffness matters most.
    """
    peak_chi = np.argmax(np.abs(f_chi) * p**2)
    peak_ZD = np.argmax(np.abs(f_ZD) * p**2)
    f_peak_chi = np.abs(f_chi[peak_chi])
    f_peak_ZD = np.abs(f_ZD[peak_ZD])

    g_chi = (np.abs(loss_chi[peak_chi]) / f_peak_chi
             if f_peak_chi > _F_FLOOR else 0.0)
    g_ZD = (np.abs(loss_ZD[peak_ZD]) / f_peak_ZD
           if f_peak_ZD > _F_FLOOR else 0.0)
    if max(f_peak_chi, f_peak_ZD) <= _F_FLOOR:
        return 0.0
    return max(g_chi, g_ZD) / H


def solve_BE(state, T_grid,
             Gamma_H_max=200.0,
             include_scatter=True,
             include_dark_compton=True,
             include_heavy_scatter=True,
             probe_f_clip=1e120,
             decay_fermion_dist="MB",
             ann_fermion_dist="MB",
             in_medium_eps_eff=False,
             ):
    """
    Integrate the phase-space Boltzmann equation on the given temperature grid.

    Parameters
    ----------
    state : PhaseSpaceState
        Must be fully precomputed. Initial conditions are read from `state.f`.
    T_grid : array of temperatures [GeV], descending
        The integration grid, prescribed and not adapted. The returned
        snapshots are on this grid.
    Gamma_H_max : float
        Rate ceiling for chi chibar <-> Z_D Z_D, in units of H. Set to `np.inf`
        to disable capping.
    include_scatter : bool
        Enable chi B -> chi B scattering off light SM partners
    include_dark_compton : bool
        Enable the chi Z_D -> chi Z_D operator. Requires `state.gain_caches_xZDxZD`.
    include_heavy_scatter : bool
        Enable chi B -> chi B off heavy SM partners with the full operator,
        used only when `state.C_grid_heavy` is set.
    probe_f_clip : float
        Upper clip applied to f before measuring the hidden-sector loss rate.
        Guards the rate probe against an overflow.
    decay_fermion_dist, ann_fermion_dist : {"MB", "FD"}
        Statistics used for the SM fermions in the decay and annihilation
        operators. "MB" (Maxwell-Boltzmann) drops Pauli blocking; "FD" includes them.
    in_medium_eps_eff : bool
        Apply thermal effects on Z_D <-> f fbar: multiply the decay term
        (both directions, so detailed balance is preserved) by the
        step-averaged F = eps_eff^2 / eps^2 from
        `in_medium.step_averaged_suppression`. Per-step factors are stored in
        `state.in_medium_F_history`. Assumes a photon-like Z_D (m_ZD << m_Z).

    Returns
    -------
    T_grid : the input grid, returned unchanged for convenience
    snapshots : list of dicts {species_name: f}, one per grid point, on the
        comoving momentum grid. Length len(T_grid); the initial condition is element 0.

    """
    Np = state.grid._Np
    names = list(state.f.keys())
    sl_chi = slice(names.index("chi") * Np, (names.index("chi") + 1) * Np)
    sl_ZD = slice(names.index("ZD") * Np, (names.index("ZD") + 1) * Np)

    scatter_enabled = bool(include_scatter)
    dc_enabled = bool(include_dark_compton)

    Gamma_H_max_dc = Gamma_H_max

    # ---- in-medium effects ----
    if in_medium_eps_eff:
        wp2_func = make_omega_p2_interp(min(T_grid), max(T_grid))
        state.in_medium_F_history = []

    # ---- heavy-partner full operator: setup and double-counting check ----
    heavy_active = include_heavy_scatter and bool(state.C_grid_heavy)
    if heavy_active:
        heavy_names = set(state.C_grid_heavy.keys())
        # W is only ever in the heavy-partner operator, so only fermions can
        # be counted twice
        double = (heavy_names - {"W"}) - set(state.gamma_fp_excluded)
        if scatter_enabled and double:
            raise ValueError(
                f"{sorted(double)} in state.C_grid_heavy but still in the FP "
                f"gamma grid: rebuild it with "
                f"build_xfxf_kernels(T_grid, state, exclude={tuple(sorted(double))}).")

    # ---- light-partner Fokker-Planck matrices ----
    if scatter_enabled and not state.FP_matrix_grid:
        raise ValueError(
            "include_scatter needs state.FP_matrix_grid: set "
            "state.FP_matrix_grid = build_fp_matrices(state.T_grid, state) "
            "(after the gamma grids), or pass include_scatter=False.")

    # Flatten the per-species initial conditions into the solve vector, and
    # seed the snapshot list with it so snapshots[i] lines up with T_grid[i].
    f_current = np.concatenate([state.f[n] for n in names])
    snapshots = [{name: state.f[name].copy() for name in names}]

    for i in range(len(T_grid) - 1):
        T_now = T_grid[i]
        T_next = T_grid[i + 1]

        # Convert the temp. step into a time step
        g_now = state.gstar_func(T_now)
        g_next = state.gstar_func(T_next)
        dt = t_of_T_RD(T_next, g_next) - t_of_T_RD(T_now, g_now)

        H = H_RD(T_next, g_next)
        p = state.grid.p_phys(T_next, gstars_func=state.gstar_func)

        # ---- per-step linear / affine pieces, all evaluated at T_next ----
        # heavy-partner matrix and light-partner FP matrices
        C_hv = (heavy_partner_matrix_from_grid_nearest(T_next, state)
                if heavy_active else None)
        M_fp = (fp_matrices_from_grid_nearest(T_next, state)
                if scatter_enabled else None)

        # decay: affine in f_ZD, so (S, G) are computed once per step
        S_dec, G_dec = decay_affine_coeffs(T_next, state,
                                           fermion_dist=decay_fermion_dist)
        if in_medium_eps_eff:
            F_med = step_averaged_suppression(T_now, T_next, state,
                                              G_dec, wp2_func)
            S_dec = S_dec * F_med
            G_dec = G_dec * F_med
            state.in_medium_F_history.append((T_next, F_med))

        # ------------------------------------------------------------------
        # Closures over this step's T_next, dt and H.
        # ------------------------------------------------------------------

        def compute_dfdt(f_eval, hs_scale=1.0, dc_scale=1.0):
            """
            All active collision operators at T_next, returned separately
            rather than summed.
            The `*_scale` arguments are the stiffness caps. Disabled operators
            return exact zeros.

            Returns
            -------
            (dfdt_ann, dfdt_dec, dfdt_hs, dfdt_dc, dfdt_lin)
            """
            # chi B -> chi B: heavy partners (full operator) and light partners (FP)
            dfdt_lin = np.zeros_like(f_eval)
            if C_hv is not None:
                dfdt_lin[sl_chi] = C_hv @ f_eval[sl_chi]
            if M_fp is not None:
                dfdt_lin[sl_chi] += M_fp["chi"] @ f_eval[sl_chi]
                dfdt_lin[sl_ZD] += M_fp["ZD"] @ f_eval[sl_ZD]

            # chi chibar -> f fbar, W+ W-: annihilation into the SM bath.
            dfdt_ann = annihilation_collision_rhs_kernel(
                T_next, f_eval, state, names, Np, fermion_dist=ann_fermion_dist)

            # chi chibar <-> Z_D Z_D, within the dark sector. Capped by hs_scale.
            dfdt_hs = hs_scale * ZDZD_chichi_collision_rhs(
                T_next, f_eval, state, names, Np)

            # Z_D -> f fbar decay and inverse decay, S - G f_ZD.
            dfdt_dec = np.zeros_like(f_eval)
            dfdt_dec[sl_ZD] = S_dec - G_dec * f_eval[sl_ZD]

            # chi Z_D -> chi Z_D dark Compton scattering.
            if dc_enabled:
                dfdt_dc = dc_scale * xZD_elastic_collision_rhs(
                    T_next, f_eval, state, names, Np)
            else:
                dfdt_dc = np.zeros_like(f_eval)

            return dfdt_ann, dfdt_dec, dfdt_hs, dfdt_dc, dfdt_lin

        def compute_dc_scale(f_eval):
            """
            Cap factor for the dark-Compton operator, from its rate at the peak
            of the distribution.

            Returns
            -------
            scale : multiplier in (0, 1] for the operator
            ratio : the measured Gamma/H, for the progress print
            """
            if not dc_enabled:
                return 1.0, 0.0
            loss_chi_dc, loss_ZD_dc = get_dark_compton_loss_term(
                T_next, f_eval, state, names, Np)
            ratio_dc = _peak_rate_over_H(loss_chi_dc, loss_ZD_dc,
                                         f_eval[sl_chi], f_eval[sl_ZD], p, H)
            scale = (Gamma_H_max_dc / ratio_dc
                     if ratio_dc > Gamma_H_max_dc else 1.0)
            return scale, ratio_dc

        # ---- measure the hidden-sector rate, to set hs_scale ----
        f_probe_hs = np.clip(np.asarray(f_current, dtype=float),
                             0.0, probe_f_clip)
        loss_chi, loss_ZD = get_HS_loss_term(T_next, f_probe_hs, state,
                                            names, Np)
        eff_ratio = _peak_rate_over_H(loss_chi, loss_ZD, f_probe_hs[sl_chi],
                                      f_probe_hs[sl_ZD], p, H)
        hs_scale = (Gamma_H_max / eff_ratio
                    if eff_ratio > Gamma_H_max else 1.0)

        dc_scale, dc_ratio = compute_dc_scale(f_current)

        def all_terms(f_eval):
            return compute_dfdt(f_eval, hs_scale=hs_scale, dc_scale=dc_scale)

        # ---- residual normalization ----
        # f spans hundreds of decades across the grid, so an unscaled residual
        # would be dominated by the largest bin. Estimate dfdt at the current f,
        # to scale the residual per bin by the expected change.
        dfdt_total_est = sum(all_terms(f_current))
        res_scale = np.maximum(np.abs(f_current),
                               np.abs(dt * dfdt_total_est))
        res_scale = np.maximum(res_scale, 1e-50)

        def residual(f_next):
            """Backward-Euler residual, normalized per bin by res_scale."""
            return (f_next - f_current - dt * sum(all_terms(f_next))) / res_scale

        # ---- Newton initial guess: step the stiff linear/affine pieces exactly ----
        # Decay is affine and diagonal in p: backward Euler is closed-form.
        f_guess = f_current.copy()
        f_guess[sl_ZD] = (f_current[sl_ZD] + dt * S_dec) / (1.0 + dt * G_dec)

        # Heavy- and light-partner scattering terms are linear:
        L_chi = None
        if C_hv is not None:
            L_chi = C_hv.copy()
        if M_fp is not None:
            L_chi = M_fp["chi"] if L_chi is None else L_chi + M_fp["chi"]
        if L_chi is not None:
            A_chi = np.eye(Np) - dt * L_chi
            f_guess[sl_chi] = np.linalg.solve(A_chi, f_current[sl_chi])
        A_ZD = None
        if M_fp is not None and np.any(M_fp["ZD"]):
            A_ZD = np.eye(Np) - dt * M_fp["ZD"]
            f_guess[sl_ZD] = np.linalg.solve(A_ZD, f_guess[sl_ZD])

        # 'hybr' is a Powell hybrid method with a numerically estimated Jacobian.
        # Find f_next such that residual(f_next) = 0, i.e. the backward Euler
        # equation f_next = f_current + dt * dfdt(f_next) is satisfied.
        result = root(residual, f_guess, method='hybr',
                      tol=1e-10, options={'maxfev': 5000})

        if not result.success:
            # Fall back so the run continues: explicit for the nonlinear
            # terms, backward Euler for decay and the linear terms.
            print(f"  Warning: Newton failed at T={T_next:.3f} GeV, "
                  f"falling back to explicit (decay, linear terms implicit)")
            terms = all_terms(f_current)
            f_next = f_current + dt * (sum(terms) - terms[1] - terms[4])
            f_next[sl_ZD] = (f_next[sl_ZD] + dt * S_dec) / (1.0 + dt * G_dec)
            if L_chi is not None:
                f_next[sl_chi] = np.linalg.solve(A_chi, f_next[sl_chi])
            if A_ZD is not None:
                f_next[sl_ZD] = np.linalg.solve(A_ZD, f_next[sl_ZD])
        else:
            f_next = result.x

        # Enforce f >= 0
        f_next = np.maximum(f_next, 0.0)

        if i % 10 == 0:
            # heavy-partner scattering-out rate at the peak of p^2 f_chi
            hv_ratio = 0.0
            if C_hv is not None:
                k_pk = np.argmax(np.abs(f_next[sl_chi]) * p**2)
                hv_ratio = abs(C_hv[k_pk, k_pk]) / H
            # *_scale columns: any value well below 1 means that operator is being capped.
            print(f"Step {i}/{len(T_grid)-1}: T={T_next:.3f} GeV "
                  f"Γ_HS/H={eff_ratio:.1e} hs_scale={hs_scale:.2e} "
                  f"Γ_dc/H={dc_ratio:.1e} dc_scale={dc_scale:.2e} "
                  f"Γ_hv/H={hv_ratio:.1e} "
                  f"|f|={np.max(f_next):.2e}")

        # unflatten solution back into per-species arrays.
        snapshots.append({name: f_next[j * Np:(j + 1) * Np].copy()
                          for j, name in enumerate(names)})
        f_current = f_next

    return T_grid, snapshots
