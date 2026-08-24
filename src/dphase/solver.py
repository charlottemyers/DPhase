"""
Time integrator for the phase-space Boltzmann equation.

For each species s in {chi, A} the Boltzmann equation on the comoving grid
reduces to

    df_s(ptilde_i) / dt = sum over collision operators,

with no drift term: because ptilde = a(t) p is comoving, free expansion moves
no probability between bins, so every right-hand side here is a genuine
collision effect. See `dphase.grid` for why that coordinate was chosen.

State vector layout
-------------------
The solver flattens all species into one vector,

    y = [ f_chi(p_0) ... f_chi(p_{Np-1}),  f_A(p_0) ... f_A(p_{Np-1}) ]

of length n_species * Np, so that a single Newton solve couples the species.
Species order is `list(state.f.keys())`, i.e. the order they were passed to
`PhaseSpaceState`.

    ORDERING ASSUMPTION: parts of this module index the vector positionally --
    `f_eval[:Np]` for chi and `f_eval[Np:2*Np]` for A.

Time stepping
-------------
The temperature grid is prescribed by the caller. Each step
T_i -> T_{i+1} is converted to a time step via `cosmology.t_of_T_RD` and then
taken as **backward Euler**,

    f_next = f_current + dt * dfdt(T_next, f_next),

solved for f_next with `scipy.optimize.root`. Backward Euler is L-stable, which is what matters here: the
collision terms are very stiff near freeze-out, where rates exceed H by many orders of magnitude. An explicit
step of the same size would be unusable.

Two deviations from a pure implicit step:

  * The elastic operator can be lagged -- evaluated at f_current and held
    fixed during the Newton iteration (`lag_scatter_in_newton`). That makes
    the scheme IMEX rather than fully implicit.
  * If `root` fails to converge, the step falls back to explicit Euler and
    prints a warning. That keeps a run alive, but the resulting step is
    unconditionally unstable, so warnings here should not be ignored.

"""

import numpy as np
from scipy.optimize import root

from dphase.collisions import (
    AA_chichi_collision_rhs,
    annihilation_collision_rhs_kernel,
    decay_collision_rhs_direct_allf,
    elastic_collision_rhs,
    kernel_from_grid_nearest,
    loss_self_annih_dfdt,
    xA_elastic_collision_rhs,
)
from dphase.cosmology import H_RD, t_of_T_RD
from dphase.kernels import apply_xA_elastic_loss, gamma_from_grid

# Positive floor applied to f before forming loss rates
_F_FLOOR = 1e-100

# Index of the elastic (chi f -> chi f) entry in the 5-tuple returned by `compute_dfdt`.
_EL_TERM = 3


def get_HS_loss_term(T, y_flat, state, names, Np):
    """
    Loss-only part of the chi chibar <-> A A operator, for both species.

    Used solely to measure how fast the hidden-sector operator is running, so
    that `solve_BE` can compare it against H and decide whether to cap it. The
    gain term is deliberately excluded: gain and loss cancel to high precision
    near equilibrium, so their sum is a poor estimate of the underlying rate.

    Parameters
    ----------
    T      : SM temperature [GeV]
    y_flat : flattened state vector, see the module docstring
    state  : PhaseSpaceState, with K_grid_xxAA / K_grid_AAxx precomputed
    names  : species names, in state-vector order
    Np     : bins per species

    Returns
    -------
    (df_chi_loss, df_A_loss), each shape (Np,) and non-positive.
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)

    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_A = slice(idx["A"] * Np, (idx["A"] + 1) * Np)

    fchi = np.maximum(y[sl_chi], _F_FLOOR)
    fA = np.maximum(y[sl_A], _F_FLOOR)

    p = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    # Nearest-neighbour lookup in T, not interpolation
    K_chi = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_xxAA)
    K_A = kernel_from_grid_nearest(T, state.T_grid, state.K_grid_AAxx)

    dfchi_loss = loss_self_annih_dfdt(p, fchi, K_chi, dlogp)
    dfA_loss = loss_self_annih_dfdt(p, fA, K_A, dlogp)
    return dfchi_loss, dfA_loss


def get_dark_compton_loss_term(T, y_flat, state, names, Np):
    """
    Loss-only part of the chi A -> chi A dark Compton operator.

    The dark-Compton counterpart of `get_HS_loss_term`, and used the same way:
    to estimate the operator's rate at the peak of the distribution so it can
    be capped against Hubble. Reads the `K_loss` matrix stored alongside the
    event cache by `kernels.elastic_hidden.build_xA_elastic_gain_cache`.
    Returns
    -------
    (df_chi_loss, df_A_loss), each shape (Np,).
    """
    T = float(T)
    y = np.asarray(y_flat, dtype=float)
    idx = {n: i for i, n in enumerate(names)}
    sl_chi = slice(idx["chi"] * Np, (idx["chi"] + 1) * Np)
    sl_A = slice(idx["A"] * Np, (idx["A"] + 1) * Np)

    fchi = np.maximum(y[sl_chi], _F_FLOOR)
    fA = np.maximum(y[sl_A], _F_FLOOR)

    p = state.grid.p_phys(T, gstars_func=state.gstar_func)
    dlogp = state.grid.dlogp

    T_grid = state.T_grid
    i_T = int(np.argmin(np.abs(np.asarray(T_grid) - T)))
    cache = state.gain_caches_xAxA[i_T]

    return apply_xA_elastic_loss(p, dlogp, cache["K_loss"], fchi, fA)


def solve_BE(state, T_grid,
             Gamma_H_max=200.0,
             explicit_threshold=0.0,
             include_scatter=True,
             include_dark_compton=True,
             lag_scatter_in_newton=True,
             scatter_step_cap=0.2,
             probe_f_clip=1e120,
             decay_fermion_dist="MB",
             ann_fermion_dist="MB",
             ):
    """
    Integrate the phase-space Boltzmann equation down the given temperature grid.

    Backward Euler with per-step Newton solves, plus the rate capping described
    in the module docstring.

    Parameters
    ----------
    state : PhaseSpaceState
        Must be fully precomputed. Initial conditions are read from `state.f`.
    T_grid : array of temperatures [GeV], descending
        The integration grid, prescribed and *not* adapted. Step size is
        therefore entirely the caller's responsibility: too coarse near
        freeze-out and the Newton solve will start failing. The returned
        snapshots are on this grid.
    Gamma_H_max : float
        Rate ceiling, in units of H. Any capped operator running faster than
        this is scaled down. Set to `np.inf` to disable capping entirely.
    explicit_threshold : float
        Take an explicit rather than implicit step while
        max|f| < explicit_threshold, as a cheap fast path in the free-streaming
        tail where nothing is stiff.
        NOTE: with the default of 0.0 this branch is unreachable, since f >= 0
        always. Explicit path is opt-in only.
    include_scatter : bool
        Enable the chi f -> chi f Fokker-Planck operator. Requires `state.gamma_grid_*`.
    include_dark_compton : bool
        Enable the chi A -> chi A operator. Requires `state.gain_caches_xAxA`.
    lag_scatter_in_newton : bool
        Hold the elastic operator fixed at f_current during the Newton
        iteration (IMEX) instead of solving it implicitly. Usually needed for
        convergence.
    scatter_step_cap : float or None
        Maximum fractional change in any bin that the elastic operator alone
        may produce in one step: eta_sc = max |dt * dfdt_el| / max(|f|, floor).
        If exceeded, the operator is scaled by scatter_step_cap / eta_sc. This
        is a *numerical* limiter distinct from the Gamma/H cap, and it applies
        even when the physical rate is modest, because the Fokker-Planck second
        derivative can be large where f varies steeply across neighbouring bins.
        None or <= 0 disables it.
    probe_f_clip : float
        Upper clip applied to f before measuring the hidden-sector loss rate.
        Guards the rate probe against an overflowing intermediate iterate; does
        not affect the solution itself.
    decay_fermion_dist, ann_fermion_dist : {"MB", "FD"}
        Statistics used for the SM fermions in the decay and annihilation
        operators. "MB" (Maxwell-Boltzmann) drops Pauli blocking, which is a
        good approximation for the non-relativistic bath and avoids the
        blocking factors entirely; "FD" includes them.

    Returns
    -------
    T_grid : the input grid, returned unchanged for convenience
    snapshots : list of dicts {species_name: f}, one per grid point, on the
        comoving momentum grid. Length len(T_grid) -- the initial condition is
        element 0.

    """
    Np = state.grid._Np
    names = list(state.f.keys())

    scatter_enabled = bool(include_scatter)

    # Flatten the per-species initial conditions into the solve vector, and
    # seed the snapshot list with it so snapshots[i] lines up with T_grid[i].
    f_current = np.concatenate([state.f[n] for n in names])
    snapshots = [{name: state.f[name].copy() for name in names}]

    for i in range(len(T_grid) - 1):
        T_now = T_grid[i]
        T_next = T_grid[i + 1]

        # Convert the temperature step into a time step. Both endpoints use
        # their own g_*, so the step accounts for entropy release between them.
        g_now = state.gstar_func(T_now)
        g_next = state.gstar_func(T_next)
        dt = t_of_T_RD(T_next, g_next) - t_of_T_RD(T_now, g_now)

        H = H_RD(T_next, g_next)

        f_max = np.max(np.abs(f_current))

        # ------------------------------------------------------------------
        # Closures over this step's T_next, dt and H.
        # ------------------------------------------------------------------

        def compute_dfdt(T_eval, f_eval, hs_scale=1.0, dc_scale=1.0,
                         sc_scale=1.0):
            """
            All active collision operators, returned separately rather than
            summed so the caller can override one of them (see `_EL_TERM`).

            The `*_scale` arguments are the stiffness caps. Disabled operators
            return exact zeros, so the sum is always well defined.

            Returns
            -------
            (dfdt_ann, dfdt_dec, dfdt_hs, dfdt_el, dfdt_dc)
            """
            # chi chibar -> f fbar: annihilation into the SM bath.
            dfdt_ann = annihilation_collision_rhs_kernel(
                T_eval, f_eval, state, names, Np, fermion_dist=ann_fermion_dist)

            # A -> f fbar decay and inverse decay.
            dfdt_dec = decay_collision_rhs_direct_allf(
                T_eval, f_eval, state, names, Np,
                fermion_dist=decay_fermion_dist)

            # chi chibar <-> A A, within the dark sector. Capped by hs_scale.
            dfdt_hs = hs_scale * AA_chichi_collision_rhs(
                T_eval, f_eval, state, names, Np)

            # chi f -> chi f elastic scattering off the bath (Fokker-Planck).
            if scatter_enabled:
                dfdt_el = sc_scale * elastic_collision_rhs(
                    T_eval, f_eval, state, names, Np)
            else:
                dfdt_el = np.zeros_like(f_eval)

            # chi A -> chi A dark Compton scattering.
            if include_dark_compton:
                dfdt_dc = dc_scale * xA_elastic_collision_rhs(
                    T_eval, f_eval, state, names, Np)
            else:
                dfdt_dc = np.zeros_like(f_eval)

            return dfdt_ann, dfdt_dec, dfdt_hs, dfdt_el, dfdt_dc

        def compute_dc_scale(T_eval, f_eval):
            """
            Cap factor for the dark-Compton operator, from its rate at the peak
            of the distribution.

            The rate is measured where p^2 f is largest -- the peak of the
            *number* density per log momentum, not of f itself, which for a
            thermal shape peaks at p = 0 and carries no particles.
            Returns
            -------
            scale : multiplier in (0, 1] for the operator
            ratio : the measured Gamma/H, for the progress print
            """
            if not include_dark_compton:
                return 1.0, 0.0

            loss_chi_dc, loss_A_dc = get_dark_compton_loss_term(
                T_eval, f_eval, state, names, Np)
            f_chi_v = f_eval[:Np]
            f_A_v = f_eval[Np:2 * Np]
            p_v = state.grid.p_phys(T_eval, gstars_func=state.gstar_func)

            peak_chi_v = np.argmax(np.abs(f_chi_v) * p_v**2)
            peak_A_v = np.argmax(np.abs(f_A_v) * p_v**2)

            f_peak_chi_v = np.abs(f_chi_v[peak_chi_v])
            f_peak_A_v = np.abs(f_A_v[peak_A_v])

            # Per-particle rate = |df/dt| / f at the peak. Skipped where f has
            # underflowed, since the ratio there is noise.
            g_chi_v = (np.abs(loss_chi_dc[peak_chi_v]) / f_peak_chi_v
                       if f_peak_chi_v > _F_FLOOR else 0.0)
            g_A_v = (np.abs(loss_A_dc[peak_A_v]) / f_peak_A_v
                     if f_peak_A_v > _F_FLOOR else 0.0)

            # Cap on whichever species is running faster.
            gamma_dc = max(g_chi_v, g_A_v)
            ratio_dc = (gamma_dc / H if max(f_peak_chi_v, f_peak_A_v) > _F_FLOOR
                        else 0.0)
            scale = (Gamma_H_max / ratio_dc if ratio_dc > Gamma_H_max else 1.0)
            return scale, ratio_dc

        def compute_sc_scale(T_eval, f_eval):
            """
            Cap factor for the elastic (chi f -> chi f) operator.

            Two independent limiters, of which the stricter wins:

            1. Physical, Gamma/H: uses the precomputed gamma(T) directly, since
               for Fokker-Planck the momentum exchange rate *is* the operator's
               coefficient.
            2. Numerical, step impact: bounds the fractional change this
               operator alone may make to any bin over dt. The Fokker-Planck
               operator differentiates f twice in p, so a steep feature between
               adjacent bins can produce a huge df/dt even at a modest physical
               rate.

            Returns
            -------
            sc_scale : multiplier in (0, 1] for the operator
            ratio_sc : measured Gamma/H (0 if the gamma grids are absent)
            eta_sc   : measured step impact, for the progress print
            """
            if not scatter_enabled:
                return 1.0, 0.0, 0.0

            # --- limiter 1: physical rate against Hubble ---
            ratio_sc = 0.0
            scale_phys = 1.0
            has_gamma_grid = all(
                getattr(state, attr, None) is not None
                for attr in ("gamma_grid_chi", "gamma_grid_A"))
            if has_gamma_grid:
                gamma_sc_chi = gamma_from_grid(T_eval, state.T_grid,
                                               state.gamma_grid_chi)
                gamma_sc_A = gamma_from_grid(T_eval, state.T_grid,
                                             state.gamma_grid_A)
                gamma_sc = max(gamma_sc_chi, gamma_sc_A)
                ratio_sc = gamma_sc / H
                scale_phys = (Gamma_H_max / ratio_sc
                              if ratio_sc > Gamma_H_max else 1.0)

            # --- limiter 2: fractional step impact ---
            dfdt_el = elastic_collision_rhs(T_eval, f_eval, state, names, Np)
            dfdt_el = np.where(np.isfinite(dfdt_el), dfdt_el, 0.0)
            denom = np.maximum(np.abs(f_eval), 1e-40)
            # A bin with f at the floor and a large dfdt gives a huge ratio;
            # that is the intended signal, so let it overflow to inf quietly
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                step_vec = (np.abs(dt) * np.abs(dfdt_el)) / denom
            step_vec = np.where(np.isfinite(step_vec), step_vec, np.inf)
            eta_sc = float(np.max(step_vec)) if step_vec.size else 0.0

            scale_num = 1.0
            if scatter_step_cap is not None and scatter_step_cap > 0.0:
                if np.isfinite(eta_sc) and eta_sc > scatter_step_cap:
                    scale_num = scatter_step_cap / eta_sc

            return min(scale_phys, scale_num), ratio_sc, eta_sc

        # ==================================================================
        # EXPLICIT branch -- opt-in fast path, see `explicit_threshold`
        # ==================================================================
        if f_max < explicit_threshold:
            # Midpoint in log T, consistent with the geometric spacing the
            # temperature grid is normally built with.
            T_mid = np.sqrt(T_now * T_next)
            dc_scale, _ = compute_dc_scale(T_mid, f_current)
            sc_scale, _, _ = compute_sc_scale(T_mid, f_current)
            terms = compute_dfdt(T_mid, f_current, hs_scale=1.0,
                                 dc_scale=dc_scale, sc_scale=sc_scale)
            # Clamp at 0: f is a phase-space density and an explicit step can
            # overshoot a fast loss term into negative values.
            f_next = np.maximum(f_current + dt * sum(terms), 0.0)

            if i % 20 == 0:
                print(f"Step {i}/{len(T_grid)-1}: T={T_next:.3f} GeV "
                      f"[EXPLICIT], |f|={np.max(f_next):.2e}")

        # ==================================================================
        # IMPLICIT branch -- backward Euler, solved by Newton
        # ==================================================================
        else:
            # ---- measure the hidden-sector rate, to set hs_scale ----
            f_probe_hs = np.clip(np.asarray(f_current, dtype=float),
                                 0.0, probe_f_clip)
            loss_chi, loss_A = get_HS_loss_term(T_next, f_probe_hs, state,
                                                names, Np)
            f_chi = f_probe_hs[:Np]
            f_A = f_probe_hs[Np:2 * Np]
            p = state.grid.p_phys(T_next, gstars_func=state.gstar_func)

            # Same peak-of-(p^2 f) logic as compute_dc_scale; see there.
            peak_chi = np.argmax(np.abs(f_chi) * p**2)
            peak_A = np.argmax(np.abs(f_A) * p**2)
            f_peak_chi = np.abs(f_chi[peak_chi])
            f_peak_A = np.abs(f_A[peak_A])

            gamma_loss_chi = (np.abs(loss_chi[peak_chi]) / f_peak_chi
                              if f_peak_chi > _F_FLOOR else 0.0)
            gamma_loss_A = (np.abs(loss_A[peak_A]) / f_peak_A
                            if f_peak_A > _F_FLOOR else 0.0)
            gamma_loss = max(gamma_loss_chi, gamma_loss_A)
            eff_ratio = (gamma_loss / H
                         if max(f_peak_chi, f_peak_A) > _F_FLOOR else 0.0)

            hs_scale = (Gamma_H_max / eff_ratio
                        if eff_ratio > Gamma_H_max else 1.0)

            dc_scale, dc_ratio = compute_dc_scale(T_next, f_current)
            sc_scale, sc_ratio, sc_step_ratio = compute_sc_scale(T_next,
                                                                 f_current)

            # ---- optionally freeze elastic term (IMEX) ----
            if lag_scatter_in_newton and scatter_enabled:
                dfdt_el_lag = sc_scale * elastic_collision_rhs(
                    T_next, f_current, state, names, Np)
            else:
                dfdt_el_lag = None

            # ---- residual normalisation ----
            # `root`'s tolerance is absolute, but f spans hundreds of decades
            # across the grid, so an unscaled residual would be dominated
            # entirely by the largest bin and the tail would go unconverged.
            terms_est = compute_dfdt(T_next, f_current, hs_scale=hs_scale,
                                     dc_scale=dc_scale, sc_scale=sc_scale)
            if dfdt_el_lag is not None:
                terms_est = list(terms_est)
                terms_est[_EL_TERM] = dfdt_el_lag
            dfdt_total_est = sum(terms_est)
            res_scale = np.maximum(np.abs(f_current),
                                   np.abs(dt * dfdt_total_est))
            res_scale = np.maximum(res_scale, 1e-50)   # never divide by zero

            def residual(f_next):
                """Backward-Euler residual, normalised per bin by res_scale."""
                terms = compute_dfdt(T_next, f_next, hs_scale=hs_scale,
                                     dc_scale=dc_scale, sc_scale=sc_scale)
                if dfdt_el_lag is not None:
                    terms = list(terms)
                    terms[_EL_TERM] = dfdt_el_lag
                dfdt_total = sum(terms)
                return (f_next - f_current - dt * dfdt_total) / res_scale

            # 'hybr' is a Powell hybrid method with a numerically estimated
            # Jacobian -- the collision operators have no analytic Jacobian
            # available, so each iteration costs ~n_species*Np extra operator evaluations
            result = root(residual, f_current, method='hybr',
                          tol=1e-10, options={'maxfev': 5000})

            if not result.success:
                # Fall back to an explicit step so the run continues
                print(f"  Warning: Newton failed at T={T_next:.3f} GeV, "
                      f"falling back to explicit")
                terms = compute_dfdt(T_next, f_current, hs_scale=hs_scale,
                                     dc_scale=dc_scale, sc_scale=sc_scale)
                if dfdt_el_lag is not None:
                    terms = list(terms)
                    terms[_EL_TERM] = dfdt_el_lag
                f_next = f_current + dt * sum(terms)
            else:
                f_next = result.x

            # Enforce f >= 0. Newton has no such constraint built in.
            f_next = np.maximum(f_next, 0.0)

            if i % 10 == 0:
                # Watch the *_scale columns: any value well below 1 means that
                # operator is being capped.
                print(f"Step {i}/{len(T_grid)-1}: T={T_next:.3f} GeV "
                      f"Γ_HS/H={eff_ratio:.1e} hs_scale={hs_scale:.2e} "
                      f"Γ_sc/H={sc_ratio:.1e} η_sc={sc_step_ratio:.1e} "
                      f"sc_scale={sc_scale:.2e} "
                      f"Γ_dc/H={dc_ratio:.1e} dc_scale={dc_scale:.2e} "
                      f"|f|={np.max(f_next):.2e}")

        # Unflatten this step's solution back into per-species arrays.
        snapshots.append({name: f_next[j * Np:(j + 1) * Np].copy()
                          for j, name in enumerate(names)})
        f_current = f_next

    return T_grid, snapshots
