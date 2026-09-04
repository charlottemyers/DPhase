"""
Coupled Boltzmann equation (CBE) solver.

Where `dphase.solver` evolves the full f(p, T), this module assumes both dark
species keep a Maxwell-Boltzmann shape and tracks only three numbers:

    u = [ln n_chi, ln n_A, ln T_hidden]

as functions of x = m_chi / T. That is the standard treatment, and it is
exactly what the phase-space solver must reproduce in the regime where kinetic
equilibrium actually holds.

The hidden-sector temperature T_hidden is a genuine dynamical variable here,
not assumed equal to the SM temperature T: `dTh_dt_rel` evolves it from the
energy balance of annihilation, decay and elastic transfer. Set
`params["no_Th_evolution"] = True` to pin T_hidden = T instead.


Params dictionary
-----------------
Required : mchi, mA, gchi, gA, sv_xxAA, sv_xxee, gamma_Aee, alphaD, epsilon
           (sv_* may be a scalar or a callable of temperature)
Optional : t_dep, include_hs_in_H, no_Th_evolution, elastic, delta_dxdt,
           max_abs_dlnydx
"""

import numpy as np
from mpmath import zeta
from scipy.integrate import solve_ivp
from scipy.special import kve

from dphase.constants import ALPHA_EM, ME, VAL_FLOOR
from dphase.cosmology import (
    H_of_T,
    dln_gstars_SM_dlnT,
    ln_neq,
    neq_stable,
    safe_exp,
)


def dx_dt(T, H, m, t_dep = True, delta = 1e-6):
    """dx/dt for x = mchi / T."""
    x = m / T
    corr = 1.0 + (1.0/3.0) * dln_gstars_SM_dlnT(T, t_dep, delta)
    return H * x / corr


def rho_i_exact(n_i, m_i, Th):
    # source: 2504.00077, eqn 2.15
    Th   = np.maximum(Th, VAL_FLOOR)        # avoid z = inf / division by zero
    z    = m_i / Th
    # Use scaled Kv: Kv(z) = e^{-z} kve(order, z);  ratio is stable as kve(1)/kve(2)
    k2   = kve(2, z)
    ratio = kve(1, z) / np.maximum(k2, VAL_FLOOR)
    return n_i * (m_i * ratio + 3.0 * Th)

def P_i_exact(n_i, Th):
    # source: 2504.00077, eqn 2.15
    # MB equation of state is ideal-gas exactly: P = n T
    return n_i * Th


def _R(z):
    """
    Bessel ratio K_1(z)/K_2(z), which is the thermally averaged <m/E> for a
    Maxwell-Boltzmann species with z = m/T. Tends to 1 in the
    non-relativistic limit and to z/2 in the relativistic one.

    Computed from the exponentially scaled kve so that the e^{-z} factors
    cancel exactly instead of underflowing at large z.
    """
    return kve(1, z) / kve(2, z)

def _dR_dz(z, rel_step=1e-6):
    """
    dR/dz by centered difference. Enters the hidden-sector heat capacity in
    `dTh_dt_rel`; done numerically because the closed form needs K_0 as well
    and buys nothing at this accuracy.
    """
    dz = np.maximum(rel_step * z, 1e-6)
    Rp = kve(1, z+dz)/kve(2, z+dz)
    Rm = kve(1, z-dz)/kve(2, z-dz)
    return (Rp - Rm) / (2.0*dz)


def dTh_dt_rel(nchi, nA, Th, mchi, mA, H, rhoh, Ph, Qh, dnchi_dt, dnA_dt):
    zc = mchi /np.maximum(Th, 1e-300)
    zA = mA   / np.maximum(Th, 1e-300)

    Rchi = _R(zc)
    RA = _R(zA)
    dRchi_dz = _dR_dz(zc)
    dRA_dz = _dR_dz(zA)

    # h_i
    h_chi = mchi * Rchi + 3.0 * Th
    h_A   = mA   * RA   + 3.0 * Th

    # C_h = sum n_i [ 3 - z_i^2 dR/dz ] HEAT CAPACITY at constant composition
    Ch = nchi * (3.0 - zc*zc * dRchi_dz) + nA * (3.0 - zA*zA * dRA_dz)
    Ch = max(Ch, 1e-300)

    numer = Qh - (h_chi * dnchi_dt + h_A * dnA_dt) - 3.0 * H * (rhoh + Ph)
    return numer / Ch


def n_rel_fermion_FD(T, g):
    # relativistic fermion number density
    return (3.0*float(zeta(3))/(4.0*np.pi**2)) * g * T**3  # g=4 for e+ and e- total

def sigma_mt_chi_e(alphaD, epsilon, mA, T):
    # momentum-transfer xsec w/ thermal screening
    q2 = 3.0*ME*T
    denom = (mA*mA + q2)**2
    return 16.0*np.pi*alphaD*ALPHA_EM*(epsilon**2)/denom

def gamma_kin(alphaD, epsilon, mA, mchi, T):
    # per-chi kinetic coupling rate
    ne = n_rel_fermion_FD(T, g=4.0) # 4 dof because I include e+ and e-
    return (2.0*ME/mchi) * ne * sigma_mt_chi_e(alphaD, epsilon, mA, T)


def _sv(value, T):
    """Cross section entries may be a constant or a callable of temperature."""
    return value(T) if callable(value) else value


def _number_collision_terms(params, nchi, nA, T, Th, svxxAA_temp="HS"):
    """
    Collision terms for the two number densities, shared by `collisions` and
    `collisions_noTh`.

        C_chi = -<sv>_xxAA (n_chi^2 - B) - <sv>_xxee (n_chi^2 - n_chi,eq^2)
        C_A   = +2 <sv>_xxAA (n_chi^2 - B)
                - Gamma_Aee (n_A <m/E>_Th - n_A,eq <m/E>_T)

    B is the detailed-balance factor (n_chi,eq^Th / n_A,eq^Th)^2 n_A^2, which
    is what makes chi chibar <-> AA relax to the *hidden-sector* equilibrium at
    T_hidden rather than to the SM one.

    The factor of 2 on the first C_A term counts the two dark photons produced
    per annihilation. <m/E> = K_1(z)/K_2(z) is the thermally averaged time
    dilation that slows the A decay.

    `svxxAA_temp` selects the temperature at which the dark-sector
    annihilation cross section is evaluated: "HS" for T_hidden (correct, since
    both incoming legs are dark) or anything else for the SM T, for comparison.

    Returns
    -------
    C_chi, C_A, nchi_eq_SM, nA_eq_SM
    """
    mchi, mA = params["mchi"], params["mA"]
    gchi, gA = params["gchi"], params["gA"]

    sv_xxAA = _sv(params["sv_xxAA"], Th if svxxAA_temp == "HS" else T)
    sv_xxee = _sv(params["sv_xxee"], T)
    gamma_Aee = params["gamma_Aee"]

    # Detailed balance in log space:
    #   ln B = 2 (ln n_chi,eq^Th - ln n_A,eq^Th) + 2 ln n_A
    ln_balance = (2.0 * (ln_neq(mchi, gchi, Th) - ln_neq(mA, gA, Th))
                  + 2.0 * np.log(max(nA, VAL_FLOOR)))
    balance = safe_exp(ln_balance)          # underflows cleanly to 0

    nchi_eq_SM = safe_exp(ln_neq(mchi, gchi, T))
    nA_eq_SM = safe_exp(ln_neq(mA, gA, T))

    # Time dilation <m/E> = K1/K2, at the hidden and the SM temperature.
    lorentz_HS = _R(mA / np.maximum(Th, VAL_FLOOR))
    lorentz_SM = _R(mA / np.maximum(T, VAL_FLOOR))

    C_chi = (-sv_xxAA * (nchi**2 - balance)
             - sv_xxee * (nchi**2 - nchi_eq_SM**2))
    C_A = (2.0 * sv_xxAA * (nchi**2 - balance)
           - gamma_Aee * (nA * lorentz_HS - nA_eq_SM * lorentz_SM))

    return C_chi, C_A, nchi_eq_SM, nA_eq_SM


def collisions(params, nchi, nA, T, Th, svxxAA_temp="HS"):
    """
    Full collision terms: number densities and the hidden-sector energy
    injection Q_h that drives dT_hidden/dt.

    The three energy channels are

        Q_ann : annihilation to SM fermions removes rest mass m_chi per event
        Q_dec : A decay removes rest mass m_A per event. No time dilation
                factor here -- this is an energy budget, not a rate.
        Q_el  : elastic chi-e scattering drives T_hidden towards T, and
                vanishes when they are equal. Gated on params["elastic"].

    Returns
    -------
    C_chi, C_A, Q_h, Q_ann, Q_dec, Q_el, Gkin
    """
    mchi, mA = params["mchi"], params["mA"]
    alphaD, epsilon = params["alphaD"], params["epsilon"]

    C_chi, C_A, nchi_eq_SM, nA_eq_SM = _number_collision_terms(
        params, nchi, nA, T, Th, svxxAA_temp)

    sv_xxee = _sv(params["sv_xxee"], T)
    Q_ann = -mchi * sv_xxee * (nchi**2 - nchi_eq_SM**2)
    Q_dec = -mA * params["gamma_Aee"] * (nA - nA_eq_SM)

    if params.get("elastic", False):
        Gkin = gamma_kin(alphaD, epsilon, mA, mchi, T)
        # Elastic transfer is proportional to the hidden-sector enthalpy and
        # to the temperature difference that drives it.
        rhoh = rho_i_exact(nchi, mchi, Th) + rho_i_exact(nA, mA, Th)
        Ph = P_i_exact(nchi, Th) + P_i_exact(nA, Th)
        Q_el = Gkin * (rhoh + Ph) * (T - Th) / max(Th, VAL_FLOOR)
    else:
        Gkin = 0.0
        Q_el = 0.0

    return C_chi, C_A, Q_ann + Q_dec + Q_el, Q_ann, Q_dec, Q_el, Gkin


def collisions_noTh(params, nchi, nA, T, Th, svxxAA_temp="HS"):
    """
    Number-density collision terms only, for runs with T_hidden pinned to T.
    Skips the energy bookkeeping, which is unused in that mode.
    """
    C_chi, C_A, _, _ = _number_collision_terms(
        params, nchi, nA, T, Th, svxxAA_temp)
    return C_chi, C_A


# ----------------------------
# x-domain RHS: d/dx = (1/(dx/dt)) d/dt
# ----------------------------
def rhs_logx(x, u, params, svxxAA_temp = "HS"):
    """
    log-space RHS:
      u = [ln n_chi, ln n_Ap, ln Th]. returns d/dx of those logs
    """

    mchi = params["mchi"];  mA = params["mA"]
    t_dep = params.get("t_dep", True)
    include_h = params.get("include_hs_in_H", True)
    T    = mchi / x  # SM temperature

    ln_nchi, ln_nA, ln_Th  = u
    nchi = safe_exp(ln_nchi) # represents nchi ONLY (not nchi + bar_nchi)
    nA   = safe_exp(ln_nA)

    # Prevent pathologies when stiff steps push logs to extreme values.
    ln_min_Th = np.log(VAL_FLOOR)
    if ln_Th < ln_min_Th:
        ln_Th = ln_min_Th

    if params.get("no_Th_evolution", False):
        Th = T
    else:
        Th = safe_exp(ln_Th)

    # HS energy density and pressure
    rhoh = rho_i_exact(nchi, mchi, Th) + rho_i_exact(nA, mA, Th)
    Ph = P_i_exact(nchi, Th) + P_i_exact(nA, Th)

    # Hubble and x-dot
    delta = params.get("delta_dxdt", 1e-2)
    H    = H_of_T(T, rhoh if include_h else 0.0, t_dep = t_dep)
    xdot = dx_dt(T, H, m = mchi, t_dep = t_dep, delta = delta)

    # collisions
    if params.get("no_Th_evolution", False):
        C_chi, C_A = collisions_noTh(params, nchi, nA, T, Th, svxxAA_temp= svxxAA_temp)
        Qh = 0.0
    else:
        C_chi, C_A, Qh, _, _, _, _ = collisions(params, nchi, nA, T, Th, svxxAA_temp = svxxAA_temp)

    # Time derivatives for number rxns
    dnchi_dt = -3.0 * H * nchi + C_chi
    dnA_dt   = -3.0 * H * nA   + C_A

    if params.get("no_Th_evolution", False):
        dTh_dt = - Th * xdot / x
    else:
        dTh_dt = dTh_dt_rel(nchi, nA, Th, mchi, mA, H, rhoh, Ph, Qh, dnchi_dt, dnA_dt)

    dnchi_dx = dnchi_dt /xdot
    dnA_dx   = dnA_dt   /xdot
    dTh_dx   = dTh_dt   /xdot

    # Return log-derivatives
    dln_nchi_dx = dnchi_dx / np.maximum(nchi, VAL_FLOOR)
    dln_nA_dx   = dnA_dx / np.maximum(nA, VAL_FLOOR)
    dln_Th_dx   = dTh_dx / np.maximum(Th, VAL_FLOOR)

    # Clip log-derivatives to keep numerical Jacobians finite in very stiff regimes.
    clip = float(params.get("max_abs_dlnydx", 1e8))
    dln_nchi_dx = np.clip(dln_nchi_dx, -clip, clip)
    dln_nA_dx   = np.clip(dln_nA_dx,   -clip, clip)
    dln_Th_dx   = np.clip(dln_Th_dx,   -clip, clip)

    if params.get("no_Th_evolution", False):
        dln_Th_dx = -1/x

    return np.array([dln_nchi_dx, dln_nA_dx, dln_Th_dx])


def compute_diagnostics(xs, sol_y, params):
    """
    Post-process a solution into rate ratios, for reading off *why* the
    abundance came out as it did.

    Every returned rate is divided by H, so the value 1 marks the point where
    that process decouples: Gamma/H >> 1 is efficient, << 1 is frozen out. The
    Q_*/(H rho) entries do the same for energy transfer rather than number
    changing. `nchi_over_nchieq` is the direct chemical-equilibrium check.

    Returns
    -------
    dict of arrays over `xs`, plus the `params` used.
    """
    ln_nchi, ln_nA, ln_Th = sol_y
    nchi = safe_exp(ln_nchi)
    nA = safe_exp(ln_nA)
    Th = safe_exp(ln_Th)

    mchi, mA = params["mchi"], params["mA"]
    gchi, gA = params["gchi"], params["gA"]
    gammaA = params["gamma_Aee"]
    t_dep = params.get("t_dep", True)
    include_h = params.get("include_hs_in_H", True)

    T = mchi / xs
    sv_xxAA = _sv(params["sv_xxAA"], Th)
    sv_xxee = _sv(params["sv_xxee"], T)

    # HS thermodynamics
    rhoh = rho_i_exact(nchi, mchi, Th) + rho_i_exact(nA, mA, Th)
    H    = H_of_T(T, rhoh if include_h else 0.0, t_dep = t_dep)


    # Equilibria at HS temperature
    nxeq_Th = neq_stable(mchi, gchi, Th)
    naeq_Th = neq_stable(mA,   gA,   Th)

    # Per-particle reaction rates (no cancellation) -
    Gamma_xAA_over_H = (sv_xxAA * nchi) / H
    Gamma_xSM_over_H = (sv_xxee * nchi) / H
    lorentz = _R(mA / np.maximum(Th, VAL_FLOOR))    # <m/E>, time dilation
    Gamma_Adec_over_H = (gammaA * lorentz) / H

    Cchi    = np.empty_like(xs)
    CA      = np.empty_like(xs)
    Qh      = np.empty_like(xs)
    Q_ann   = np.empty_like(xs)
    Q_dec   = np.empty_like(xs)
    Q_el    = np.empty_like(xs)
    Gkin    = np.empty_like(xs)

    for i in range(xs.size):
        Cchi[i], CA[i], Qh[i], Q_ann[i], Q_dec[i], Q_el[i], Gkin[i] = collisions(
            params, nchi[i], nA[i], T[i], Th[i]
        )

    Gamma_chem_over_H = np.abs(Cchi) / (H * np.maximum(np.abs(nchi - nxeq_Th), VAL_FLOOR))

    # --- energy-transfer diagnostics ---
    # Guarded: with elastic transfer disabled these are identically zero, and
    # rhoh can underflow to a denormal deep in the tail. Diagnostics must never
    # be the thing that fails a run, so fall back to zeros.
    try:
        rhoh_floor = np.maximum(rhoh, VAL_FLOOR)
        Q_ann_over_Hrho = np.abs(Q_ann) / (H * rhoh_floor)
        Q_dec_over_Hrho = np.abs(Q_dec) / (H * rhoh_floor)
        Q_el_over_Hrho  = np.abs(Q_el)  / (H * rhoh_floor)

        Gamma_el_over_H = np.abs(Gkin) / np.maximum(H, VAL_FLOOR)
        Q_total_over_Hrho = np.abs(Qh) / (H * rhoh_floor)
    except (FloatingPointError, ZeroDivisionError, ValueError):
        Q_ann_over_Hrho = np.zeros_like(xs)
        Q_dec_over_Hrho = np.zeros_like(xs)
        Q_el_over_Hrho  = np.zeros_like(xs)
        Gamma_el_over_H = np.zeros_like(xs)
        Q_total_over_Hrho = np.zeros_like(xs)

    return {
        "x": xs, "T": T, "Th": Th, "H": H,
        # number-changing rates
        "Gamma_xAA_over_H": Gamma_xAA_over_H,
        "Gamma_xSM_over_H": Gamma_xSM_over_H,
        "Gamma_Adec_over_H": Gamma_Adec_over_H,
        "Gamma_chem_over_H": Gamma_chem_over_H,
        "nchi_over_nchieq": nchi / np.maximum(nxeq_Th, VAL_FLOOR),
        "nA_over_nAeq":     nA   / np.maximum(naeq_Th, VAL_FLOOR),

        # energy stuff
        "Q_ann_over_Hrho": Q_ann_over_Hrho,
        "Q_dec_over_Hrho": Q_dec_over_Hrho,
        "Q_el_over_Hrho":  Q_el_over_Hrho,
        "Gamma_el_over_H": Gamma_el_over_H,
        "Q_total_over_Hrho": Q_total_over_Hrho,
        "params": params,
    }



def evolve(params, x_initial, x_final, y0, xs, log_space = True, rtol = 1e-7, atol=1e-14,
            method = 'Radau', svxxAA_temp = 'HS'):
    def _solve(fun, t_span, y_init, args, t_eval, meth, rt, at):
        # Stiff solvers evaluate numerical Jacobians with tiny perturbations;
        # allow harmless fp over/underflow there instead of raising.
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            return solve_ivp(
                fun,
                t_span,
                y_init,
                args=args,
                t_eval=t_eval,
                method=meth,
                rtol=rt,
                atol=at,
            )

    if log_space:
        y0 = np.asarray(y0, dtype=float)
        if np.any(y0 <= 0.0) or np.any(~np.isfinite(y0)):
            raise ValueError("All initial conditions must be finite and > 0 in log-space mode.")

        u0 = np.log(y0)
        # Pass svxxAA_temp via args so rhs_logx actually receives it.
        sol = _solve(
            rhs_logx,
            (x_initial, x_final),
            u0,
            (params, svxxAA_temp),
            xs,
            method,
            rtol,
            atol,
        )

        if (not sol.success) and ("Required step size is less than spacing between numbers" in str(sol.message)) and (method.upper() != "LSODA"):
            sol_try = _solve(
                rhs_logx,
                (x_initial, x_final),
                u0,
                (params, svxxAA_temp),
                xs,
                "LSODA",
                max(rtol, 1e-6),
                max(atol, 1e-9),
            )
            if sol_try.success and np.isfinite(sol_try.y).all():
                sol = sol_try
    else:
        y0 = np.asarray(y0, dtype=float)
        sol = _solve(
            rhs_logx,
            (x_initial, x_final),
            y0,
            (params, svxxAA_temp),
            xs,
            method,
            rtol,
            atol,
        )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    if not np.isfinite(sol.y).all():
        raise RuntimeError(
            "Integration produced non-finite values. "
            "Try lowering stiffness (set params['Gamma_H_max'], e.g. 20-100) "
            "or relax tolerances / adjust initial conditions."
        )

    x_arr = sol.t
    diag = compute_diagnostics(x_arr, sol.y, params)

    if log_space:
        soly = safe_exp(sol.y)
    else:
        soly = sol.y
    return {"sol": soly, "x_arr": x_arr, "diag": diag}
