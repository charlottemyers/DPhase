"""
Coupled Boltzmann equation (cBE) solver.

Where `dphase.solver` evolves the full f(p, T), this module assumes both dark
species keep a Maxwell-Boltzmann shape and tracks only three numbers:

    u = [ln n_chi, ln n_ZD, ln T_hidden]

as functions of x = m_chi / T.

The hidden-sector temperature T_hidden is a genuine dynamical variable here,
not assumed equal to the SM temperature T.


Params dictionary
-----------------
Required : mchi, mZD, gchi, gZD, epsilon
           sv_xxZDZD, sv_xxee, sv_xxWW : <sigma v> for chi chibar -> Z_D Z_D,
               -> f fbar (summed over f) and -> W+ W-; each a scalar or a
               callable of temperature.
           gamma_ZDff : total Z_D -> f fbar width [GeV] (`model.gammaZD_allf_mixing`)
           gamma_T   : momentum-exchange rate gamma(T) [GeV] for chi scattering
               off the SM bath, scalar or callable of T. Required when elastic=True.
Optional : t_dep, include_hs_in_H, no_Th_evolution, elastic (default True),
           delta_dxdt, max_abs_dlnydx,
           xxff_forward_at_Th (default True, see `_number_collision_terms`),
           in_medium_eps_eff (thermal correction to Z_D <-> f fbar, see
               `in_medium.cbe_decay_factors`)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import kve

from dphase.constants import VAL_FLOOR
from dphase.in_medium import cbe_decay_factors
from dphase.cosmology import (
    H_of_T,
    dln_gstars_SM_dlnT,
    ln_neq,
    safe_exp,
)


def dx_dt(T, H, m, t_dep = True, delta = 1e-6):
    """
    dx/dt for x = m / T, including the change in g_{*s}.

    Entropy conservation gives a T g_{*s}^{1/3} = const, so

        dx/dt = H x / [1 + (1/3) d ln g_{*s} / d ln T],

    which reduces to H x when g_{*s} is constant.

    Parameters
    ----------
    T     : SM temperature [GeV]
    H     : Hubble rate at T [GeV]
    m     : mass defining x, normally m_chi [GeV]
    t_dep : include d ln g_{*s} / d ln T; if False the correction is dropped
    delta : relative step in T for the finite-difference derivative

    Returns
    -------
    dx/dt [GeV]
    """
    x = m / T
    corr = 1.0 + (1.0/3.0) * dln_gstars_SM_dlnT(T, t_dep, delta)
    return H * x / corr


def rho_i_exact(n_i, m_i, Th):
    """
    Exact energy density of a Maxwell-Boltzmann species at temperature Th,

        rho_i = n_i [ m_i K_1(z)/K_2(z) + 3 Th ],   z = m_i / Th,

    i.e. n_i times the mean energy per particle. Valid at any z.

    Parameters
    ----------
    n_i : number density [GeV^3]
    m_i : mass [GeV]
    Th  : hidden-sector temperature [GeV]

    Returns
    -------
    rho_i [GeV^4]
    """
    Th   = np.maximum(Th, VAL_FLOOR)        # avoid z = inf / division by zero
    z    = m_i / Th
    # Use scaled Kv: Kv(z) = e^{-z} kve(order, z).
    k2   = kve(2, z)
    ratio = kve(1, z) / np.maximum(k2, VAL_FLOOR)
    return n_i * (m_i * ratio + 3.0 * Th)

def P_i_exact(n_i, Th):
    """
    Exact pressure of a Maxwell-Boltzmann species. The MB equation of state
    is exactly the ideal gas, P = n T, for any mass.

    Parameters
    ----------
    n_i : number density [GeV^3]
    Th  : hidden-sector temperature [GeV]

    Returns
    -------
    P_i [GeV^4]
    """
    return n_i * Th


def _R(z):
    """
    Bessel ratio K_1(z)/K_2(z), which is the thermally averaged <m/E> for a
    Maxwell-Boltzmann species with z = m/T. Tends to 1 in the
    non-relativistic limit and to z/2 in the relativistic one.

    Computed from the exponentially scaled kve so that the e^{-z} factors
    cancel exactly instead of underflowing at large z.

    Parameters
    ----------
    z : m / T, scalar or array

    Returns
    -------
    K_1(z) / K_2(z), same shape as z
    """
    return kve(1, z) / kve(2, z)

def _dR_dz(z, rel_step=1e-6):
    """
    dR/dz for R(z) = K_1(z)/K_2(z), by centered finite difference. Enters
    the hidden-sector heat capacity (`heat_capacity_i`).

    Parameters
    ----------
    z        : m / T, scalar or array
    rel_step : relative step in z (with an absolute floor of 1e-6)

    Returns
    -------
    dR/dz, same shape as z
    """
    dz = np.maximum(rel_step * z, 1e-6)
    Rp = kve(1, z+dz)/kve(2, z+dz)
    Rm = kve(1, z-dz)/kve(2, z-dz)
    return (Rp - Rm) / (2.0*dz)


def heat_capacity_i(n_i, m_i, Th):
    """
    Heat capacity of one MB species at fixed number density,

        d rho_i / d T_h |_{n_i} = n_i [3 - z^2 R'(z)],   z = m_i / T_h,

    from differentiating rho_i = n_i (m_i R(z) + 3 T_h).

    Parameters
    ----------
    n_i : number density [GeV^3]
    m_i : mass [GeV]
    Th  : hidden-sector temperature [GeV]

    Returns
    -------
    d rho_i / d T_h [GeV^3]
    """
    z = m_i / np.maximum(Th, VAL_FLOOR)
    return n_i * (3.0 - z*z*_dR_dz(z))


def dTh_dt_rel(nchi, nZD, Th, mchi, mZD, H, rhoh, Ph, Qh, dnchi_dt, dnZD_dt):
    """
    dT_h/dt from hidden-sector energy conservation.

    The hidden sector obeys d rho_h/dt = -3 H (rho_h + P_h) + Q_h. Writing
    rho_h = sum_i n_i h_i(T_h), with h_i = m_i R(z_i) + 3 T_h the mean energy
    per particle, and separating the change in composition from the change
    in temperature gives

        dT_h/dt = [ Q_h - sum_i h_i dn_i/dt - 3 H (rho_h + P_h) ] / C_h,

    where C_h = sum_i d rho_i / d T_h at fixed n_i. The chi terms are doubled for chi + chibar;
    nchi is the chi density alone.

    Parameters
    ----------
    nchi, nZD          : number densities [GeV^3]
    Th                 : hidden-sector temperature [GeV]
    mchi, mZD          : masses [GeV]
    H                  : Hubble rate [GeV]
    rhoh, Ph           : hidden-sector energy density and pressure [GeV^4]
    Qh                 : energy injection rate into the hidden sector [GeV^5],
                         from `collisions`
    dnchi_dt, dnZD_dt  : total dn/dt for each species, including the -3 H n
                         dilution [GeV^4]

    Returns
    -------
    dT_h/dt [GeV^2]
    """
    zc = mchi /np.maximum(Th, 1e-300)
    zZD = mZD   / np.maximum(Th, 1e-300)

    Rchi = _R(zc)
    RZD = _R(zZD)

    # hidden-sector energies per particle
    h_chi = mchi * Rchi + 3.0 * Th
    h_ZD   = mZD   * RZD   + 3.0 * Th

    # C_h: heat capacity at constant composition
    # multiply the chi contribution by 2 to account for chi and chi-bar
    Ch = 2.0 * heat_capacity_i(nchi, mchi, Th) + heat_capacity_i(nZD, mZD, Th)
    Ch = max(Ch, 1e-300)

    numer = Qh - (2.0* h_chi * dnchi_dt + h_ZD * dnZD_dt) - 3.0 * H * (rhoh + Ph)
    return numer / Ch


def _sv(value, T):
    """
    Evaluate a params entry that may be a constant or a callable of
    temperature (cross sections, gamma_T).

    Parameters
    ----------
    value : scalar, or callable T -> scalar
    T     : temperature to evaluate a callable at [GeV]

    Returns
    -------
    value(T) if callable, else value
    """
    return value(T) if callable(value) else value


def _in_medium_factors(params, T, Th):
    """
    Thermal in-medium factors for the Z_D <-> f fbar decay terms, from
    `in_medium.cbe_decay_factors`. Returns early with all ones when the
    correction is off.

    Parameters
    ----------
    params : cBE parameter dict; reads params["in_medium_eps_eff"]
    T      : SM temperature [GeV]
    Th     : hidden-sector temperature [GeV]

    Returns
    -------
    (r_dec_gain, r_dec_loss, r_E_gain, r_E_loss): the gain (inverse decay)
    factors multiply the terms thermal at T, the loss (decay) factors those
    thermal at Th. r_dec scales the number equation, r_E the energy injection.
    """
    if not params.get("in_medium_eps_eff", False):
        return 1.0, 1.0, 1.0, 1.0
    return cbe_decay_factors(params, T, Th)


def _number_collision_terms(params, nchi, nZD, T, Th, svxxZDZD_temp="HS"):
    """
    Collision terms for the two number densities, shared by `collisions` and
    `collisions_noTh`.

    chi chibar <-> SM has its two directions in different baths: the forward
    annihilation is a hidden-sector pair, thermal at T_h, while the inverse
    is an SM pair, thermal at T.

    The factor of 2 on the first C_ZD term counts the two dark photons produced
    per annihilation. <m/E> = K_1(z)/K_2(z) is the thermally averaged time
    dilation; with params["in_medium_eps_eff"] the two decay directions are also
    multiplied by the in-medium factors.

    Parameters
    ----------
    params        : cBE parameter dict
    nchi, nZD     : number densities [GeV^3]; nchi is chi alone, not chi + chibar
    T, Th         : SM and hidden-sector temperatures [GeV]
    svxxZDZD_temp : temperature at which the dark-sector annihilation cross
                    section is evaluated: "HS" for T_hidden (both incoming
                    legs are dark) or anything else for the SM T, for comparison

    Returns
    -------
    C_chi, C_ZD, nchi_eq_SM, nZD_eq_SM, sv_SM, sv_HS
    with sv_SM / sv_HS = (<sv>_ff + <sv>_WW) at T and at the forward temperature.
    """
    mchi, mZD = params["mchi"], params["mZD"]
    gchi, gZD = params["gchi"], params["gZD"]

    sv_xxZDZD = _sv(params["sv_xxZDZD"], Th if svxxZDZD_temp == "HS" else T)

    # chi chibar -> SM: inverse at T, forward at T_h
    T_fwd = Th if params.get("xxff_forward_at_Th", True) else T
    sv_xxff_SM = _sv(params["sv_xxee"], T)
    sv_xxWW_SM = _sv(params["sv_xxWW"], T)
    sv_xxff_HS = _sv(params["sv_xxee"], T_fwd)
    sv_xxWW_HS = _sv(params["sv_xxWW"], T_fwd)
    gamma_ZDff = params["gamma_ZDff"]

    # Detailed balance in log space:
    #   ln B = 2 (ln n_chi,eq^Th - ln n_ZD,eq^Th) + 2 ln n_ZD
    ln_balance = (2.0 * (ln_neq(mchi, gchi, Th) - ln_neq(mZD, gZD, Th))
                  + 2.0 * np.log(max(nZD, VAL_FLOOR)))
    balance = safe_exp(ln_balance)          # underflows cleanly to 0

    nchi_eq_SM = safe_exp(ln_neq(mchi, gchi, T))
    nZD_eq_SM = safe_exp(ln_neq(mZD, gZD, T))

    # Time dilation <m/E> = K1/K2, at the hidden and the SM temperature,
    # times the in-medium factors (all 1 unless in_medium_eps_eff).
    r_dec_gain, r_dec_loss, _, _ = _in_medium_factors(params, T, Th)
    lorentz_HS = _R(mZD / np.maximum(Th, VAL_FLOOR)) * r_dec_loss
    lorentz_SM = _R(mZD / np.maximum(T, VAL_FLOOR)) * r_dec_gain

    hs = sv_xxZDZD * (nchi**2 - balance)
    ann_ff = sv_xxff_HS * nchi**2 - sv_xxff_SM * nchi_eq_SM**2
    ann_WW = sv_xxWW_HS * nchi**2 - sv_xxWW_SM * nchi_eq_SM**2
    dec = gamma_ZDff * (nZD * lorentz_HS - nZD_eq_SM * lorentz_SM)

    C_chi = -hs - ann_ff - ann_WW
    C_ZD = 2.0 * hs - dec

    return (C_chi, C_ZD, nchi_eq_SM, nZD_eq_SM,
            sv_xxff_SM + sv_xxWW_SM, sv_xxff_HS + sv_xxWW_HS)


def collisions(params, nchi, nZD, T, Th, svxxZDZD_temp="HS"):
    """
    Full collision terms: number densities and the hidden-sector energy
    injection Q_h that drives dT_hidden/dt.

    The three energy channels are

        Q_ann : chi chibar <-> f fbar, W+ W-. Each forward annihilation
                removes the mean energy of a hidden-sector pair, thermal at
                T_h; each inverse event injects that of a pair produced from
                the SM bath, which by detailed balance has the equilibrium
                spectrum at T.
        Q_dec : Z_D decay removes energy m_ZD per decay in the rest frame; the
                <m/E> time dilation cancels against <E> = m_ZD <E/m>.
        Q_el  : elastic chi f -> chi f drives T_hidden towards T,
                3 gamma(T) n_chi (T - T_h), with gamma from params["gamma_T"].
                Switched off by params["elastic"] = False.

    Parameters
    ----------
    Same as `_number_collision_terms`.

    Returns
    -------
    C_chi, C_ZD   : number-density collision terms [GeV^4]
    Q_h           : total energy injection Q_ann + Q_dec + Q_el [GeV^5]
    Q_ann, Q_dec, Q_el : the individual channels above [GeV^5]
    Gkin          : gamma(T) used for Q_el [GeV]; 0 if elastic is off
    """
    mchi, mZD = params["mchi"], params["mZD"]

    C_chi, C_ZD, nchi_eq_SM, nZD_eq_SM, sv_SM, sv_HS = _number_collision_terms(
        params, nchi, nZD, T, Th, svxxZDZD_temp)

    # energy injection; in-medium factors are all 1 unless in_medium_eps_eff
    _, _, r_E_gain, r_E_loss = _in_medium_factors(params, T, Th)
    Q_dec = -mZD * params["gamma_ZDff"] * (nZD * r_E_loss - nZD_eq_SM * r_E_gain)

    # mean chi energy of the annihilating (HS, T_h) and produced (SM, T) pairs
    eps_bar_HS = mchi * _R(mchi / np.maximum(Th, VAL_FLOOR)) + 3.0 * Th
    eps_bar_SM = mchi * _R(mchi / np.maximum(T, VAL_FLOOR)) + 3.0 * T
    Q_ann = -2.0 * (eps_bar_HS * sv_HS * nchi**2
                    - eps_bar_SM * sv_SM * nchi_eq_SM**2)

    if params.get("elastic", True):
        gamma_T = params.get("gamma_T", params.get("gamma_el", None))
        if gamma_T is None:
            raise KeyError("elastic=True requires params['gamma_T'] (or 'gamma_el')")
        Gkin = float(_sv(gamma_T, T))
        # only chi scatters off the SM bath; Z_D does not.
        Q_el = 3.0 * Gkin * (T - Th) * nchi
    else:
        Gkin = 0.0
        Q_el = 0.0

    return C_chi, C_ZD, Q_ann + Q_dec + Q_el, Q_ann, Q_dec, Q_el, Gkin


def collisions_noTh(params, nchi, nZD, T, Th, svxxZDZD_temp="HS"):
    """
    Number-density collision terms only, without the energy injection. Used
    when params["no_Th_evolution"] is True, so T_hidden is pinned to T and
    Q_h is not needed.

    Parameters
    ----------
    Same as `_number_collision_terms`.

    Returns
    -------
    C_chi, C_ZD : number-density collision terms [GeV^4]
    """
    C_chi, C_ZD = _number_collision_terms(
        params, nchi, nZD, T, Th, svxxZDZD_temp)[:2]
    return C_chi, C_ZD


# ----------------------------
# x-domain RHS: d/dx = (1/(dx/dt)) d/dt
# ----------------------------
def rhs_logx(x, u, params, svxxZDZD_temp = "HS"):
    """
    Right-hand side for `solve_ivp`: d/dx of u = [ln n_chi, ln n_ZD, ln T_h].

    Each step evaluates H, the collision terms, dn/dt = -3 H n + C and dT_h/dt (`dTh_dt_rel`),
    then converts to x with d/dx = (dx/dt)^{-1} d/dt. Each log-derivative is clipped
    to +- params["max_abs_dlnydx"] to keep stiff steps finite. With
    params["no_Th_evolution"], T_h = T and d ln T_h/dx = -1/x.

    Parameters
    ----------
    x             : m_chi / T
    u             : [ln n_chi, ln n_ZD, ln T_h], densities in GeV^3,
                    T_h in GeV; n_chi is chi alone, not chi + chibar
    params        : cBE parameter dict
    svxxZDZD_temp : passed through to `_number_collision_terms`

    Returns
    -------
    array [d ln n_chi/dx, d ln n_ZD/dx, d ln T_h/dx]
    """

    mchi = params["mchi"];  mZD = params["mZD"]
    t_dep = params.get("t_dep", True)
    include_h = params.get("include_hs_in_H", False)
    T    = mchi / x  # SM temperature

    ln_nchi, ln_nZD, ln_Th  = u
    nchi = safe_exp(ln_nchi) # represents chi population ONLY (not nchi + bar_nchi)
    nZD   = safe_exp(ln_nZD)

    # Prevent pathologies when stiff steps push logs to extreme values.
    ln_min_Th = np.log(VAL_FLOOR)
    if ln_Th < ln_min_Th:
        ln_Th = ln_min_Th

    if params.get("no_Th_evolution", False):
        Th = T
    else:
        Th = safe_exp(ln_Th)

    # HS energy density and pressure
    # multiply by 2 to account for both chi and chi-bar
    rhoh = 2.0*rho_i_exact(nchi, mchi, Th) + rho_i_exact(nZD, mZD, Th)
    Ph = 2.0*P_i_exact(nchi, Th) + P_i_exact(nZD, Th)

    # Hubble and x-dot
    delta = params.get("delta_dxdt", 1e-2)
    H    = H_of_T(T, rhoh if include_h else 0.0, t_dep = t_dep)
    xdot = dx_dt(T, H, m = mchi, t_dep = t_dep, delta = delta)

    # Collisions
    if params.get("no_Th_evolution", False):
        C_chi, C_ZD = collisions_noTh(params, nchi, nZD, T, Th, svxxZDZD_temp= svxxZDZD_temp)
        Qh = 0.0
    else:
        C_chi, C_ZD, Qh, _, _, _, _ = collisions(params, nchi, nZD, T, Th, svxxZDZD_temp = svxxZDZD_temp)

    # Time derivatives for number-changing processes
    dnchi_dt = -3.0 * H * nchi + C_chi
    dnZD_dt   = -3.0 * H * nZD   + C_ZD

    if params.get("no_Th_evolution", False):
        dTh_dt = - Th * xdot / x
    else:
        dTh_dt = dTh_dt_rel(nchi, nZD, Th, mchi, mZD, H, rhoh, Ph, Qh, dnchi_dt, dnZD_dt)

    dnchi_dx = dnchi_dt /xdot
    dnZD_dx   = dnZD_dt   /xdot
    dTh_dx   = dTh_dt   /xdot

    # Return log-derivatives
    dln_nchi_dx = dnchi_dx / np.maximum(nchi, VAL_FLOOR)
    dln_nZD_dx   = dnZD_dx / np.maximum(nZD, VAL_FLOOR)
    dln_Th_dx   = dTh_dx / np.maximum(Th, VAL_FLOOR)

    # Clip log-derivatives
    clip = float(params.get("max_abs_dlnydx", 1e8))
    dln_nchi_dx = np.clip(dln_nchi_dx, -clip, clip)
    dln_nZD_dx   = np.clip(dln_nZD_dx,   -clip, clip)
    dln_Th_dx   = np.clip(dln_Th_dx,   -clip, clip)

    if params.get("no_Th_evolution", False):
        dln_Th_dx = -1/x

    return np.array([dln_nchi_dx, dln_nZD_dx, dln_Th_dx])



def evolve(params, x_initial, x_final, y0, xs, rtol = 1e-7, atol=1e-14,
            method = 'Radau', svxxZDZD_temp = 'HS', max_step = None):
    """
    Integrate the cBE from x_initial to x_final, output at `xs`. The
    integration is done in u = ln y (see `rhs_logx`), which keeps the
    densities positive across many decades.

    Parameters
    ----------
    params             : cBE parameter dict (see module docstring)
    x_initial, x_final : integration range in x = m_chi / T
    y0                 : initial [n_chi, n_ZD, T_h], all finite and > 0
    xs                 : x values at which to return the solution
    rtol, atol         : solver tolerances
    method             : `solve_ivp` method
    svxxZDZD_temp      : passed through to `_number_collision_terms`
    max_step           : cap on the solver step in x.

    Returns
    -------
    dict with "sol", shape (3, len(x_arr)), rows n_chi, n_ZD, T_h, and
    "x_arr", the x values
    """
    if max_step is None:
        # resolve resonance when in-medium effects are considered
        max_step = (0.005 * params["mchi"] / params["mZD"]
                    if params.get("in_medium_eps_eff", False) else np.inf)

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
                max_step=max_step,
            )

    y0 = np.asarray(y0, dtype=float)
    if np.any(y0 <= 0.0) or np.any(~np.isfinite(y0)):
        raise ValueError("All initial conditions must be finite and > 0.")

    u0 = np.log(y0)
    sol = _solve(
        rhs_logx,
        (x_initial, x_final),
        u0,
        (params, svxxZDZD_temp),
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
            (params, svxxZDZD_temp),
            xs,
            "LSODA",
            max(rtol, 1e-6),
            max(atol, 1e-9),
        )
        if sol_try.success and np.isfinite(sol_try.y).all():
            sol = sol_try

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    if not np.isfinite(sol.y).all():
        raise RuntimeError(
            "Integration produced non-finite values. "
            "Try relaxing tolerances or adjusting initial conditions."
        )

    return {"sol": safe_exp(sol.y), "x_arr": sol.t}
