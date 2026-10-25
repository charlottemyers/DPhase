"""
Post-processing diagnostics for the phase-space solver output: how far an
evolved distribution is from a thermal (Maxwell-Boltzmann) shape.

Each snapshot f(p) is compared to the MB distribution f_eq that has the same
number density and the same mean energy per particle. Matching the mean
energy fixes the effective temperature T_eff of f_eq; matching the number
density fixes its normalization. Then

  - Tratio     = T_eff / T measures kinetic decoupling from the SM bath,
  - kurt_ratio = (<p^4>/<p^2>^2)_f / (<p^4>/<p^2>^2)_eq measures the shape. It
    is independent of normalization and temperature by construction: 1 for a
    thermal shape, > 1 (< 1) for a distribution more (less) tail-heavy,
  - Evar_ratio = Var(E)_f / Var(E)_eq compares the energy spread to thermal,
  - KL         = (1/n) int f ln(f / f_eq) is the entropy deficit per particle
    in nats; >= 0, and 0 only for f = f_eq,
  - TV         = (1/2n) int |f - f_eq| is the fraction of particles that would
    have to move to make f thermal.

"""

import numpy as np
from scipy.optimize import brentq


def _mean(W, g, q):
    """Weighted mean of q over the distribution g: sum W g q / sum W g."""
    return np.sum(W * g * q) / np.sum(W * g)


def _mb_shape(E, T_eff):
    """Unnormalized MB e^{-E/T_eff}, shifted by E_min to avoid underflow."""
    return np.exp(-(E - E.min()) / T_eff)


def mb_reference(f, E, W, T_guess):
    """
    The MB distribution matched to f in number density and mean energy per
    particle, on the same momentum grid.

    Parameters
    ----------
    f       : distribution on the grid
    E       : energies of the grid points [GeV]
    W       : per-bin phase-space weight, with n = sum(W f)
    T_guess : scale for the root bracket, normally the SM temperature [GeV]

    Returns
    -------
    (f_eq, T_eff), with sum(W f_eq) = sum(W f); (None, NaN) if f is empty or
    no MB on this grid matches its mean energy (e.g. f piled up against a
    grid edge)
    """
    n = np.sum(W * f)
    if not n > 0.0:
        return None, np.nan
    target = _mean(W, f, E)

    def mismatch(log_T):
        return _mean(W, _mb_shape(E, np.exp(log_T)), E) - target

    lo, hi = np.log(1e-4 * T_guess), np.log(1e4 * T_guess)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        f_lo, f_hi = mismatch(lo), mismatch(hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.0:
            return None, np.nan
        T_eff = float(np.exp(brentq(mismatch, lo, hi, xtol=1e-10)))

    shape = _mb_shape(E, T_eff)
    return n * shape / np.sum(W * shape), T_eff


def thermality_metrics(T_vals, snapshots, state, species="chi",
                       x_min=None, x_max=None, gstar_func=None):
    """
    Deviation from a thermal shape of one species, at each snapshot.

    Parameters
    ----------
    T_vals       : SM temperatures of the snapshots [GeV], as returned by
                   `solve_BE`
    snapshots    : list of {species_name: f} on the comoving grid, from
                   `solve_BE`
    state        : PhaseSpaceState used for the run (grid and masses)
    species      : which species to analyze
    x_min, x_max : optional range in x = m_chi / T to keep
    gstar_func   : g_*(T) used to map comoving to physical momenta; None uses
                   state.gstar_func, which is what the solver used

    Returns
    -------
    dict of arrays, one entry per kept snapshot:
        x          : m_chi / T
        T          : SM temperature [GeV]
        T_eff      : temperature of the matched MB [GeV]
        Tratio     : T_eff / T
        kurt_ratio, Evar_ratio, KL, TV : see the module docstring
    Snapshots where the species is absent, or no matching MB exists, are
    skipped.
    """
    if gstar_func is None:
        gstar_func = state.gstar_func
    T_vals = np.asarray(T_vals, dtype=float)
    m_chi = state.species["chi"].mass_GeV
    m = state.species[species].mass_GeV
    dlogp = np.asarray(state.grid.dlogp, dtype=float)

    keys = ("x", "T", "T_eff", "Tratio", "kurt_ratio", "Evar_ratio", "KL", "TV")
    out = {key: [] for key in keys}
    for k, T in enumerate(T_vals):
        T = float(T)
        x = m_chi / T
        if (x_min is not None and x < x_min) or (x_max is not None and x > x_max):
            continue

        f = np.maximum(np.asarray(snapshots[k][species], dtype=float), 0.0)
        p = state.grid.p_phys(T, gstars_func=gstar_func)
        E = np.sqrt(p**2 + m**2)
        W = p**3 * dlogp / (2.0 * np.pi**2)        # n = sum(W f), per dof

        f_eq, T_eff = mb_reference(f, E, W, T)
        if f_eq is None:
            continue
        n = np.sum(W * f)

        kurt = ((_mean(W, f, p**4) / _mean(W, f, p**2)**2)
                / (_mean(W, f_eq, p**4) / _mean(W, f_eq, p**2)**2))
        Evar_f = _mean(W, f, (E - _mean(W, f, E))**2)
        Evar_eq = _mean(W, f_eq, (E - _mean(W, f_eq, E))**2)
        both = (f > 0.0) & (f_eq > 0.0)
        KL = np.sum(W[both] * f[both] * np.log(f[both] / f_eq[both])) / n
        TV = 0.5 * np.sum(W * np.abs(f - f_eq)) / n

        for key, val in zip(keys, (x, T, T_eff, T_eff / T, kurt,
                                   Evar_f / Evar_eq, KL, TV)):
            out[key].append(val)

    return {key: np.asarray(val, dtype=float) for key, val in out.items()}
