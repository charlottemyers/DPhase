"""
Comoving momentum grid and the species records that live on it.

The solver works in comoving momentum ptilde = a(t) p, which is constant
under free expansion. That choice removes the redshift drift term from the
Boltzmann equation: with ptilde as the coordinate, a free-streaming
distribution is static, and every df/dt the solver computes is a
genuine collision effect.

The price is that collision rates depend on physical momentum, so every
kernel has to be rebuilt at each temperature -- see `PhaseSpaceGrid.p_phys`
and the  builders in `dphase.kernels`.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PhaseSpaceSpecies:
    """
    One evolving species.

    name            : key used to index `PhaseSpaceState.f`. Solver looks
                      for the literal names "chi" and "A".
    mass_GeV        : rest mass [GeV]
    dof             : internal degrees of freedom g. Use 2 for chi
                      (2 spin states of the Dirac fermion; the
                      antiparticle is a separate population, accounted for by
                      `PhaseSpaceState.total_DM_number_density`)
                      Use 3 for the DP's polarizations.
    decay_width_GeV : total width [GeV]; 0 for stable species.
    """
    name: str
    mass_GeV: float
    dof: float
    decay_width_GeV: float = 0.0


def log_bin_edges(p_centers):
    """
    Bin widths for a log-spaced grid given its centers.

    Interior edges are the geometric midpoints between neighboring centers.
    The two outermost edges are reflected outward so that the first and last
    bins get a width comparable to their neighbors.

    Parameters
    ----------
    p_centers : strictly positive, ascending, shape (Np,)

    Returns
    -------
    dlogp : bin widths in log p, shape (Np,)
    """
    p = np.asarray(p_centers, dtype=float)
    if not np.all(p > 0):
        raise ValueError("momentum grid centers must be strictly positive")

    logp = np.log(p)
    edges = np.empty(logp.size + 1, dtype=float)

    # element-wise averaging of neighboring log-centers
    edges[1:-1] = 0.5 * (logp[1:] + logp[:-1])

    # (edges[1] - edges[0]) = how far the first bin extends below the first center, so reflect that distance outward
    edges[0] = logp[0] - (edges[1] - logp[0])
    edges[-1] = logp[-1] + (logp[-1] - edges[-2])
    return np.diff(edges)


class PhaseSpaceGrid:
    """
    Log-spaced comoving momentum grid, ptilde = a(t) * p_phys.

    Parameters
    ----------
    ptilde_min, ptilde_max : grid extent in comoving momentum [GeV]
    Np                     : number of bins
    T_ref                  : reference temperature, where a = 1 by definition.
                             Comoving and physical momentum coincide at T_ref,
                             so choosing T_ref as the initial temperature
                             makes the initial condition easy to write down.

    Attributes
    ----------
    ptilde     : bin centers, shape (Np,)
    log_ptilde : log of the centers
    dlogp      : bin widths in log p, shape (Np,). Every phase-space integral
                 in the code is a sum over bins weighted by p^3 * dlogp.
    """

    def __init__(self, ptilde_min, ptilde_max, Np, T_ref):
        if not ptilde_min > 0:
            raise ValueError("ptilde_min must be > 0 (the grid is log-spaced)")
        if not ptilde_max > ptilde_min:
            raise ValueError("ptilde_max must exceed ptilde_min")

        self.ptilde = np.geomspace(ptilde_min, ptilde_max, Np)
        self.log_ptilde = np.log(self.ptilde)
        self.dlogp = log_bin_edges(self.ptilde)
        self.T_ref = float(T_ref)
        self._Np = Np

    def a_of_T(self, T, gstars_func=None):
        """
        Scale factor, normalised to a(T_ref) = 1.

        Comoving entropy conservation gives a T g_{*s}^{1/3} = const, hence

            a(T) = (T_ref / T) (g_{*s}(T_ref) / g_{*s}(T))^{1/3}.

        With `gstars_func=None` the g_* factor is dropped and a = T_ref / T.
        """
        T = float(T)
        if gstars_func is None:
            return self.T_ref / T

        gS_T = float(gstars_func(T))
        gS_ref = float(gstars_func(self.T_ref))
        return (self.T_ref / T) * (gS_ref / gS_T) ** (1.0 / 3.0)

    def p_phys(self, T, gstars_func=None):
        """
        Physical momentum of each comoving bin at temperature T, p = ptilde / a.

        The comoving centers are fixed for the whole run, so this is where the
        redshifting of momentum by the expansion enters. Every collision term
        must be evaluated on these physical momenta, NOT on `self.ptilde`.

        Returns
        -------
        p_phys [GeV], shape (Np,)
        """
        return self.ptilde / self.a_of_T(T, gstars_func=gstars_func)
