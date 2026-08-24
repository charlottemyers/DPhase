"""
DPhase -- phase-space Boltzmann solver for dark photon dark matter.

Solves the momentum-resolved Boltzmann equation for a Dirac fermion chi
coupled to the Standard Model through a kinetically mixed dark photon Z_D,
tracking the distribution function f_chi(p, T) rather than the number density
alone. Covers both freeze-out and freeze-in regimes.

Two solvers are provided:

  `solve_BE`  -- the phase-space solver. Evolves f_chi(p, T) and f_A(p, T) on
                 a comoving momentum grid with no assumption that either
                 species keeps a thermal shape.
  `cbe`       -- the number-density solver, tracking (n_chi, n_A, T_hidden)
                 under an assumed Maxwell-Boltzmann shape. Much cheaper, and
                 the baseline the phase-space solver is validated against in
                 the limit where kinetic equilibrium holds.

Everything in this package except `model.py` is independent of the particular
dark sector model; see that module's docstring to swap in a different mediator.

Basic usage
-----------
>>> import numpy as np
>>> import dphase
>>> chi = dphase.PhaseSpaceSpecies(name="chi", mass_GeV=10.0, dof=2.0)
>>> A = dphase.PhaseSpaceSpecies(name="A", mass_GeV=5.0, dof=3.0)
>>> grid = dphase.PhaseSpaceGrid(ptilde_min=0.1, ptilde_max=1e4, Np=120,
...                             T_ref=200.0)
>>> state = dphase.PhaseSpaceState(grid, [chi, A])

The state must then be populated with couplings and precomputed collision
kernels before `solve_BE` can run -- see `examples/example.ipynb`.
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Public API.
#
# Only the names below are considered stable. Everything else stays reachable
# at dphase.collisions.<...>, dphase.model.<...> etc. for anyone who wants the
# internals, but may be renamed without notice.
# ---------------------------------------------------------------------------

from .grid import PhaseSpaceGrid, PhaseSpaceSpecies
from .state import PhaseSpaceState
from .solver import solve_BE
from .cosmology import (H_of_T, H_RD, t_of_T_RD, gstar_interp, s_SM,
                        entropy_density, Y, relic_abundance)

__all__ = [
    # phase-space solver
    "PhaseSpaceGrid",
    "PhaseSpaceSpecies",
    "PhaseSpaceState",
    "solve_BE",
    # background cosmology
    "H_of_T",
    "H_RD",
    "t_of_T_RD",
    "gstar_interp",
    "s_SM",
    "entropy_density",
    "Y",
    "relic_abundance",
    "__version__",
]
