"""
Container for the evolving distribution functions and everything the solver
needs with them.

`PhaseSpaceState` is a mutable class, populated in two stages:

  1. construction  -- grid and species list
  2. precomputation -- couplings, the temperature grid, and the collision
                       kernels/caches tabulated on it

Stage 2 is done by hand at the call site (see `examples/example.ipynb`)
and every attribute it sets is declared in `__init__` below, initialised to None.
"""

import numpy as np

import dphase
from dphase.cosmology import entropy_density
from dphase.grid import PhaseSpaceGrid, PhaseSpaceSpecies


class PhaseSpaceState:
    """
    Holds f(ptilde) for each species, plus the precomputed collision data.

    f is stored on the comoving momentum grid, so it is not redistributed by
    expansion. Collision terms need physical momenta, which the solver obtains
    from `self.grid.p_phys(T, self.gstar_func)` at each temperature.
    """

    def __init__(self, grid: PhaseSpaceGrid, species: list[PhaseSpaceSpecies]):
        # --- set at construction ---
        self.grid = grid
        self.species = {sp.name: sp for sp in species}
        self.f = {sp.name: np.zeros_like(grid.ptilde) for sp in species}

        # --- model parameters  ---
        self.epsilon = None                              # kinetic mixing
        self.alphaD = None                               # dark fine structure constant
        self.gstar_func = dphase.cosmology.gstar_interp  # g_*(T) function for SM plasma
        self.Nmu = 8                                     # Gauss-Legendre nodes for angular integrals

        # --- temperature grid the kernels below are tabulated on ---
        self.T_grid = None       # shape (NT,)

        # --- chi chibar -> f fbar, from kernels.annihilation_sm ---
        self.K_grid_xxff = None  # (NT, Np, Np)

        # --- chi chibar <-> A A, from kernels.annihilation_hidden ---
        self.K_grid_xxAA = None      # (NT, Np, Np) loss kernel, chi side
        self.K_grid_AAxx = None      # (NT, Np, Np) loss kernel, A side
        self.gain_caches_xxAA = None  # list of NT gain-cache dicts

        # --- chi f -> chi f, from kernels.elastic_sm   ---
        self.gamma_grid_chi = None   # momentum exchange rate [GeV], per T
        self.gamma_grid_A = None

        # --- chi A -> chi A dark Compton, from kernels.elastic_hidden ---
        self.gain_caches_xAxA = None  # list of NT event-cache dicts

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def set_initial_f(self, name, f_ptilde):
        """Set the initial distribution for one species, on the comoving grid."""
        f_ptilde = np.asarray(f_ptilde, dtype=float)
        if f_ptilde.shape != self.f[name].shape:
            raise ValueError(
                f"f for {name!r} has shape {f_ptilde.shape}, "
                f"expected {self.f[name].shape}")
        self.f[name] = f_ptilde.copy()

    # ------------------------------------------------------------------
    # Derived quantities
    #
    # All three integrals below are trapezoidal in physical momentum, which
    # is why each one needs T: the comoving grid maps to a different set of
    # physical momenta at every temperature.
    # ------------------------------------------------------------------
    def number_density(self, name, T):
        """n = g/(2 pi^2) int dp p^2 f(p)   [GeV^3]"""
        sp = self.species[name]
        p = self.grid.p_phys(T, gstars_func=self.gstar_func)
        return sp.dof / (2.0 * np.pi**2) * np.trapezoid(p**2 * self.f[name], p)

    def energy_density(self, name, T):
        """rho = g/(2 pi^2) int dp p^2 E(p) f(p),  E = sqrt(p^2 + m^2)  [GeV^4]"""
        sp = self.species[name]
        p = self.grid.p_phys(T, gstars_func=self.gstar_func)
        E = np.sqrt(p**2 + sp.mass_GeV**2)
        return sp.dof / (2.0 * np.pi**2) * np.trapezoid(p**2 * E * self.f[name], p)

    def total_DM_number_density(self, T):
        """
        n_chi + n_chibar. The solver evolves f_chi only; with no particle
        -antiparticle asymmetry the two populations are identical, so this is
        just twice n_chi. This -- not `number_density("chi", T)` -- is what
        goes into the relic abundance.
        """
        return 2.0 * self.number_density("chi", T)

    def s_SM(self, T):
        """
        SM entropy density at T, using *this state's* g_* function rather than
        the module-level tabulation -- they can differ if a custom g_*(T) was
        supplied. Needed to convert a number density into a comoving yield.
        """
        return entropy_density(T, self.gstar_func(T))
