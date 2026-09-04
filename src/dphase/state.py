"""
Container for the evolving distribution functions and everything the solver
needs with them.

`PhaseSpaceState` is a mutable class, populated in two stages:

  1. construction   -- grid and species list
  2. precomputation -- couplings, the temperature grid, and the collision
                       kernels/caches tabulated on it

Stage 2 is done by hand at the call site (see `example.ipynb`)
and every attribute it sets is declared in `__init__` below.
"""

import numpy as np

import dphase
from dphase.cosmology import entropy_density
from dphase.grid import PhaseSpaceGrid, PhaseSpaceSpecies


class _ZeroDefaultRateGrid:
    """
    A per-temperature rate grid that reads as zeros until it is assigned.

    Some rate contributions are identically zero for the dark photon model
    A f -> A f elastic scattering, for instance, whose correct rate scales as
    epsilon^4 and is dropped by `kernels.elastic_sm`. Rather than make the
    caller build and assign an array of zeros by hand, those attributes default
    to zeros here, so only the non-trivial grids need assigning.
    """

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = getattr(obj, self.private_name, None)
        if value is not None:
            return value
        if obj.T_grid is None:
            raise AttributeError(
                f"{self.public_name} defaults to zeros sized from T_grid, but "
                "T_grid has not been set yet. Assign state.T_grid first.")
        return np.zeros(len(obj.T_grid), dtype=float)

    def __set__(self, obj, value):
        setattr(obj, self.private_name,
                None if value is None else np.asarray(value, dtype=float))


class PhaseSpaceState:
    """
    Holds f(ptilde) for each species, plus the precomputed collision data.

    f is stored on the comoving momentum grid, so it is not redistributed by
    expansion. Collision terms need physical momenta, which the solver obtains
    from `self.grid.p_phys(T, self.gstar_func)` at each temperature.
    """

    # Rate grids that read as zeros until assigned
    gamma_grid_chi = _ZeroDefaultRateGrid()
    gamma_grid_A = _ZeroDefaultRateGrid()

    #: names of the attributes above, for `zeroed_rate_grids`
    _RATE_GRIDS = ("gamma_grid_chi", "gamma_grid_A")

    def __init__(self, grid: PhaseSpaceGrid, species: list[PhaseSpaceSpecies]):
        # --- set at construction ---
        self.grid = grid # comoving momentum grid, ptilde = p / a(T)
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
        # Both default to zeros (see the class body); assign gamma_grid_chi
        # from `kernels.build_xfxf_kernels` to switch elastic scattering on.
        self.gamma_grid_chi = None   # momentum exchange rate [GeV], per T
        self.gamma_grid_A = None

        # --- chi A -> chi A dark Compton, from kernels.elastic_hidden ---
        self.gain_caches_xAxA = None  # list of NT event-cache dicts

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def zeroed_rate_grids(self):
        """
        Names of the rate grids currently falling back to their zero default.
        """
        return tuple(name for name in self._RATE_GRIDS
                     if getattr(self, "_" + name, None) is None)

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

    def total_DM_number_density(self, T):
        """
        n_chi + n_chibar. The solver evolves f_chi only; with no particle
        -antiparticle asymmetry the two populations are identical, so this is
        just twice n_chi. This (NOT `number_density("chi", T)`) is what
        goes into the relic abundance.
        """
        return 2.0 * self.number_density("chi", T)

    def s_SM(self, T):
        """
        SM entropy density at T, using this state's g_* function rather than
        the module-level tabulation -- they can differ if a custom g_*(T) was
        supplied. Needed to convert a number density into a comoving yield.
        """
        return entropy_density(T, self.gstar_func(T))
