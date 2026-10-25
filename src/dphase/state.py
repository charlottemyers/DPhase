"""
Container for the evolving distribution functions.

`PhaseSpaceState` is a mutable class, populated in two stages:

  1. construction   -- grid and species list
  2. precomputation -- couplings, the temperature grid, and the collision
                       kernels/caches tabulated on it

Stage 2 is done by hand (see `example.ipynb`)
and every attribute it sets is declared in `__init__` below.
"""

import numpy as np

import dphase
from dphase.cosmology import entropy_density
from dphase.grid import PhaseSpaceGrid, PhaseSpaceSpecies


class _ZeroDefaultRateGrid:
    """
    A per-temperature rate grid that reads as zeros until it is assigned.

    Some rate contributions are identically zero for the DP model, i.e.
    Z_D f -> Z_D f elastic scattering, whose rate scales as epsilon^4`.
    Rather than make the caller build and assign an array of zeros, those
    attributes default to zeros here, so only the non-trivial grids need assigning.
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
    gamma_grid_ZD = _ZeroDefaultRateGrid()

    #: names of the attributes above, for `zeroed_rate_grids`
    _RATE_GRIDS = ("gamma_grid_chi", "gamma_grid_ZD")

    def __init__(self, grid: PhaseSpaceGrid, species: list[PhaseSpaceSpecies]):
        # --- set at construction ---
        self.grid = grid # comoving momentum grid, ptilde = p * a(T)
        self.species = {sp.name: sp for sp in species}
        self.f = {sp.name: np.zeros_like(grid.ptilde) for sp in species}

        # --- model parameters  ---
        self.epsilon = None
        self.alphaD = None
        self.gstar_func = dphase.cosmology.gstar_interp  # g_*(T) function for SM plasma
        self.Nmu = 8                                     # GL nodes for angular integrals

        # --- temperature grid the kernels below are tabulated on ---
        self.T_grid = None       # shape (NT,)

        # --- chi chibar -> f fbar, W+ W-, from kernels.annihilation_sm ---
        self.K_grid_xxff = None  # (NT, Np, Np)
        self.K_grid_xxWW = None  # (NT, Np, Np); None switches the WW channel off

        # --- chi chibar <-> Z_D Z_D, from kernels.annihilation_hidden ---
        self.K_grid_xxZDZD = None      # (NT, Np, Np) loss kernel, chi side
        self.K_grid_ZDZDxx = None      # (NT, Np, Np) loss kernel, Z_D side
        self.gain_caches_xxZDZD = None  # list of NT gain-cache dicts

        # --- chi B -> chi B off light SM partners, from kernels.elastic_sm_light ---
        # Both default to zeros; assign gamma_grid_chi to switch elastic scattering on.
        self.gamma_grid_chi = None   # momentum exchange rate [GeV], per T
        self.gamma_grid_ZD = None
        self.gamma_fp_excluded = ()  # fermions left out of gamma_grid_chi
        # FP matrices, from kernels.build_fp_matrices
        self.FP_matrix_grid = None   # {"chi": (NT, Np, Np), "ZD": (NT, Np, Np)}

        # --- chi B -> chi B off heavy SM partners, from kernels.elastic_sm_heavy ---
        self.C_grid_heavy = None     # {partner name: (NT, Np, Np)}

        # --- Z_D -> f fbar coefficients, most recent T only ---
        self._decay_affine_cache = None

        # --- chi Z_D -> chi Z_D dark Compton, from kernels.elastic_hidden ---
        self.gain_caches_xZDxZD = None  # list of NT event-cache dicts

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
        just 2*n_chi. This is what goes into the relic abundance.
        """
        return 2.0 * self.number_density("chi", T)

    def s_SM(self, T):
        """
        SM entropy density at T, using this state's g_* function.
        Needed to convert a number density into a comoving yield.
        """
        return entropy_density(T, self.gstar_func(T))
