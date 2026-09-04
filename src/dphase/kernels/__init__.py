"""
Precomputed collision kernels and event caches.

Organized along two axes: which sector the process couples to, and whether
it changes particle number.

                    | dark sector             | SM bath
  ------------------+-----------------------------------------------
  number-changing   | chi chibar <-> Z_D Z_D  | chi chibar -> f fbar
  number-preserving | chi Z_D    <-> chi Z_D  | chi f -> chi f

Each submodule owns one cell, and exposes three kinds of function:

  build_*                build-time. Run once, before the solve, to tabulate
                         a kernel or event cache on a temperature grid.
  *_from_grid* / *_index run-time. Select the tabulated entry for the
                         current T.
  apply_*                run-time. Contract a cache against the current
                         distribution to give df/dt.


Cross sections come from `dphase.model`. To swap in a different mediator,
replace that module -- but note that the njit cache builders inline
`_sigma_s_xxAA_scalar`, so a replacement must be a numba scalar function
with the same signature, not plain NumPy.
"""

# --- chi chibar <-> A' A' -------------------------------------------------

from .annihilation_hidden import (
    # build-time
    build_annihilation_loss_kernels,
    build_gain_cache_xxAA,
    build_gain_cache_AAxx,
    build_fixed_grid_gain_caches_xxAA,
    build_annihilation_gain_caches,
    # run-time
    apply_gain_cache,
    gain_cache_from_grid_nearest,
)

# --- chi chibar -> f fbar -------------------------------------------------

from .annihilation_sm import (
    # build-time
    kernel_xxff,
    build_xxff_kernels,
    # run-time
    kernel_from_grid_nearest,
)

# --- chi A' -> chi A' (dark Compton) --------------------------------------

from .elastic_hidden import (
    # build-time
    build_xA_elastic_gain_cache,
    build_xAxA_gain_caches,
    # run-time
    apply_xA_cache,
    apply_xA_elastic_loss,
    _xA_kernel_index,
)

# --- chi f -> chi f (Fokker-Planck coefficient) ---------------------------

from .elastic_sm import (
    # build-time
    gamma_single_fermion,
    gamma_total,
    build_xfxf_kernels,
    # run-time
    gamma_from_grid,
)


__all__ = [
    # chi chibar <-> A' A'
    "build_annihilation_loss_kernels",
    "build_gain_cache_xxAA",
    "build_gain_cache_AAxx",
    "build_fixed_grid_gain_caches_xxAA",
    "build_annihilation_gain_caches",
    "apply_gain_cache",
    "gain_cache_from_grid_nearest",
    # chi chibar -> f fbar
    "kernel_xxff",
    "build_xxff_kernels",
    "kernel_from_grid_nearest",
    # chi A' -> chi A'
    "build_xA_elastic_gain_cache",
    "build_xAxA_gain_caches",
    "apply_xA_cache",
    "apply_xA_elastic_loss",
    "_xA_kernel_index",
    # chi f -> chi f
    "gamma_single_fermion",
    "gamma_total",
    "build_xfxf_kernels",
    "gamma_from_grid",
]
