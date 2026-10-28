"""
Precomputed collision kernels and event caches.

Organized along two axes: which sector the process couples to, and whether
it changes particle number.

                    | dark sector             | SM bath
  ------------------+-----------------------------------------------
  number-changing   | chi chibar <-> Z_D Z_D  | chi chibar -> f fbar, W+ W-
  number-preserving | chi Z_D    <-> chi Z_D  | chi B -> chi B (B = f, W)

The SM elastic cell is split by partner mass: `elastic_sm_light` (Fokker-
Planck, m_B << m_chi) and `elastic_sm_heavy` (full 2 -> 2 operator).

Each submodule owns one cell, and exposes three kinds of function:

  build_*                build-time. Run once, before the solve, to tabulate
                         a kernel or event cache on a temperature grid.
  *_from_grid* / *_index run-time. Select the tabulated entry for the
                         current T.
  apply_*                run-time. Contract a cache against the current
                         distribution to give df/dt.


Cross sections come from `dphase.model`. To swap in a different mediator,
replace that module -- but note that the njit cache builders inline
`_sigma_s_xxZDZD_scalar` and `_M2_xB_xB_scalar`, so a replacement must provide
numba scalar functions with the same signatures, not plain NumPy.
"""

# --- chi chibar <-> Z_D Z_D -------------------------------------------------

from .annihilation_hidden import (
    # build-time
    build_annihilation_loss_kernels,
    build_gain_cache_xxZDZD,
    build_gain_cache_ZDZDxx,
    build_fixed_grid_gain_caches_xxZDZD,
    build_annihilation_gain_caches,
    # run-time
    apply_gain_cache,
    gain_cache_from_grid_nearest,
)

# --- chi chibar -> f fbar, W+ W- -------------------------------------------------

from .annihilation_sm import (
    # build-time
    kernel_xxff,
    build_xxff_kernels,
    kernel_xxWW,
    build_xxWW_kernels,
    # run-time
    kernel_from_grid_nearest,
)

# --- chi Z_D -> chi Z_D (dark Compton) --------------------------------------

from .elastic_hidden import (
    # build-time
    build_xZD_elastic_gain_cache,
    build_xZDxZD_gain_caches,
    # run-time
    apply_xZD_cache,
    apply_xZD_elastic_loss,
    _xZD_kernel_index,
)

# --- chi B -> chi B, light SM partners (Fokker-Planck) --------------------

from .elastic_sm_light import (
    # build-time
    gamma_single_fermion,
    gamma_single_W,
    gamma_total,
    gamma_total_W,
    build_xfxf_kernels,
    build_fp_matrices,
    # run-time
    gamma_from_grid,
    gamma_chi_from_grid,
    fp_matrices_from_grid_nearest,
)

# --- chi B -> chi B, heavy SM partners (full operator) --------------------

from .elastic_sm_heavy import (
    # build-time
    make_partner_W,
    make_partner_fermion,
    build_heavy_partner_matrix,
    build_heavy_partner_matrices,
    # run-time
    heavy_partner_matrix_from_grid_nearest,
)

__all__ = [
    # chi chibar <-> Z_D Z_D
    "build_annihilation_loss_kernels",
    "build_gain_cache_xxZDZD",
    "build_gain_cache_ZDZDxx",
    "build_fixed_grid_gain_caches_xxZDZD",
    "build_annihilation_gain_caches",
    "apply_gain_cache",
    "gain_cache_from_grid_nearest",
    # chi chibar -> f fbar
    "kernel_xxff",
    "build_xxff_kernels",
    "kernel_xxWW",
    "build_xxWW_kernels",
    "kernel_from_grid_nearest",
    # chi Z_D -> chi Z_D
    "build_xZD_elastic_gain_cache",
    "build_xZDxZD_gain_caches",
    "apply_xZD_cache",
    "apply_xZD_elastic_loss",
    "_xZD_kernel_index",
    # chi B -> chi B, light partners
    "gamma_single_fermion",
    "gamma_single_W",
    "gamma_total",
    "gamma_total_W",
    "build_xfxf_kernels",
    "build_fp_matrices",
    "gamma_from_grid",
    "gamma_chi_from_grid",
    "fp_matrices_from_grid_nearest",
    # chi B -> chi B, heavy partners
    "make_partner_W",
    "make_partner_fermion",
    "build_heavy_partner_matrix",
    "build_heavy_partner_matrices",
    "heavy_partner_matrix_from_grid_nearest",
]
