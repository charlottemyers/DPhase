# DPhase

Phase-space Boltzmann solver for dark photon (DP) dark matter.

DPhase solves the momentum-resolved Boltzmann equation for a Dirac
fermion dark matter candidate $\chi$ coupled to the Standard Model through a kinetically mixed dark photon $Z_D$. It evolves the distribution functions $f_\chi(p,T)$ and $f_{Z_D}(p,T)$ directly rather than assuming a thermal shape, so it captures departures from kinetic equilibrium.

## Features
- Full phase-space evolution of $f_\chi(p,T)$ and $f_{Z_D}(p,T)$ on
  a comoving momentum grid, with no thermal-shape assumption on either species
- Freeze-out, freeze-in, and the transitional regime are handled by the same solver, with no approximations at the boundary.
- Full $Z$--$Z_D$ mass mixing, so dark photon masses up to the EW scale are treated consistently
- Collision terms:
  - $\chi\bar{\chi} \rightarrow f\bar{f},\ W^+W^-$: annihilation to all kinematically accessible SM final states through $s$-channel $Z$ / $Z_D$
  - $\chi\bar{\chi} \leftrightarrow Z_D Z_D$: hidden sector annihilation, as a full $2\rightarrow2$ redistribution operator
  - $\chi B \rightarrow \chi B$: elastic scattering off the SM bath, as a Fokker-Planck operator for light fermions and a full $2\rightarrow2$ operator for heavy partners (top, $W^\pm$)
  - $\chi Z_D \rightarrow \chi Z_D$: dark Compton scattering, as a full $2\rightarrow2$ redistribution operator
  - $Z_D \rightarrow f\bar{f}$ decay and inverse decay, optionally with in-medium corrections to the effective mixing
- A number-density solver ([`cbe.py`](src/dphase/cbe.py)), evolving $n_\chi$, $n_{Z_D}$ and a separate dark-sector temperature, to compare the full vs moment-based results


## Installation

```bash
git clone https://github.com/charlottemyers/dphase.git
cd dphase
pip install -e .
```

For the example notebook, install the extras too:

```bash
pip install -e ".[examples]"
```

`requirements-lock.txt` records the exact versions used to run the example notebook; if you want to reproduce them precisely:

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```


## Quickstart

A run has four steps: set up the species and grid, set the couplings, precompute the collision kernels on a temperature grid, and solve.

```python
import numpy as np
import dphase
from dphase import kernels
from dphase.cosmology import compute_yields_from_snaps

# 1. Species and comoving momentum grid
m_chi, m_ZD = 10.0, 5.0                          # [GeV]
chi = dphase.PhaseSpaceSpecies(name="chi", mass_GeV=m_chi, dof=2.0)
ZD  = dphase.PhaseSpaceSpecies(name="ZD",  mass_GeV=m_ZD,  dof=3.0)

T_ref, T_final = 150.0, m_chi / 100.0
grid = dphase.PhaseSpaceGrid(ptilde_min=1e-2 * m_chi, ptilde_max=100.0 * T_ref,
                             Np=120, T_ref=T_ref)

# 2. State and couplings
state = dphase.PhaseSpaceState(grid, [chi, ZD])
state.epsilon = 1e-10                            # kinetic mixing
state.alphaD  = 1e-6                             # dark coupling strength
state.T_grid  = np.geomspace(T_ref, T_final, 120)
T_grid = state.T_grid

# 3. Precompute collision kernels
# chi chibar -> f fbar, W+ W-
state.K_grid_xxff = kernels.build_xxff_kernels(T_grid, state, m_initial=m_chi, m_mediator=m_ZD)
state.K_grid_xxWW = kernels.build_xxWW_kernels(T_grid, state, m_initial=m_chi, m_mediator=m_ZD)

# chi chibar <-> Z_D Z_D
state.K_grid_xxZDZD = kernels.build_annihilation_loss_kernels(T_grid, state, m_chi, m_ZD, "xxZDZD")
state.K_grid_ZDZDxx = kernels.build_annihilation_loss_kernels(T_grid, state, m_chi, m_ZD, "ZDZDxx")
state.gain_caches_xxZDZD = kernels.build_annihilation_gain_caches(T_grid, state, angular="anisotropic")

# chi B -> chi B: light fermions by Fokker-Planck
gamma_grid = kernels.build_xfxf_kernels(T_grid, state, exclude=("t",))
state.gamma_grid_chi = np.array([g["chi"] for g in gamma_grid])
state.FP_matrix_grid = kernels.build_fp_matrices(T_grid, state)

# 4. Solve from an empty dark sector
state.set_initial_f("chi", np.zeros_like(grid.ptilde))
state.set_initial_f("ZD",  np.zeros_like(grid.ptilde))
T_vals, snapshots = dphase.solve_BE(state, T_grid, Gamma_H_max=50)

# 5. Yields and relic abundance
Y_DM, Y_ZD, *_ = compute_yields_from_snaps(T_vals, m_chi, m_ZD, state, snapshots)
print("Omega h^2 =", dphase.relic_abundance(Y_DM[-1], m_chi))
```

`snapshots` is a list of `{species_name: f}` dictionaries, one per temperature in `T_grid`, on the comoving grid. `compute_yields_from_snaps` returns $Y_{\rm DM} = Y_\chi + Y_{\bar\chi}$ (plus the $Z_D$ and equilibrium yields), which is what `relic_abundance` expects.

See [`examples/example.ipynb`](examples/example.ipynb) for a worked example with plots. For the number-density solver, see the parameter list at the top of [`cbe.py`](src/dphase/cbe.py) and `cbe.evolve`.


## Module layout

| Module | Contents |
|---|---|
| `constants.py` | SM fermion content, EW parameters, unit conversions |
| `kinematics.py` | Kinematic helper functions|
| `grid.py` | Comoving momentum grid and species records |
| `state.py` | Parameter and kernel container; derived densities |
| `cosmology.py` | $g_*(T)$, Hubble rate, entropy, yields and relic abundance |
| `model.py` | DP couplings ($Z$--$Z_D$ mixing), matrix elements, cross sections, widths, thermal averages |
| `kernels/` | Precomputed collision kernels, one submodule per process |
| `collisions.py` | Collision operators, assembled from the kernels |
| `solver.py` | Phase-space solver (`solve_BE`) |
| `cbe.py` | Number-density (cBE) Boltzmann solver |
| `diagnostics.py` | Thermality of the evolved distributions|

`kernels/` is organized on two axes: which sector the process couples to, and whether it changes particle number.

|  | Dark Sector | SM Bath |
|---|---|---|
| **Number-changing** | `annihilation_hidden.py` ($\chi\bar{\chi} \leftrightarrow Z_D Z_D$) | `annihilation_sm.py` ($\chi\bar{\chi} \rightarrow f\bar{f},\ W^+W^-$) |
| **Number-preserving** | `elastic_hidden.py` ($\chi Z_D \rightarrow \chi Z_D$) | `elastic_sm_light.py` (light fermions, Fokker-Planck), `elastic_sm_heavy.py` (top and $W^\pm$, full operator) |

**On using a different model:** Everything except `model.py` is independent of the DP. To study a different mediator, replace that one module with different cross sections and decay widths, keeping the same function signatures and units.
