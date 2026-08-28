# DPhase

Phase-space Boltzmann solver for dark photon (DP) dark matter.

DPhase solves the momentum-resolved Boltzmann equation for a Dirac
fermion dark matter candidate $\chi$ coupled to the Standard Model through a kinetically mixed dark photon $Z_D$. It evolves the distribution functions $f_\chi(p,T)$ and $f_{Z_D}(p,T)$ directly rather than assuming a thermal shape, so it captures departures from kinetic equilibrium.


## Features

- Full phase-space evolution of $f_\chi(p,T)$ and $f_{Z_D}(p,T)$ on
  a comoving momentum grid, with no thermal-shape assumption on either species
- Freeze-out, freeze-in, and the transitional regime are handled by the same solver — no switching between codes or approximations at the boundary.
- Collision terms:
  - $\chi\bar{\chi} \rightarrow f\bar{f}$: annihilation to all kinematically accessible SM fermions, with full $Z_D$--$Z$ mixing
  - $\chi\bar{\chi} \leftrightarrow Z_D Z_D$: hidden sector annihilation, as a full $2\rightarrow2$ redistribution operator
  - $\chi f \rightarrow \chi f$: elastic scattering off the SM bath, as a semi-relativistic Fokker--Planck operator
  - $\chi Z_D \rightarrow \chi Z_D$ dark Compton elastic scattering,
    as a full $2\rightarrow2$ redistribution operator
  - $Z_D \rightarrow f\bar{f}$ decay and inverse decay
- A number-density solver ([`cbe.py`](src/dphase/cbe.py)) alongside the
  phase-space one to compare the full vs moment-based results


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

Requires Python 3.10+, NumPy 2.0+, SciPy, numba 0.61+, and mpmath.


## Quickstart

*Notation note*: the dark photon is $Z_D$ throughout this document and `A` in the source, e.g. `species["A"]`, `K_grid_xxAA`.

<!-- The phase-space solver needs its collision kernels tabulated on a temperature grid before it can run. -->

```python
import numpy as np
import dphase
from dphase import kernels

# 1. Species and grid
chi = dphase.PhaseSpaceSpecies(name="chi", mass_GeV=10.0, dof=2.0)
A   = dphase.PhaseSpaceSpecies(name="A",   mass_GeV=5.0,  dof=3.0)

T_ref, T_final = 10.0 / 0.05, 10.0 / 600.0        # x = m_chi/T from 0.05 to 600
grid = dphase.PhaseSpaceGrid(ptilde_min=0.1, ptilde_max=50.0 * T_ref,
                             Np=120, T_ref=T_ref)

# 2. State and model parameters
state = dphase.PhaseSpaceState(grid, [chi, A])
state.epsilon    = 1e-10       # kinetic mixing
state.alphaD     = 1e-6        # dark fine structure constant
state.T_grid     = np.geomspace(T_ref, T_final, 150)

# 3. Precompute collision kernels on state.T_grid (slow)
#    ... see examples/example.ipynb

# 4. Solve
T_grid, snapshots = dphase.solve_BE(state, state.T_grid)
```

`snapshots` is a list of `{species_name: f}` dictionaries, one per temperature step, on the comoving grid. Convert to an abundance with
`state.total_DM_number_density(T)`, `dphase.Y`, and
`dphase.relic_abundance`.

See [`examples/example.ipynb`](examples/example.ipynb) for a worked example.


## Module layout

| Module | Contents |
|---|---|
| `constants.py` | SM fermion content, EW parameters, unit conversions |
| `kinematics.py` | Källén function, Møller velocity |
| `grid.py` | Comoving momentum grid, species records, grid setup |
| `state.py` | Parameter and kernel container; derived densities |
| `model.py` | DP cross sections, widths, and couplings |
| `kernels/` | Precomputed collision kernels, one submodule per process |
| `collisions.py` | Collision operators, assembled from the kernels |
| `solver.py` | Phase-space ODE solver (`solve_BE`) |
| `cosmology.py` | $g_*(T)$, Hubble rate, entropy |
| `cbe.py` | Number-density Boltzmann solver  |

`kernels/` is organized on two axes: which sector the process couples to, and whether it changes particle number.

|  | Dark Sector | SM Bath |
|---|---|---|
| **Number-changing** | `annihilation_hidden.py` ($\chi\bar{\chi} \leftrightarrow Z_D Z_D$) | `annihilation_sm.py` ($\chi\bar{\chi} \rightarrow f\bar{f}$) |
| **Number-preserving** | `elastic_hidden.py` ($\chi Z_D \rightarrow \chi Z_D$) | `elastic_sm.py` ($\chi f \rightarrow \chi f$) |

**On using a different model:** Everything except `model.py` is independent of the DP. To study a different mediator, replace that one module with different cross sections and decay widths, keeping the same function signatures and units.


<!-- ## Physics and references

Formulas are taken from the literature as follows. Equation numbers refer to
the cited papers.

| Quantity | Implementation | Source |
|---|---|---|
| σ(s) for χχ̄ → A′A′ | `model.sigma_s_xxAA` | Coy, Kimus, Tytgat, [arXiv:2405.10792](https://arxiv.org/abs/2405.10792), Eq. (B.1) |
| A′–Z mass and coupling mixing | `model.compute_couplings` | TODO — add reference for Eq. (A.2) |
| Γ(A′ → f f̄), \|M\|² with mixing | `model.A_ff_decay_m2` | Curtin et al., Eq. (2.12), inverted — TODO add arXiv number |
| γ(T) momentum exchange rate | `kernels.elastic_sm.gamma_single_fermion` | Binder et al., [arXiv:1706.07433](https://arxiv.org/abs/1706.07433), Eqs. (6)–(7) |
| Fokker–Planck elastic operator | `collisions.fokker_planck_dfdt` | Binder et al., [arXiv:1706.07433](https://arxiv.org/abs/1706.07433), Eq. (8) |
| Hidden-sector ρ, P, dT<sub>h</sub>/dt | `cbe.rho_i_exact`, `cbe.dTh_dt_rel` | [arXiv:2504.00077](https://arxiv.org/abs/2504.00077), Eq. (2.15) |
| \|M\|² for χf → χf, t-channel A′ | `model.M2_chi_f_avg` | Standard trace calculation |
| \|M\|² for χA′ → χA′ | `model.xZD_xZD_m2` | Standard trace calculation | -->
