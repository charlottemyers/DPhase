# DPhase

Phase-space Boltzmann solver for dark photon (DP) dark matter.

DPhase solves the momentum-resolved Boltzmann equation for a Dirac
fermion dark matter candidate χ coupled to the Standard Model through a
kinetically mixed dark photon A′. It evolves the distribution function
f<sub>χ</sub>(p, T) directly rather than assuming a thermal shape and tracking
only the number density, so it captures departures from kinetic equilibrium
during and after freeze-out.

## Scope

**What it does**

- Full phase-space evolution of f<sub>χ</sub>(p, T) and f<sub>A′</sub>(p, T) on
  a log-spaced comoving momentum grid, with no thermal-shape assumption on
  either species
- Freeze-out and freeze-in regimes
- Collision terms:
  - χχ̄ → f f̄ annihilation to all kinematically accessible SM fermions, with
    full A′–Z mixing and s-dependent cross sections
  - χχ̄ ↔ A′A′, both directions, with an explicit gain (redistribution) term
    rather than a number-conserving approximation
  - χ f → χ f elastic scattering off the SM bath, as a semi-relativistic
    Fokker–Planck operator
  - χ A′ → χ A′ "dark Compton" elastic scattering, as a full 2→2
    redistribution operator
  - A′ → f f̄ decay and inverse decay
- Hidden-sector temperature evolved as a dynamical variable, not assumed equal
  to the SM temperature
- A number-density solver ([`cbe.py`](src/dphase/cbe.py)) alongside the
  phase-space one, sharing the same cross sections — so the two can be compared
  on identical physics inputs


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

The phase-space solver needs its collision kernels tabulated on a temperature
grid before it can run. That precomputation is the expensive part and is
currently driven explicitly:

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
state.gstar_func = dphase.gstar_interp
state.epsilon    = 1e-10       # kinetic mixing
state.alphaD     = 1e-6        # dark fine structure constant
state.T_grid     = np.geomspace(T_ref, T_final, 150)

# 3. Precompute collision kernels on state.T_grid (slow)
#    ... see examples/example.ipynb

# 4. Solve
T_grid, snapshots = dphase.solve_BE(state, state.T_grid)
```

`snapshots` is a list of `{species_name: f}` dictionaries, one per temperature
step, on the comoving grid. Convert to an abundance with
`state.total_DM_number_density(T)`, `dphase.Y`, and
`dphase.relic_abundance`.

See [`examples/example.ipynb`](examples/example.ipynb) for a worked example
with plots.


## Module layout

| Module | Contents |
|---|---|
| `constants.py` | SM fermion content, electroweak parameters, numerical guards |
| `kinematics.py` | Källen function, Møller velocity |
| `grid.py` | Comoving momentum grid, species records, log bin edges |
| `state.py` | Parameter and kernel container; derived densities |
| `model.py` | **Dark photon cross sections, widths, and couplings** |
| `kernels/` | Precomputed collision kernels, one submodule per process class |
| `collisions.py` | Collision operators, assembled from the kernels |
| `solver.py` | Phase-space ODE driver (`solve_BE`) |
| `cosmology.py` | g<sub>*</sub>(T), Hubble rate, entropy, t(T) |
| `cbe.py` | Number-density solver — the validation baseline |

`kernels/` is organized on two axes: which sector the process couples to, and
whether it changes particle number.

|  | dark sector | SM bath |
|---|---|---|
| **number-changing** | `annihilation_hidden.py` (χχ̄ ↔ A′A′) | `annihilation_sm.py` (χχ̄ → f f̄) |
| **number-preserving** | `elastic_hidden.py` (χA′ → χA′) | `elastic_sm.py` (χf → χf) |

**On using a different model.** Everything except `model.py` is independent of
the dark photon. To study a different mediator, replace that one module with
your own cross sections and decay widths, keeping the same function signatures
and units.

<!-- ## Conventions

- Natural units, ħ = c = k<sub>B</sub> = 1
- All masses, energies, momenta and temperatures in GeV; cross sections in -->
  <!-- GeV<sup>-2</sup>
- Hypercharge normalised as Q = T<sup>3</sup> + Y, with **no** factor of ½
  (some references use Q = T<sup>3</sup> + Y/2 and doubled Y)
- T<sup>3</sup> = 0 for all right-handed fields
- `dof` counts internal degrees of freedom of one species: 2 for χ (spin
  states only — χ̄ is a separate population, added by
  `total_DM_number_density`), 3 for the A′ polarisations
- |M|² for annihilation is **averaged** over initial spins and **summed** over
  final spins; colour multiplicity N<sub>c</sub> is applied by the caller, not
  folded into the matrix element
- Cross sections use the per-dof-averaged convention, so the g<sub>χ</sub>
  factors are applied at the rate level in `collisions.py`, not inside `model.py`
- Comoving momentum p̃ = a(t) p, with a(T<sub>ref</sub>) = 1
- Non-reduced Planck mass, pairing with H = √(8πρ/3)/M<sub>Pl</sub> -->


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
