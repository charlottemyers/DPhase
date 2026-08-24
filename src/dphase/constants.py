"""
Physical constants and SM fermion content.

Units
-----
Natural units throughout: hbar = c = k_B = 1. Every mass, energy, momentum
and temperature is in GeV; cross sections are in GeV^-2.

Electroweak conventions
-----------------------
Hypercharge is normalised so that Q = T3 + Y, with no factor of 1/2. T3 = 0 for all
right-handed fields. `dphase.model.compute_couplings` and
`dphase.model._chiral_couplings_ZD` both assume this convention.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------

# Planck mass, G^{-1/2}. This is the non-reduced convention, which pairs with
# the Friedmann equation as written in `cosmology.H_of_T`:
#     H = sqrt(8 pi rho / 3) / MPL
MPL = 1.2209e19
MPL_REDUCED = MPL / np.sqrt(8.0 * np.pi)   # = 2.435e18, the reduced Planck mass

# ---------------------------------------------------------------------------
# Electroweak sector
# ---------------------------------------------------------------------------

ALPHA_EM = 1.0 / 137.0
E_EM = np.sqrt(4.0 * np.pi * ALPHA_EM)     # EM gauge coupling e

SIN2_THETA_W = 0.23122                     # sin^2(theta_W)
SIN_THETA_W = np.sqrt(SIN2_THETA_W)
COS_THETA_W = np.sqrt(1.0 - SIN2_THETA_W)
THETA_W = np.arcsin(SIN_THETA_W)

G_SU2 = E_EM / SIN_THETA_W                 # SU(2)_L gauge coupling g

MZ = 91.1876
GAMMA_Z = 2.4952                           # total Z width, for the s-channel propagator

# ---------------------------------------------------------------------------
# Numerical guards
#
# Shared by `cosmology` and `cbe`, which both work with log-space number
# densities that underflow during freeze-out.
# ---------------------------------------------------------------------------

LOG_TINY = -500.0        # ~ log(np.finfo(float).tiny); clamp before exp()
LOG_HUGE = 500.0         # ~ log(np.finfo(float).max)
VAL_FLOOR = 1e-170       # positive floor, to keep divisions and logs finite

# Fallback g_* used when the tabulated temperature dependence is switched off
# (t_dep=False). 90 is roughly the SM value above the QCD transition.
GSTAR_S_DEFAULT = 90.0

# ---------------------------------------------------------------------------
# Standard Model fermions
#
# Keys:
#   mass_GeV : current mass (MS-bar for quarks) [GeV]
#   Nc       : colour multiplicity, 1 for leptons and 3 for quarks
#   Q        : electric charge, sign-preserving
#   Q2       : Q**2, precomputed because it appears in most rates
#   T3_L     : weak isospin of the left-handed field (T3_R = 0 always)
#   Y_L, Y_R : hypercharge of the left- and right-handed fields, Q = T3 + Y
#
# MASSIVE_FERMIONS excludes neutrinos; FERMIONS adds them back as massless.
# Use MASSIVE_FERMIONS for processes that need a nonzero mass, and FERMIONS
# wherever the neutral channels contribute.
# ---------------------------------------------------------------------------

MASSIVE_FERMIONS = {
    # --- charged leptons ---
    "e": {
        "mass_GeV": 0.000511,
        "Nc": 1,
        "Q": -1.0,
        "Q2": 1.0,
        "T3_L": -0.5,
        "Y_L": -0.5,
        "Y_R": -1.0,
    },
    "mu": {
        "mass_GeV": 0.106,
        "Nc": 1,
        "Q": -1.0,
        "Q2": 1.0,
        "T3_L": -0.5,
        "Y_L": -0.5,
        "Y_R": -1.0,
    },
    "tau": {
        "mass_GeV": 1.777,
        "Nc": 1,
        "Q": -1.0,
        "Q2": 1.0,
        "T3_L": -0.5,
        "Y_L": -0.5,
        "Y_R": -1.0,
    },
    # --- up-type quarks ---
    "u": {
        "mass_GeV": 0.0022,
        "Nc": 3,
        "Q": 2.0 / 3.0,
        "Q2": 4.0 / 9.0,
        "T3_L": 0.5,
        "Y_L": 1.0 / 6.0,
        "Y_R": 2.0 / 3.0,
    },
    "c": {
        "mass_GeV": 1.28,
        "Nc": 3,
        "Q": 2.0 / 3.0,
        "Q2": 4.0 / 9.0,
        "T3_L": 0.5,
        "Y_L": 1.0 / 6.0,
        "Y_R": 2.0 / 3.0,
    },
    # --- down-type quarks ---
    "d": {
        "mass_GeV": 0.0047,
        "Nc": 3,
        "Q": -1.0 / 3.0,
        "Q2": 1.0 / 9.0,
        "T3_L": -0.5,
        "Y_L": 1.0 / 6.0,
        "Y_R": -1.0 / 3.0,
    },
    "s": {
        "mass_GeV": 0.095,
        "Nc": 3,
        "Q": -1.0 / 3.0,
        "Q2": 1.0 / 9.0,
        "T3_L": -0.5,
        "Y_L": 1.0 / 6.0,
        "Y_R": -1.0 / 3.0,
    },
    "b": {
        "mass_GeV": 4.18,
        "Nc": 3,
        "Q": -1.0 / 3.0,
        "Q2": 1.0 / 9.0,
        "T3_L": -0.5,
        "Y_L": 1.0 / 6.0,
        "Y_R": -1.0 / 3.0,
    },
}

# The top quark is omitted: mt = 173 GeV puts it out of reach for the
# mchi <~ 100 GeV benchmarks. For the same reason, we exclude gauge boson
# final states

_NEUTRINOS = {
    name: {
        "mass_GeV": 0.0,
        "Nc": 1,
        "Q": 0.0,
        "Q2": 0.0,
        "T3_L": 0.5,
        "Y_L": -0.5,
        "Y_R": 0.0,
    }
    for name in ("nue", "numu", "nutau")
}

FERMIONS = {**MASSIVE_FERMIONS, **_NEUTRINOS}

ME = MASSIVE_FERMIONS["e"]["mass_GeV"]
