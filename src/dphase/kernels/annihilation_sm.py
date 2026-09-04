"""
Collision kernel for chi chibar -> f fbar, the number-changing annihilation channel
that connects the dark sector to the SM bath.

Only a loss kernel is needed here, unlike the dark-sector channel: the SM fermions are held
in equilibrium at T, so detailed balance fixes the inverse reaction and no gain cache is required.

Every kinematically accessible SM fermion contributes, summed with its color
multiplicity, and each one gets its own Z_D-Z mixed couplings from
`model.compute_couplings`.
"""

import numpy as np
from dphase.model import  sigma_s_xxff, compute_couplings
from dphase.constants import FERMIONS
from dphase.kinematics import v_moller_from_s_vectorized


def kernel_xxff(
    p,
    m_initial,
    m_mediator,
    epsilon,
    alphaD,
    fermions_info,
    Nmu=16,
    const_xsec=None,
):
    """
    Build the annihilation loss kernel

        K_ij = Int dcos(theta)/2  sum_f  Nc_f sigma_f(s_ij) v_Moller(s_ij),

    where bins i and j hold the two incoming chi and theta is the angle between
    their momenta. The sum runs over every SM fermion the pair can produce,
    each with its own A'-Z mixed couplings and full s-dependent cross section
    (no threshold approximation).

    Fully vectorized: the angular integral is carried as a third axis and
    reduced by a single tensordot at the end. Note the intermediate arrays are
    (Np, Np, Nmu), so memory grows quadratically in the grid size.

    Parameters
    ----------
    p             : physical momenta [GeV], shape (Np,)
    m_initial     : mass of the annihilating pair, i.e. m_chi [GeV]
    m_mediator    : dark photon mass [GeV], entering the mixing and the
                    propagator
    epsilon       : kinetic mixing
    alphaD        : dark fine structure constant
    fermions_info : dict of SM fermion properties, normally
                    `constants.FERMIONS`. Pass `MASSIVE_FERMIONS` to drop the
                    neutrino channels.
    Nmu           : Gauss-Legendre nodes for the angular integral
    const_xsec    : if given, bypass the physics and return a constant kernel.
                    Useful for testing the solver against an analytic case;
                    nothing in the package passes it.

    Returns
    -------
    K : shape (Np, Np), symmetric, units of <sigma v> [GeV^-2].
    """
    p = np.asarray(p, dtype=float)
    E = np.sqrt(p**2 + m_initial**2)
    Np = p.size

    if const_xsec is not None:
        return const_xsec * np.ones((Np, Np), dtype=float)

    mu_nodes, mu_w = np.polynomial.legendre.leggauss(Nmu)

    # Shapes:
    # p1, E1: (Np, 1, 1)
    # p2, E2: (1, Np, 1)
    # mu:     (1, 1, Nmu)
    p1 = p[:, None, None]
    p2 = p[None, :, None]
    E1 = E[:, None, None]
    E2 = E[None, :, None]
    mu = mu_nodes[None, None, :]
    # CM energy of two equal-mass particles at relative angle mu.
    s = 2.0 * m_initial**2 + 2.0 * (E1 * E2 - p1 * p2 * mu)

    sigma_sum = np.zeros_like(s)

    for name, fdata in fermions_info.items():
        mf = fdata["mass_GeV"]
        Nc = fdata["Nc"]

        couplings = compute_couplings(
            epsilon=epsilon,
            fermion=name,
            mA=m_mediator,
            fermions_info=fermions_info,
        )

        params = {
            "couplings": {
                "C_A": couplings["C_A"],
                "C_V": couplings["C_V"],
                "C_A_prime": couplings["C_A_prime"],
                "C_V_prime": couplings["C_V_prime"],
                "m_ZD": couplings["m_ZD"],
                "m_Z": couplings["m_Z"],
                "alpha": couplings["alpha"],
            },
            "mchi": m_initial,
            "mf": mf,
            "alphaD": alphaD,
            "epsilon": epsilon,
        }

        sigma_f = sigma_s_xxff(
            s,
            m_initial=m_initial,
            m_final=mf,
            Msq_const=None,
            const_xsec=None,
            params=params,
        )

        # Colour multiplicity applied here, not inside matrix element.
        sigma_sum += Nc * sigma_f

    vMol = v_moller_from_s_vectorized(s, E1, E2, m_initial)

    # sigma * v_Moller is the Lorentz-invariant combination; the /2 is the
    # dcos(theta)/2 angular average, applied to the weights
    integrand = sigma_sum * vMol
    return np.tensordot(integrand, mu_w / 2.0, axes=([2], [0]))



def build_xxff_kernels(T_span, state, m_initial, m_mediator):
    """
    Tabulate the chi chibar -> f fbar kernel on a temperature grid.

    One kernel per temperature, because the comoving grid maps to different
    physical momenta at each T -- see `dphase.grid`. Always sums over the full
    `constants.FERMIONS` set.

    Parameters
    ----------
    T_span     : temperatures [GeV]
    state      : PhaseSpaceState, supplying grid, gstar_func, epsilon,
                 alphaD and Nmu
    m_initial  : chi mass [GeV]
    m_mediator : dark photon mass [GeV]

    Returns
    -------
    list of (Np, Np) kernels, aligned with T_span. Assign to
    `state.K_grid_xxff`.
    """
    out = []
    for T in T_span:
        T = float(T)
        p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
        K = kernel_xxff(
            p=p_phys,
            m_initial=m_initial,
            m_mediator=m_mediator,
            epsilon=state.epsilon,
            alphaD=state.alphaD,
            fermions_info=FERMIONS,
            Nmu=state.Nmu,
        )
        out.append(K)
    return out


def kernel_from_grid_nearest(T, T_grid, K_grid):
    """
    Pick the tabulated kernel closest to T.

    Nearest-neighbour rather than interpolation: the kernels are (Np, Np)
    matrices, and blending two of them every solver step would cost more than
    simply tabulating T_span more finely. Accuracy is therefore controlled by
    the density of the build grid, and the solver's temperature grid should be
    no finer than it.

    `T_grid` must be an ndarray, not a list: the subtraction below is
    elementwise.
    """
    T = float(T)
    i = int(np.argmin(np.abs(T_grid - T)))
    return K_grid[i]
