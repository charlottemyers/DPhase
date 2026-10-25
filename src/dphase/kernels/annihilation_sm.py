"""
Collision kernels for annihilation into the SM bath, the number-changing
channels that connect the dark sector to the SM:

  chi chibar -> f fbar   (s-channel Z / Z_D, every SM fermion)
  chi chibar -> W+ W-    (s-channel Z / Z_D, WWZ vertex)

Only loss kernels are needed here, unlike the dark-sector channel: the SM
final states are held in equilibrium at T, so detailed balance fixes the
inverse reaction and no gain cache is required.

The two channels are tabulated separately so the WW contribution can be
switched on/off; `collisions.annihilation_collision_rhs_kernel` adds them.
"""

import numpy as np
from dphase.model import sigma_s_xxff, sigma_s_xxWW, compute_couplings
from dphase.constants import FERMIONS, MW
from dphase.kinematics import v_moller_from_s_vectorized


def _annihilation_kernel(p, m_initial, sigma_of_s, Nmu):
    """
    Shared kernel construction for chi chibar -> X annihilation:

        K_ij = Int dcos(theta)/2  sigma(s_ij) v_Moller(s_ij),

    where bins i and j hold the two incoming chi and theta is the angle
    between their momenta. vectorized: the angular integral is carried
    as a third axis and reduced by a tensordot.

    sigma_of_s : callable, sigma(s) [GeV^-2] for an (Np, Np, Nmu) array of s,
                 zero below threshold
    """
    p = np.asarray(p, dtype=float)
    E = np.sqrt(p**2 + m_initial**2)

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

    vMol = v_moller_from_s_vectorized(s, E1, E2, m_initial)

    # sigma * v_Moller is the Lorentz-invariant combination; the /2 is the
    # dcos(theta)/2 angular average, applied to the weights
    integrand = sigma_of_s(s) * vMol
    return np.tensordot(integrand, mu_w / 2.0, axes=([2], [0]))


def kernel_xxff(
    p,
    m_initial,
    m_mediator,
    epsilon,
    alphaD,
    fermions_info,
    Nmu=16
):
    """
    Build the annihilation loss kernel

        K_ij = Int dcos(theta)/2  sum_f  Nc_f sigma_f(s_ij) v_Moller(s_ij),

    where bins i and j hold the two incoming chi and theta is the angle between
    their momenta. The sum runs over every SM fermion the pair can produce,
    each with its own couplings and cross section (no threshold approximation).

    Parameters
    ----------
    p             : physical momenta [GeV], shape (Np,)
    m_initial     : mass of the annihilating pair, i.e. m_chi [GeV]
    m_mediator    : dark photon mass [GeV], entering the mixing and the propagator
    epsilon       : kinetic mixing
    alphaD        : dark coupling strength
    fermions_info : dict of SM fermion properties, normally
                    `constants.FERMIONS`. Pass `MASSIVE_FERMIONS` to drop the
                    neutrino channels.
    Nmu           : Gauss-Legendre nodes for the angular integral

    Returns
    -------
    K : shape (Np, Np), symmetric, units of <sigma v> [GeV^-2].
    """
    # couplings are s-independent, so build them once per fermion
    channels = []
    for name, fdata in fermions_info.items():
        params = compute_couplings(
            epsilon=epsilon,
            fermion=name,
            mZD=m_mediator,
            alphaD=alphaD,
            fermions_info=fermions_info,
        )
        params["mchi"] = m_initial
        channels.append((fdata["mass_GeV"], fdata["Nc"], params))

    def sigma_sum(s):
        out = np.zeros_like(s)
        for mf, Nc, params in channels:
            out += sigma_s_xxff(s, m_initial=m_initial, m_final=mf,
                                params=params, Nc=Nc)
        return out

    return _annihilation_kernel(p, m_initial, sigma_sum, Nmu)


def build_xxff_kernels(T_span, state, m_initial, m_mediator):
    """
    Tabulate the chi chibar -> f fbar kernel on a temperature grid.

    One kernel per temperature, because the comoving grid maps to different
    physical momenta at each T; see `dphase.grid`. Always sums over the full
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


def kernel_xxWW(
    p,
    m_initial,
    m_mediator,
    epsilon,
    alphaD,
    mW=MW,
    Nmu=16
):
    """
    Build the annihilation loss kernel for chi chibar -> W+ W-,

        K_ij = Int dcos(theta)/2  sigma_WW(s_ij) v_Moller(s_ij),

    on the same footing as `kernel_xxff`, with sigma_WW from
    `model.sigma_s_xxWW` (s-channel Z / Z_D, averaged over initial spins,
    summed over W polarizations).

    K is nonzero only where s > 4 mW^2, so for m_chi < mW the grid must reach
    p ~ sqrt(mW^2 - m_chi^2) for this channel to contribute at all.

    Parameters
    ----------
    p          : physical momenta [GeV], shape (Np,)
    m_initial  : mass of the annihilating pair, i.e. m_chi [GeV]
    m_mediator : dark photon mass [GeV], entering the mixing and the propagator
    epsilon    : kinetic mixing
    alphaD     : dark coupling strength
    mW         : W mass [GeV]
    Nmu        : Gauss-Legendre nodes for the angular integral

    Returns
    -------
    K : shape (Np, Np), symmetric, units of <sigma v> [GeV^-2].
    """
    # chi-W couplings are fermion-independent; "e" is only a placeholder key
    params = compute_couplings(
        epsilon=epsilon,
        fermion="e",
        mZD=m_mediator,
        alphaD=alphaD,
    )

    def sigma_WW(s):
        return sigma_s_xxWW(s, m_initial, params, mW=mW)

    return _annihilation_kernel(p, m_initial, sigma_WW, Nmu)


def build_xxWW_kernels(T_span, state, m_initial, m_mediator, mW=MW):
    """
    Tabulate the chi chibar -> W+ W- kernel on a temperature grid. Build it
    on the same grid as `build_xxff_kernels` (state.T_grid).

    Parameters
    ----------
    T_span     : temperatures [GeV]
    state      : PhaseSpaceState, supplying grid, gstar_func, epsilon,
                 alphaD and Nmu
    m_initial  : chi mass [GeV]
    m_mediator : dark photon mass [GeV]
    mW         : W mass [GeV]

    Returns
    -------
    list of (Np, Np) kernels, aligned with T_span. Assign to
    `state.K_grid_xxWW`; leave it as None to switch the channel off.
    """
    out = []
    for T in T_span:
        T = float(T)
        p_phys = state.grid.p_phys(T, gstars_func=state.gstar_func)
        K = kernel_xxWW(
            p=p_phys,
            m_initial=m_initial,
            m_mediator=m_mediator,
            epsilon=state.epsilon,
            alphaD=state.alphaD,
            mW=mW,
            Nmu=state.Nmu,
        )
        out.append(K)
    return out


def kernel_from_grid_nearest(T, T_grid, K_grid):
    """
    Pick the tabulated kernel closest to T.

    Nearest-neighbor rather than interpolation, so accuracy is
    therefore controlled by the density of the temperature grid.
    `T_grid` must be an ndarray, not a list.
    """
    T = float(T)
    i = int(np.argmin(np.abs(T_grid - T)))
    return K_grid[i]
