"""
Some kinematic helpers shared by the collision kernels.
"""

import numpy as np


def kallen(a, b, c):
    """
    Kallen triangle function,

        lambda(a, b, c) = a^2 + b^2 + c^2 - 2(ab + ac + bc).

    Called with (s, m1^2, m2^2), lambda > 0 marks the physically allowed
    region and sqrt(lambda)/(2 sqrt(s)) is the CM momentum. For equal masses it
    collapses to s(s - 4m^2).
    """
    return a*a + b*b + c*c - 2.0*(a*b + a*c + b*c)


def v_moller_from_s_vectorized(s, E1, E2, m):
    """
    Moller velocity for two particles of equal mass `m`,

        v_Moller = sqrt(lambda(s, m^2, m^2)) / (2 E1 E2).

    This is the flux factor that makes sigma * v_Moller Lorentz invariant, so
    it is the combination that belongs inside a thermal average.

    Broadcasts over array-valued s, E1 and E2. Bins below threshold
    (lambda <= 0) return exactly 0, so unphysical points drop out of a
    quadrature sum.
    """
    s = np.asarray(s, dtype=float)
    E1 = np.broadcast_to(E1, s.shape)
    E2 = np.broadcast_to(E2, s.shape)
    lam = kallen(s, m * m, m * m)

    out = np.zeros_like(s, dtype=float)
    valid = lam > 0.0
    out[valid] = np.sqrt(lam[valid]) / (2.0 * E1[valid] * E2[valid])
    return out
