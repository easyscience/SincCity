"""Size polydispersity: the size factors of a whole structure and their shares.

A size factor f stretches every distance by f and every amplitude by f³, so a
size contributes f⁶ D(f q) to the Debye sum and f³ V to the volume.
"""

import numpy as np


def gaussian_sizes(pd: float, n: int, nsigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Size factors f and their shares, which add up to 1.

    sasmodels' ``GaussianDispersion``, as coords2sans uses it: *n* factors
    evenly from 1 - nsigma·pd to 1 + nsigma·pd, factors at or below 0 dropped,
    each with the Gaussian's height there as its share. ``pd = 0`` or ``n = 1``
    is one size: ``([1.0], [1.0])``.

    Args:
        pd: Relative standard deviation of the size factor, >= 0.
        n: Number of sizes, >= 1.
        nsigma: Half-range of the sizes in standard deviations, > 0.
    """
    if pd == 0 or n == 1:
        return np.array([1.0]), np.array([1.0])
    reach = nsigma * pd
    factors = np.linspace(1 - reach, 1 + reach, int(n))
    factors = factors[factors > 0]
    shares = np.exp(-((factors - 1) ** 2) / (2 * pd**2))
    return factors, shares / shares.sum()
