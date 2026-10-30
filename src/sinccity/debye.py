"""Debye sums for a set of point scatterers.

Every kernel computes the same quantity::

    D(q) = sum_j sum_k w_j w_k sin(q r_jk) / (q r_jk),    w_j = b_j - rho v_j

for scattering lengths ``b`` (Å), volumes ``v`` (Å³) and a solvent SLD
``rho`` (Å⁻²), at positions in Å. ``debye_pairs`` enumerates pairs and works
for any point set. The spectrum builders first collect the pairs into a
:class:`DistanceSpectrum`, a list of distances with the weights b·b, b·v and
v·v kept apart, so another solvent costs only the sinc sums:

- ``spectrum_lattice``: the autocorrelation of weight grids, exact for points
  on a lattice and fast for large grids.
- ``spectrum_histogram``: pairs counted by distance in bins of a given width,
  for any point set (coords2sans's pass).
- ``spectrum_radial``: each point's distance from a centre, the squared
  orientation-averaged amplitude ⟨A⟩², exact only for spherically symmetric
  structures (SasView's ``is_avg``).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Pairs per block. Small blocks stay in cache; larger ones were measured slower.
PAIR_BLOCK_ELEMENTS = 50_000

#: Pairs per block of the histogram pass: one matrix product of about this size.
HISTOGRAM_BLOCK_PAIRS = 1_000_000

#: Largest zero-padded grid the lattice path will allocate, in cells.
LATTICE_MAX_CELLS = 8_000_000


class CalculationInterrupted(Exception):
    """Raised when a progress callback asks a running calculation to stop."""


def _sinc_sum(q: np.ndarray, r: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return ``sum(weight * sin(q r) / (q r))`` for every q, with sinc(0) = 1.

    Pairs at r = 0 and q values of 0 contribute their full weight. The two
    buffers are reused across q, so memory does not grow with len(q).
    """
    out = np.empty(q.size)
    at_zero = r == 0
    flat = weight[at_zero].sum()
    r, weight = r[~at_zero], weight[~at_zero]
    total = weight.sum()
    qr = np.empty_like(r)
    s = np.empty_like(r)
    for i, qi in enumerate(q):
        if qi == 0:
            out[i] = flat + total
            continue
        np.multiply(r, qi, out=qr)
        np.sin(qr, out=s)
        s /= qr
        out[i] = flat + np.dot(weight, s)
    return out


@dataclass(frozen=True, eq=False)
class DistanceSpectrum:
    """The part of a Debye sum that does not depend on q or the solvent.

    Attributes:
        r: Distances, Å; only those with a non-zero component.
        components: ``(3, n)``: the summed pair weights b·b, b·v and v·v at
            each distance, so that ``D = sum (bb - 2 rho bv + rho^2 vv) sinc(q r)``.
            For a squared spectrum ``(2, n)``: b and v at each distance, so
            that ``D = (sum (b - rho v) sinc(q r))^2``.
        squared: Which of the two forms.
    """

    r: np.ndarray
    components: np.ndarray
    squared: bool = False

    def debye(self, q, rho: float) -> np.ndarray:
        """D(q) at a solvent SLD *rho* in Å⁻²."""
        q = np.asarray(q, dtype=float)
        if self.squared:
            b, v = self.components
            return _sinc_sum(q, self.r, b - rho * v) ** 2
        bb, bv, vv = self.components
        return _sinc_sum(q, self.r, bb - 2 * rho * bv + rho**2 * vv)


def _occupied(r: np.ndarray, components: np.ndarray, keep=None) -> DistanceSpectrum:
    """A pair spectrum without the distances no pair contributes to."""
    keep = np.any(components != 0, axis=0) if keep is None else keep
    return DistanceSpectrum(r[keep], components[:, keep])


def debye_pairs(
    q,
    xyz,
    w,
    *,
    block: int | None = None,
    progress: Callable[[float], bool | None] | None = None,
) -> np.ndarray:
    """Exact Debye sum by enumerating the upper triangle of pairs.

    Args:
        q: Momentum transfers, 1/Å, any values >= 0.
        xyz: Positions, shape (N, 3), Å.
        w: Weights, shape (N,).
        block: Rows per block. Defaults to a size that keeps each pair buffer
            near ``PAIR_BLOCK_ELEMENTS``.
        progress: Called with the completed fraction after each block. A
            return value of True stops the sum with ``CalculationInterrupted``.
    """
    q = np.asarray(q, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    w = np.asarray(w, dtype=float)
    n = w.size
    result = np.full(q.size, float(np.dot(w, w)))  # self terms, r = 0
    if n < 2:
        return result
    block = max(1, block or PAIR_BLOCK_ELEMENTS // n)
    total_pairs = n * (n - 1) / 2
    done = 0
    for start in range(0, n - 1, block):
        stop = min(start + block, n - 1)
        rows = np.arange(start, stop)
        cols = np.arange(start + 1, n)
        upper = cols[None, :] > rows[:, None]
        delta = xyz[rows][:, None, :] - xyz[cols][None, :, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))[upper]
        weight = (w[rows][:, None] * w[cols][None, :])[upper]
        result += 2.0 * _sinc_sum(q, r, weight)
        done += weight.size
        if progress is not None and progress(done / total_pairs):
            raise CalculationInterrupted('Debye sum stopped by the progress callback.')
    return result


def lattice_cells(shape) -> int:
    """Cells in the zero-padded grid ``spectrum_lattice`` allocates for *shape*."""
    return int(np.prod([2 * n - 1 for n in shape]))


def spectrum_lattice(grid_b, grid_v, step) -> DistanceSpectrum:
    """The exact spectrum of weights on a rectangular lattice.

    On a lattice every pair displacement is an integer vector Δ, so::

        C_bv(Δ) = sum_x b(x) v(x + Δ)          (correlation of two grids)
        D(q) = sum_Δ [C_bb - 2 rho C_bv + rho^2 C_vv](Δ) sinc(q |Δ|)

    Lags are grouped by the exact integer key (|i|, |j|, |k|), so no
    floating-point rounding enters the grouping. A lag that some pair of
    occupied sites has gets C_vv of at least the smallest volume squared, far
    above the FFT noise of the empty lags, which are dropped.

    Args:
        grid_b: Scattering lengths on the grid, shape (nx, ny, nz); empty sites hold 0.
        grid_v: Volumes on the same grid; positive at every occupied site.
        step: Lattice steps along x, y and z, Å.

    Raises:
        ValueError: If no site has a positive volume, or a site with a
            scattering length has none.
    """
    grid_b = np.asarray(grid_b, dtype=float)
    grid_v = np.asarray(grid_v, dtype=float)
    if not np.any(grid_v > 0) or np.any((grid_b != 0) & (grid_v <= 0)):
        raise ValueError('Every occupied lattice site needs a positive volume.')
    shape = grid_b.shape
    padded = [2 * n - 1 for n in shape]
    fb, fv = (np.fft.rfftn(g, s=padded, axes=(0, 1, 2)) for g in (grid_b, grid_v))

    # Index i of a padded axis of length m = 2n - 1 holds lag i for i < n and
    # lag i - m above that; only |lag| matters for the distance.
    lags = [np.minimum(np.arange(m), m - np.arange(m)) for m in padded]
    key = (
        lags[0][:, None, None] * (shape[1] * shape[2])
        + lags[1][None, :, None] * shape[2]
        + lags[2][None, None, :]
    ).ravel()
    summed = np.array(
        [
            np.bincount(
                key,
                weights=np.fft.irfftn(a * c.conj(), s=padded, axes=(0, 1, 2)).ravel(),
                minlength=int(np.prod(shape)),
            )
            for a, c in ((fb, fb), (fb, fv), (fv, fv))
        ]
    )
    ix, iy, iz = np.unravel_index(np.arange(summed.shape[1]), shape)
    step = np.asarray(step, dtype=float)
    r = np.sqrt((ix * step[0]) ** 2 + (iy * step[1]) ** 2 + (iz * step[2]) ** 2)
    return _occupied(r, summed, keep=summed[2] > 0.5 * grid_v[grid_v > 0].min() ** 2)


def spectrum_histogram(
    xyz,
    b,
    v,
    bin_width: float,
    *,
    block: int | None = None,
    progress: Callable[[float], bool | None] | None = None,
) -> DistanceSpectrum:
    """The spectrum of all pairs, counted by distance rounded to *bin_width*.

    Squared distances come from a matrix product per block of rows; each
    distance is rounded to the nearest multiple of the bin, which moves a pair
    by at most bin/2, a phase error of at most q·bin/2. Only occupied bins are
    kept.

    Args:
        xyz: Positions, shape (N, 3), Å.
        b, v: Scattering lengths and volumes, shape (N,).
        bin_width: Bin width, Å.
        block: Rows per block. Defaults to about ``HISTOGRAM_BLOCK_PAIRS`` pairs.
        progress: Called with the completed fraction after each block. A
            return value of True stops the pass with ``CalculationInterrupted``.
    """
    xyz = np.asarray(xyz, dtype=float)
    xyz = xyz - xyz.mean(axis=0)  # small coordinates keep |a|^2 + |b|^2 - 2 a.b accurate
    b = np.asarray(b, dtype=float)
    v = np.asarray(v, dtype=float)
    n = b.size
    squares = (xyz**2).sum(axis=1)
    bins = int(np.linalg.norm(np.ptp(xyz, axis=0)) / bin_width) + 2
    out = np.zeros((3, bins))
    block = max(1, block or HISTOGRAM_BLOCK_PAIRS // n)
    starts = range(0, n, block)
    total = sum((min(s + block, n) - s) * (n - s) for s in starts)
    done = 0
    for start in starts:
        stop = min(start + block, n)
        squared = (
            squares[start:stop, None] + squares[start:] - 2 * (xyz[start:stop] @ xyz[start:].T)
        )
        index = np.rint(np.sqrt(np.maximum(squared, 0)) / bin_width).astype(np.intp).ravel()
        # Pairs inside the block come in both orders; pairs with later rows
        # come once, so they count twice.
        twice = np.where(np.arange(start, n) < stop, 1.0, 2.0)
        bt, vt = b[start:] * twice, v[start:] * twice
        rows_b, rows_v = b[start:stop, None], v[start:stop, None]
        out[0] += np.bincount(index, (rows_b * bt).ravel(), bins)
        out[1] += np.bincount(index, (0.5 * (rows_b * vt + rows_v * bt)).ravel(), bins)
        out[2] += np.bincount(index, (rows_v * vt).ravel(), bins)
        done += (stop - start) * (n - start)
        if progress is not None and progress(done / total):
            raise CalculationInterrupted('Histogram pass stopped by the progress callback.')
    return _occupied(np.arange(bins) * bin_width, out)


def spectrum_radial(xyz, b, v, centre) -> DistanceSpectrum:
    """The squared spectrum ⟨A⟩²: each point's distance from *centre*, Å.

    The orientation-averaged amplitude is ``A(q) = sum_j w_j sinc(q |x_j - centre|)``,
    and ``A(q)^2`` equals D(q) only for spherically symmetric structures.
    """
    r = np.linalg.norm(np.asarray(xyz, dtype=float) - np.asarray(centre, dtype=float), axis=1)
    return DistanceSpectrum(r, np.array([b, v], dtype=float), squared=True)
