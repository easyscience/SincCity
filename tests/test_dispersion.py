"""Size factors and shares."""

import numpy as np
import pytest
from support import quiet_calculator

from sinccity.dispersion import gaussian_sizes


def test_one_size():
    for pd, n in ((0.0, 35), (0.2, 1)):
        factors, shares = gaussian_sizes(pd, n, 3.0)
        assert factors.tolist() == [1.0] and shares.tolist() == [1.0]


def test_gaussian_over_plus_minus_nsigma():
    factors, shares = gaussian_sizes(0.1, 35, 3.0)
    assert factors.size == 35 and factors[0] == pytest.approx(0.7) and factors[-1] == 1.3
    assert shares.sum() == pytest.approx(1.0) and shares.argmax() == 17  # the mean, n odd
    assert np.allclose(shares, shares[::-1])
    assert shares[0] / shares[17] == pytest.approx(np.exp(-4.5))  # 3 sigma out


def test_sizes_at_or_below_zero_are_dropped():
    factors, shares = gaussian_sizes(0.5, 7, 3.0)  # from -0.5 to 2.5 in steps of 0.5
    assert factors.tolist() == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert shares.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    'params',
    [{'size_pd': -0.1}, {'size_pd_n': 0}, {'size_pd_n': 2.5}, {'size_pd_nsigma': 0.0}],
)
def test_refusals(params):
    with pytest.raises(ValueError, match=next(iter(params))):
        quiet_calculator().set_params(**params)
