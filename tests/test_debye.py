"""The Debye kernels, and the physics the calculator reproduces with them."""

import numpy as np
import pytest
from support import C60, Q, brute, close, quiet_calculator, sphere_pixels

from sinccity import CalculationInterrupted, Structure
from sinccity.debye import (
    debye_lattice,
    debye_pairs,
    spectrum_histogram,
    spectrum_lattice,
    spectrum_radial,
)


class TestKernels:
    def test_two_points_and_q_zero(self):
        xyz = np.array([[0.0, 0, 0], [0, 0, 2.5]])
        w = np.array([2.0, -0.5])
        expected = (w**2).sum() + 2 * w[0] * w[1] * np.sinc(Q * 2.5 / np.pi)
        assert np.allclose(debye_pairs(Q, xyz, w), expected, rtol=0, atol=1e-14)
        assert debye_pairs([0.0], xyz, w)[0] == pytest.approx(w.sum() ** 2)

    def test_matches_brute_force_with_coincident_points_and_any_block(self):
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(120, 3)) * 10
        xyz[5] = xyz[7]
        w = rng.normal(size=120)
        reference = brute(Q, xyz, w)
        for block in (1, 7, 120, 1000, None):
            assert close(debye_pairs(Q, xyz, w, block=block), reference, w)

    def test_translation_and_rotation_invariance(self):
        rng = np.random.default_rng(1)
        xyz, w = rng.normal(size=(50, 3)) * 8, rng.normal(size=50)
        theta = 0.7
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        moved = xyz @ rotation.T + [10, -3, 4]
        assert close(debye_pairs(Q, xyz, w), debye_pairs(Q, moved, w), w)

    def test_lattice_path_equals_pair_sum(self):
        rng = np.random.default_rng(2)
        grid = rng.normal(size=(8, 6, 5)) * (rng.random((8, 6, 5)) < 0.6)
        step = (3.0, 2.0, 4.5)
        index = np.argwhere(grid != 0)
        w = grid[grid != 0]
        lattice = debye_lattice(Q, grid, step)
        assert close(lattice, debye_pairs(Q, index * np.array(step), w), w)
        assert np.all(lattice >= -1e-12 * np.abs(w).sum() ** 2)  # D is an average of |A|^2

    def test_progress_can_stop_the_pair_loop(self):
        xyz = np.random.default_rng(3).normal(size=(50, 3))
        with pytest.raises(CalculationInterrupted):
            debye_pairs(Q, xyz, np.ones(50), block=5, progress=lambda f: f > 0.3)


class TestSpectra:
    def test_histogram_against_the_closed_form_of_a_line(self):
        """n equal atoms d apart: D = n + 2 sum_k (n - k) sinc(q k d)."""
        q = np.geomspace(1e-3, 0.3, 20)
        for n in (3, 1500):  # 1500 atoms cross several blocks
            d = 1.2345
            xyz = np.zeros((n, 3))
            xyz[:, 0] = np.arange(n) * d
            k = np.arange(1, n)
            exact = n + 2 * np.array([((n - k) * np.sinc(qi * k * d / np.pi)).sum() for qi in q])
            spectrum = spectrum_histogram(xyz, np.ones(n), np.zeros(n), 1e-3 / q[-1])
            assert np.allclose(spectrum.debye(q, 0.0), exact, rtol=1e-4, atol=0)

    def test_histogram_against_the_pair_sum_and_any_block(self):
        rng = np.random.default_rng(5)
        xyz = rng.normal(size=(300, 3)) * 15
        b, v = rng.random(300) + 0.5, rng.random(300) * 10
        rho = 0.02
        exact = debye_pairs(Q[1:], xyz, b - rho * v)
        whole = spectrum_histogram(xyz, b, v, 1e-3 / Q[-1])
        assert np.allclose(whole.debye(Q[1:], rho), exact, rtol=1e-4)
        blocked = spectrum_histogram(xyz, b, v, 1e-3 / Q[-1], block=7)
        assert np.allclose(blocked.components, whole.components, rtol=1e-12)

    def test_three_components_equal_the_single_weight_spectrum(self):
        rng = np.random.default_rng(6)
        xyz = rng.normal(size=(80, 3)) * 10
        b, v = rng.normal(size=80), rng.random(80) + 0.1
        three = spectrum_histogram(xyz, b, v, 0.01)
        for rho in (-0.5, 0.0, 0.7):
            single = spectrum_histogram(xyz, b - rho * v, np.zeros(80), 0.01)
            assert close(three.debye(Q, rho), single.debye(Q, 0.0), b - rho * v)

    def test_only_occupied_distances_are_kept(self):
        structure = quiet_calculator(C60).structure
        n = structure.n_points
        spectrum = spectrum_histogram(structure.xyz, structure.sld, structure.volume, 0.0033)
        assert spectrum.r.size < n * (n - 1) // 2  # C60 has few distinct distances
        assert np.all(np.any(spectrum.components != 0, axis=0))

    def test_lattice_spectrum_against_debye_lattice(self):
        rng = np.random.default_rng(7)
        occupied = rng.random((6, 5, 4)) < 0.5
        grid_b = rng.normal(size=occupied.shape) * occupied
        grid_v = rng.random(occupied.shape) * occupied
        spectrum = spectrum_lattice(grid_b, grid_v, (2.0, 3.0, 1.5))
        rho = 0.3
        reference = debye_lattice(Q, grid_b - rho * grid_v, (2.0, 3.0, 1.5))
        assert close(spectrum.debye(Q, rho), reference, (grid_b - rho * grid_v)[occupied])
        assert spectrum.r.size < 6 * 5 * 4  # lags no pair has are dropped

    def test_radial_spectrum_is_the_squared_amplitude_about_the_centre(self):
        rng = np.random.default_rng(8)
        xyz = rng.normal(size=(40, 3)) * 5
        b, v = rng.normal(size=40), rng.random(40)
        centre = (xyz.min(0) + xyz.max(0)) / 2
        spectrum = spectrum_radial(xyz, b, v, centre)
        r = np.linalg.norm(xyz - centre, axis=1)
        amplitude = np.array([((b - 0.2 * v) * np.sinc(qi * r / np.pi)).sum() for qi in Q])
        assert np.allclose(spectrum.debye(Q, 0.2), amplitude**2)
        assert spectrum.debye([0.0], 0.2)[0] == pytest.approx((b - 0.2 * v).sum() ** 2)


class TestPhysics:
    def test_sasview_cuboid_regression_values(self):
        """10x10x10 lattice, steps 6/3/6 Ang, SLD 0.1 1/Ang^2, as in SasView's 1D test."""
        index = np.stack(np.meshgrid(*[np.arange(10)] * 3, indexing='ij'), -1).reshape(-1, 3)
        xyz = index * np.array([6.0, 3.0, 6.0])
        structure = Structure.from_arrays(xyz, np.full(1000, 0.1), np.full(1000, 108.0))
        gsc = quiet_calculator(structure)
        gsc.set_q([0.01, 0.03])
        profile = gsc.calculate()
        assert profile.method == 'lattice'
        expected = 1e8 * brute([0.01, 0.03], xyz, np.full(1000, 10.8)) / 108000.0
        assert np.allclose(profile.intensity, expected, rtol=1e-10)
        assert np.allclose(profile.intensity, [1.05619e11, 8.82750e10], rtol=1e-5)

    def test_sphere_of_pixels_matches_the_analytic_sphere_at_scale_one(self):
        """At scale 1 the pixel sphere is the sphere form factor: V * contrast^2 * F(qR)^2."""
        gsc = quiet_calculator(sphere_pixels())
        gsc.set_params(sld_solvent=6.34)
        gsc.set_q(np.geomspace(1e-3, 0.1, 40))
        profile = gsc.calculate()
        radius, contrast = 30.0, (1.0 - 6.34) * 1e-6
        x = gsc.q * radius
        form = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        volume = 4 / 3 * np.pi * radius**3
        analytic = 1e8 * contrast**2 * volume * form**2
        low = x <= 3
        assert np.allclose(profile.intensity[low], analytic[low], rtol=0.02)

    def test_average_kernel_matches_the_analytic_sphere(self):
        gsc = quiet_calculator(sphere_pixels())
        gsc.set_params(sld_solvent=6.34, kernel='average')
        gsc.set_q(np.geomspace(1e-3, 0.1, 40))
        profile = gsc.calculate()
        assert profile.method == 'average' and '<A>^2 kernel' in profile.notes[-1]
        radius, contrast = 30.0, (1.0 - 6.34) * 1e-6
        x = gsc.q * radius
        form = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        analytic = 1e8 * contrast**2 * (4 / 3 * np.pi * radius**3) * form**2
        low = x <= 3
        assert np.allclose(profile.intensity[low], analytic[low], rtol=0.02)

    @pytest.mark.slow
    def test_pair_loop_agrees_with_lattice_path_on_the_sphere(self):
        structure = sphere_pixels()
        w = (structure.sld - 6.34e-6) * structure.volume
        active = w != 0
        index = structure.lattice.indices[active]
        grid = np.zeros(tuple(index.max(0) + 1))
        np.add.at(grid, tuple(index.T), w[active])
        q = np.geomspace(1e-3, 0.3, 10)
        pairs = debye_pairs(q, structure.xyz[active], w[active])
        assert close(debye_lattice(q, grid, structure.lattice.step), pairs, w)
