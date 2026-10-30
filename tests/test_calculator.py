"""The calculator: normalisation, warnings, state, output and plots."""

import copy
import json
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest
from support import C60, quiet_calculator, sphere_pixels

import sinccity
from sinccity import CalculationInterrupted, GenericScatteringCalculator, Structure, StructureError
from sinccity.debye import debye_pairs
from sinccity.elements import constants

# ============================================================================
# Normalisation and warnings
# ============================================================================


class TestNormalisation:
    def setup_method(self):
        self.xyz = np.array([[0.0, 0, 0], [0, 0, 5], [0, 5, 0], [5, 0, 0]])
        self.structure = Structure.from_arrays(
            self.xyz, np.array([2, 2, 1, 1]) * 1e-6, np.array([1.0, 2.0, 3.0, 4.0])
        )

    def test_active_volume_excludes_matched_points(self):
        calc = quiet_calculator(self.structure, qmax=0.2)
        calc.set_params(sld_solvent=1.0)
        profile = calc.calculate()
        assert profile.normalisation_volume == pytest.approx(3.0)
        assert profile.n_zero_contrast == 2
        w = np.array([1e-6, 2e-6])
        expected = 1e8 * debye_pairs(calc.q, self.xyz[:2], w) / 3.0
        assert np.allclose(profile.intensity, expected, rtol=1e-12)

    def test_explicit_volume_and_auto_restore(self):
        calc = quiet_calculator(self.structure, qmax=0.2)
        auto = calc.calculate()
        calc.set_params(normalisation_volume=20.0)
        explicit = calc.calculate()
        assert explicit.normalisation == 'explicit'
        assert np.allclose(explicit.intensity * 20.0, auto.intensity * 10.0)
        calc.set_params(normalisation_volume='auto')
        assert calc.calculate().normalisation_volume == pytest.approx(10.0)

    def test_scale_and_background(self):
        calc = quiet_calculator(self.structure, qmax=0.2)
        base = calc.calculate().intensity
        calc.set_params(scale=2.0, background=0.5)
        assert np.allclose(calc.calculate().intensity, 2 * base + 0.5)

    def test_all_zero_contrast_returns_the_background(self):
        structure = Structure.from_arrays(self.xyz, np.full(4, 2e-6), np.ones(4))
        calc = quiet_calculator(structure, qmax=0.2)
        calc.set_params(sld_solvent=2.0, background=0.1)
        with pytest.warns(UserWarning, match='Every point matches the solvent'):
            profile = calc.calculate()
        assert np.all(profile.intensity == 0.1)
        assert np.isnan(profile.normalisation_volume)
        calc.set_params(normalisation_volume=7.0)
        with pytest.warns(UserWarning):
            assert calc.calculate().normalisation_volume == 7.0

    def test_near_match_warns(self):
        structure = Structure.from_arrays(
            self.xyz, np.array([1e-6, 1e-6 + 1e-17, 3e-6, 3e-6]), np.ones(4)
        )
        calc = quiet_calculator(structure, qmax=0.2)
        calc.set_params(sld_solvent=1.0)
        with pytest.warns(UserWarning, match='almost match the solvent'):
            profile = calc.calculate()
        assert any('almost match' in note for note in profile.notes)

    def test_deuterated_equals_a_structure_written_with_d(self):
        xyz = np.random.default_rng(4).normal(size=(6, 3)) * 3
        resnames = ['CTA'] * 3 + ['SOL'] * 3
        resids = [1, 1, 1, 2, 2, 2]
        hydrogen = Structure.from_atoms(
            xyz, ['C', 'H', 'H', 'O', 'H', 'H'], resids=resids, resnames=resnames
        )
        written = Structure.from_atoms(
            xyz, ['C', 'D', 'D', 'O', 'H', 'H'], resids=resids, resnames=resnames
        )
        calc = quiet_calculator(hydrogen, qmax=0.5)
        calc.set_params(sld_solvent=-0.56, deuterated='CTA')
        assert calc.get_params()['deuterated'] == ('CTA',)
        reference = quiet_calculator(written, qmax=0.5)
        reference.set_params(sld_solvent=-0.56)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Atomic volumes')
            assert np.allclose(calc.calculate().intensity, reference.calculate().intensity)
        colours = calc.plot_points(color='sld', show=False).data[0].marker.color
        assert np.isclose(colours, constants('D')[0] * 1e6).sum() == 2  # the two CTA hydrogens

        calc.set_params(deuterated=['CTAB'])
        with pytest.raises(StructureError, match=r'CTAB\. Pass set_params\(deuterated=\(\)\)'):
            calc.calculate()
        with pytest.raises(StructureError, match='CTAB'):
            calc.describe()  # counts zero contrast with the deuterated SLD
        with pytest.raises(ValueError, match='residue names'):
            calc.set_params(deuterated=[1])
        calc.set_params(deuterated=None)
        assert calc.get_params()['deuterated'] == ()

    def test_solvent_in_wrong_units_warns(self):
        with pytest.warns(UserWarning, match='Did you mean 6.4'):
            quiet_calculator().set_params(sld_solvent=6.4e-6)

    def test_cell_size_check_for_pixels(self):
        calc = quiet_calculator(sphere_pixels(radius=6.0, step=3.0))
        calc.set_params(sld_solvent=6.34)
        calc.set_q_grid(qmax=0.5)
        with pytest.warns(UserWarning, match='q_max \\* cell size'):
            calc.calculate()  # 0.5 * 3 > 1
        calc.set_q_grid(qmax=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            calc.calculate()  # 0.3 * 3 < 1: no warning

    def test_atoms_in_solvent_warn_with_the_volume_ratio(self):
        calc = quiet_calculator(C60)
        calc.set_params(sld_solvent=6.34)
        with pytest.warns(UserWarning, match='times a protein-like estimate'):
            calc.calculate()


# ============================================================================
# Kernels and the spectrum cache
# ============================================================================


def counting(calls):
    """A progress callback that counts its calls."""

    def progress(fraction):
        calls.append(fraction)

    return progress


class TestKernels:
    def test_atoms_default_to_the_histogram(self):
        calc = quiet_calculator(C60)
        histogram = calc.calculate()
        assert histogram.method == 'histogram'
        assert histogram.histogram_bin == pytest.approx(1e-3 / 0.3)
        assert 'histogram (bin 0.00333 Ang)' in histogram._process_text()
        calc.set_params(kernel='pairs')
        pairs = calc.calculate()
        assert pairs.method == 'pairs' and pairs.histogram_bin is None
        assert np.allclose(histogram.intensity, pairs.intensity, rtol=1e-4)

    def test_bin_follows_q_max_and_can_be_set(self):
        calc = quiet_calculator(C60)
        calc.set_q_grid(qmax=0.05)
        assert calc.calculate().histogram_bin == 0.01  # never coarser than 0.01 Ang
        calc.set_q_grid(qmax=3.0)
        assert calc.calculate().histogram_bin == pytest.approx(1e-3 / 3.0)
        explicit = quiet_calculator(C60)
        explicit.set_params(histogram_bin=0.02)
        assert explicit.calculate().histogram_bin == 0.02

    def test_kernel_choices_and_refusals(self):
        calc = quiet_calculator(C60)
        calc.set_params(kernel='lattice')
        with pytest.raises(ValueError, match="kernel='lattice' needs points on a lattice"):
            calc.calculate()
        with pytest.raises(ValueError, match='kernel must be one of'):
            calc.set_params(kernel='fast')
        with pytest.raises(ValueError, match='histogram_bin'):
            calc.set_params(histogram_bin=0.0)
        pixels = quiet_calculator(sphere_pixels(radius=6.0))
        assert pixels.calculate().method == 'lattice'
        pixels.set_params(kernel='histogram')
        assert pixels.calculate().method == 'histogram'

    def test_the_spectrum_is_reused_across_solvents(self):
        calls = []
        calc = quiet_calculator(C60)
        calc.calculate(progress=counting(calls))
        assert calls
        calls.clear()
        calc.set_params(sld_solvent=6.34, scale=2.0, background=0.1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reused = calc.calculate(progress=counting(calls))
            assert calls == []  # no pair pass
            fresh = quiet_calculator(C60)
            fresh.set_params(sld_solvent=6.34, scale=2.0, background=0.1)
            assert np.allclose(reused.intensity, fresh.calculate().intensity, rtol=1e-12)

    def test_a_finer_cached_bin_is_reused(self):
        calls = []
        calc = quiet_calculator(C60)
        calc.set_params(histogram_bin=0.001)
        calc.calculate()
        calc.set_params(histogram_bin=0.002)
        profile = calc.calculate(progress=counting(calls))
        assert calls == [] and profile.histogram_bin == 0.001
        calc.set_params(histogram_bin=0.0005)
        calc.calculate(progress=counting(calls))
        assert calls  # a finer bin needs a new pass
        calc.set_params(deuterated=[], kernel='histogram')
        calls.clear()
        calc.calculate(progress=counting(calls))
        assert calls == []  # 'auto' and 'histogram' resolve to the same kernel


# ============================================================================
# Calculator state and output
# ============================================================================


class TestCalculator:
    def test_setters_clear_the_profile_and_failures_are_atomic(self):
        calc = quiet_calculator(C60)
        first = calc.calculate()
        calc.set_params(normalisation_volume=100.0)
        assert calc.profile is None
        with pytest.raises(ValueError):
            calc.set_params(scale=2.0, background=-1.0)
        assert calc.get_params()['scale'] == 1.0
        with pytest.raises(StructureError):
            calc.load_structure('missing.pdb')
        assert calc.structure.n_points == 60
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            calc.load_structure(C60)
        assert calc.get_params()['normalisation_volume'] == 'auto'
        with pytest.raises(AttributeError):
            calc.structure = None
        assert first.params['normalisation_volume'] == 'auto'  # the old profile is unchanged

    def test_q_grid_validation(self):
        calc = GenericScatteringCalculator()
        with pytest.raises(ValueError, match='spacing'):
            calc.set_q_grid(spacing='cubic')
        with pytest.raises(ValueError, match='npoints'):
            calc.set_q_grid(npoints=1)
        with pytest.raises(ValueError, match='qmin < qmax'):
            calc.set_q_grid(qmin=0.3, qmax=0.1)
        for bad in ([], [0.0, 0.1], [0.2, 0.1], [0.1, np.nan]):
            with pytest.raises(ValueError):
                calc.set_q(bad)
        calc.set_q_grid(1e-2, 1e-1, 10, spacing='linear')
        assert np.allclose(calc.q, np.linspace(1e-2, 1e-1, 10))

    def test_no_structure_and_no_profile(self):
        calc = GenericScatteringCalculator()
        with pytest.raises(ValueError, match='load_structure'):
            calc.calculate()
        with pytest.raises(ValueError, match='calculate'):
            calc.plot_profile(show=False)

    def test_profiles_and_structures_are_snapshots(self):
        xyz = np.array([[0.0, 0, 0], [0, 0, 5]])
        values = np.array([1e-6, 2e-6])
        structure = Structure.from_arrays(xyz, values, np.ones(2))
        values[0] = 9.0  # the caller edits their array afterwards
        assert structure.sld[0] == 1e-6
        profile = quiet_calculator(structure, qmax=0.2).calculate()
        params = profile.params
        params['scale'] = 99
        assert profile.params['scale'] == 1.0
        for clone in (pickle.loads(pickle.dumps(profile)), copy.deepcopy(profile)):
            assert np.array_equal(clone.intensity, profile.intensity)
        assert np.array_equal(pickle.loads(pickle.dumps(structure)).xyz, structure.xyz)

    def test_progress_stop_leaves_the_profile_unchanged(self):
        structure = Structure.from_arrays(
            np.random.default_rng(0).normal(size=(200, 3)), np.full(200, 1e-6), np.ones(200)
        )
        calc = quiet_calculator(structure)
        with pytest.raises(CalculationInterrupted):
            calc.calculate(progress=lambda fraction: True)
        assert calc.profile is None
        with pytest.raises(KeyboardInterrupt):
            calc.calculate(progress=lambda fraction: (_ for _ in ()).throw(KeyboardInterrupt))

    def test_to_data1d_is_fit_ready(self):
        calc = quiet_calculator(sphere_pixels())
        calc.set_params(sld_solvent=6.34)
        profile = calc.calculate()
        data = profile.to_data1d()
        assert np.array_equal(data.x, profile.q) and np.array_equal(data.y, profile.intensity)
        assert data.dy is None and data.dx is None
        assert (data.qmin, data.qmax) == (profile.q[0], profile.q[-1])
        assert data.mask.shape == profile.q.shape and not data.mask.any()
        assert data.title.startswith('GSC ')
        assert data.process[-1].name == 'sinccity'
        assert 'sld_solvent=6.34' in data.process[-1].description
        data.x[0] = -1.0  # the Data1D owns its arrays
        assert profile.q[0] > 0
        with_di = profile.to_data1d(dI=np.full(profile.q.size, 0.1))
        assert np.all(with_di.dy == 0.1)
        with pytest.raises(ValueError, match='dI'):
            profile.to_data1d(dI=[1.0])
        with pytest.raises(ValueError, match='dI'):
            profile.to_data1d(dI=np.full(profile.q.size, -0.1))

    def test_describe_settings_and_replay(self, tmp_path, capsys):
        path = tmp_path / 'c60.pdb'
        path.write_bytes(open(C60, 'rb').read())
        calc = quiet_calculator(path)
        calc.set_q_grid(1e-3, 0.5, 20, spacing='linear')
        calc.set_params(scale=2.0, normalisation_volume=600.0)
        calc.describe()
        profile = calc.calculate()
        profile.describe()
        assert 'V_norm' in capsys.readouterr().out and 'V_norm' in profile._repr_html_()
        assert 'Rg (mass)' in calc.structure._repr_html_()
        settings = json.loads(json.dumps(calc.get_settings()))
        assert settings['params']['normalisation_volume'] == 600.0
        assert settings['versions']['periodictable']
        assert settings['versions']['sinccity'] == sinccity.__version__
        replay = GenericScatteringCalculator.from_settings(settings)
        assert np.allclose(replay.calculate().intensity, profile.intensity)
        with open(path, 'a') as handle:
            handle.write('REMARK changed\n')
        with pytest.warns(UserWarning, match='has changed'):
            GenericScatteringCalculator.from_settings(settings)
        profile.save_csv(tmp_path / 'profile.csv')
        assert np.loadtxt(tmp_path / 'profile.csv', delimiter=',').shape == (20, 2)

    def test_settings_without_a_kernel_replay_the_kernel_of_0_1_0(self, tmp_path):
        calc = quiet_calculator(C60)
        calc.set_params(kernel='pairs')
        recorded = calc.calculate().intensity
        settings = calc.get_settings()
        for key in ('kernel', 'histogram_bin', 'deuterated'):  # 0.1.0 had none of them
            del settings['params'][key]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # the guessed elements of c60.pdb
            replay = GenericScatteringCalculator.from_settings(settings).calculate()
        assert replay.method == 'pairs' and np.array_equal(replay.intensity, recorded)

        path = tmp_path / 'pixels.sld'
        corners = [f'{x} {y} {z} 1e-6' for x in (0, 2) for y in (0, 2) for z in (0, 2)]
        path.write_text('\n'.join(['x y z sld', *corners]) + '\n')
        pixels = quiet_calculator(sinccity.load(path)).get_settings()
        del pixels['params']['kernel']
        assert GenericScatteringCalculator.from_settings(pixels).get_params()['kernel'] == 'auto'

    def test_settings_replay_deuteration(self, tmp_path):
        path = tmp_path / 'cta.pdb'
        atoms = [('C1', 0.0, 'C'), ('H1', 1.1, 'H'), ('H2', -1.1, 'H')]
        path.write_text(
            ''.join(
                f'ATOM  {i:>5} {name:<4} CTA A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}'
                f'{1.0:6.2f}{0.0:6.2f}          {element:>2}\n'
                for i, (name, x, element) in enumerate(atoms, start=1)
            )
        )
        calc = quiet_calculator(sinccity.load(path))
        calc.set_params(sld_solvent=-0.56, deuterated='CTA')
        settings = json.loads(json.dumps(calc.get_settings()))
        assert settings['params']['deuterated'] == ['CTA']
        replay = GenericScatteringCalculator.from_settings(settings)
        assert replay.get_params()['deuterated'] == ('CTA',)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Atomic volumes')
            assert np.allclose(replay.calculate().intensity, calc.calculate().intensity)

    def test_settings_keep_the_reader_options_of_an_installed_structure(self, tmp_path):
        path = tmp_path / 'two.pdb'
        path.write_text(
            'ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n'
            'HETATM    2 FE   HEM A   2       2.000   0.000   0.000  1.00  0.00          FE\n'
        )
        calc = quiet_calculator(sinccity.load(path, include_hetatm=True))
        settings = json.loads(json.dumps(calc.get_settings()))
        assert settings['reader_options']['include_hetatm'] is True
        replay = GenericScatteringCalculator.from_settings(settings)
        assert replay.structure.n_points == 2

    def test_in_memory_settings_cannot_be_replayed(self):
        calc = quiet_calculator(sphere_pixels(radius=6.0))
        settings = calc.get_settings()
        assert settings['source'] == 'in-memory'
        with pytest.raises(ValueError, match='set_structure'):
            GenericScatteringCalculator.from_settings(settings)

    def test_optional_dependencies_are_imported_lazily(self):
        code = (
            "import sys, sinccity; sys.exit(any(m in sys.modules for m in ('plotly', 'sasdata')))"
        )
        assert subprocess.run([sys.executable, '-c', code]).returncode == 0

    def test_set_verbosity(self, capsys):
        calc = quiet_calculator(C60)
        capsys.readouterr()  # drop the load line
        sinccity.set_verbosity('quiet')
        try:
            calc.set_q_grid(qmax=0.2)
            assert capsys.readouterr().out == ''
        finally:
            sinccity.set_verbosity('info')
        calc.set_q_grid(qmax=0.2)
        assert 'q grid' in capsys.readouterr().out
        with pytest.raises(ValueError, match='Unknown verbosity'):
            sinccity.set_verbosity('loud')


# ============================================================================
# Plots
# ============================================================================


class TestPlots:
    def test_points_by_element_with_bonds(self):
        fig = quiet_calculator(C60).plot_points(show=False)
        assert [t.name for t in fig.data] == ['C', 'bonds']

    def test_thinning_reports_counts_and_keeps_the_lattice(self):
        structure = sphere_pixels(radius=12.0, step=1.0)
        fig = structure.plot_points(max_points=2000, show=False)
        shown = sum(len(t.x) for t in fig.data)
        assert shown <= 2000 and f'showing {shown} of {structure.n_points}' in fig.layout.title.text
        atoms = Structure.from_arrays(
            np.random.default_rng(0).normal(size=(3000, 3)) * 10,
            np.full(3000, 1e-6),
            np.ones(3000),
            symbols=['C'] * 3000,
        )
        assert sum(len(t.x) for t in atoms.plot_points(max_points=500, show=False).data) <= 500

    def test_zero_contrast_points_follow_the_solvent(self):
        calc = quiet_calculator(sphere_pixels(radius=6.0))
        calc.set_params(sld_solvent=6.34)
        names = [t.name for t in calc.plot_points(show=False).data]
        assert names == ['SLD', 'zero contrast']

    def test_profile_plot(self):
        calc = quiet_calculator(C60)
        profile = calc.calculate()
        fig = calc.plot_profile(data=profile.to_data1d(), show=False)
        assert len(fig.data) == 2 and fig.layout.xaxis.type == 'log'
        assert fig.data[0].error_y.array is None  # no dI: no error bars
        with_di = profile.to_data1d(dI=np.full(profile.q.size, 0.5))
        assert len(calc.plot_profile(data=with_di, show=False).data[0].error_y.array) == 30
        zero = Structure.from_arrays([[0.0, 0, 0]], [1e-6], [1.0])
        flat = quiet_calculator(zero)
        flat.set_params(sld_solvent=1.0)
        with pytest.warns(UserWarning):
            flat.calculate()
        assert flat.plot_profile(show=False).layout.yaxis.type == 'linear'
