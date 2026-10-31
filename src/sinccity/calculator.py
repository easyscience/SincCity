"""The Generic Scattering Calculator: I(q) of a real-space structure."""

import copy
import html
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import __version__
from ._console import ANGSTROM, CUBIC_ANGSTROM, INVERSE_ANGSTROM, OK, SLD_UNIT, logger
from .debye import (
    LATTICE_MAX_CELLS,
    CalculationInterrupted,
    debye_pairs,
    lattice_cells,
    spectrum_histogram,
    spectrum_lattice,
    spectrum_radial,
)
from .dispersion import gaussian_sizes
from .elements import AVOGADRO
from .model import PROTEIN_SPECIFIC_VOLUME, Structure, StructureError, residue_names
from .readers import load

#: SasView's default 1D grid: 30 log-spaced points up to 0.3 1/Å.
DEFAULT_Q_GRID = {'qmin': 3e-4, 'qmax': 0.3, 'npoints': 30, 'spacing': 'log'}

#: Above this many active points the pair loop warns about run time.
LARGE_PAIR_LOOP = 20_000

#: Above this many active points the histogram pass warns about run time (about
#: 6 min at 130 000 atoms).
LARGE_HISTOGRAM = 100_000

#: The ``kernel`` choices; ``'auto'`` is lattice when possible, else histogram.
KERNELS = ('auto', 'lattice', 'histogram', 'average', 'pairs')

#: The largest histogram bin, Å (coords2sans's, validated to 1e-4 up to q = 0.2 1/Å).
HISTOGRAM_BIN_MAX = 0.01

#: Largest phase error the automatic bin allows at f_max * q_max, rad; rounding
#: moves a pair by at most half a bin, so the bin is 1e-3 / (q_max f_max).
HISTOGRAM_PHASE = 1e-3

#: A population is "near-matched" when its contrast is this small relative to the largest.
NEAR_MATCH = 1e-6

#: The point-per-cell approximation is flagged above q * cell size = 1.
CELL_SIZE_LIMIT = 1.0

#: The parameters ``set_params`` takes, and their defaults.
DEFAULT_PARAMS: dict[str, Any] = {
    'scale': 1.0,
    'background': 0.0,
    'sld_solvent': 0.0,
    'normalisation_volume': 'auto',
    'deuterated': (),
    'kernel': 'auto',
    'histogram_bin': 'auto',
    'size_pd': 0.0,
    'size_pd_n': 35,
    'size_pd_nsigma': 3.0,
}

_UNSET = object()


def _check_param(name: str, value: Any) -> Any:
    """*value* checked and normalised for the parameter *name*.

    Raises:
        ValueError: If the value is not allowed.
    """
    if name in ('normalisation_volume', 'histogram_bin') and value == 'auto':
        return value
    if name == 'deuterated':
        return () if value is None else tuple(residue_names(value, 'deuterated'))
    if name == 'kernel':
        if value not in KERNELS:
            raise ValueError(f'kernel must be one of {", ".join(KERNELS)}, got {value!r}.')
        return value
    if name == 'size_pd_n':
        if int(value) != value or value < 1:
            raise ValueError(f'size_pd_n must be an integer of at least 1, got {value!r}.')
        return int(value)
    value = float(value)
    if name == 'sld_solvent':
        if not np.isfinite(value):
            raise ValueError(f'sld_solvent must be finite, got {value}.')
        if 0 < abs(value) < 1e-3:
            warnings.warn(
                f'sld_solvent={value:g} is in 10^-6 Ang^-2 and looks like a value in Ang^-2. '
                f'Did you mean {value * 1e6:g}?',
                stacklevel=4,
            )
        return value
    if name in ('normalisation_volume', 'histogram_bin'):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be > 0 or 'auto', got {value}.")
        return value
    positive = name == 'size_pd_nsigma'  # scale, background and size_pd may be 0
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"> 0" if positive else ">= 0"}, got {value}.')
    return value


def _checked_params(params: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """A copy of *params* with *changes* checked and applied."""
    new = dict(params)
    for name, value in changes.items():
        if name not in DEFAULT_PARAMS:
            raise ValueError(f'Unknown parameter {name!r}; use {", ".join(DEFAULT_PARAMS)}.')
        new[name] = _check_param(name, value)
    return new


def validate_q_grid(q: Any) -> np.ndarray:
    """Return *q* as a float array: 1D, non-empty, finite, positive, strictly increasing.

    The same rules sasmodels needs for a 1D grid, so a grid accepted here can
    be handed on as it is.
    """
    q_values = np.asarray(q, dtype=float)
    if q_values.ndim != 1 or q_values.size == 0:
        raise ValueError('q must be a non-empty 1D array.')
    if not np.all(np.isfinite(q_values)):
        raise ValueError('Q values must be finite.')
    if np.any(q_values <= 0):
        raise ValueError('Q values must be positive.')
    if np.any(np.diff(q_values) <= 0):
        raise ValueError('Q values must be strictly increasing.')
    return q_values


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - a missing distribution must not block a settings record
        return None


def _q_grid(qmin: float, qmax: float, npoints: int, spacing: str) -> np.ndarray:
    if spacing not in ('log', 'linear'):
        raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}.")
    if int(npoints) != npoints or npoints < 2:
        raise ValueError(f'npoints must be an integer of at least 2, got {npoints!r}.')
    if not 0 < qmin < qmax:
        raise ValueError(f'Need 0 < qmin < qmax, got qmin={qmin}, qmax={qmax}.')
    if spacing == 'log':
        return validate_q_grid(np.geomspace(qmin, qmax, int(npoints)))
    return validate_q_grid(np.linspace(qmin, qmax, int(npoints)))


@dataclass(frozen=True, eq=False)
class ScatteringProfile:
    """A computed I(q) with everything needed to know how it was made.

    Attributes:
        q: Momentum transfer, 1/Å.
        intensity: I(q), 1/cm.
        normalisation_volume: The volume V_norm the Debye sum was divided by,
            Å³; nan when nothing scatters and no volume was set.
        normalisation: ``'active-volume'`` (the default) or ``'explicit'``.
        n_zero_contrast: Points whose SLD equals the solvent's.
        method: Kernel used: ``'lattice'``, ``'histogram'``, ``'average'``,
            ``'pairs'``, or ``'none'`` when nothing scatters.
        histogram_bin: The histogram bin used, Å; None for other kernels.
        n_sizes: Number of size factors summed (1 without polydispersity).
        label: The curve's name; the structure file's name unless given.
        deuterated: The residue names whose hydrogens counted as D.
        notes: Warnings raised while reading and calculating.
        elapsed: Calculation time, s.
    """

    q: np.ndarray
    intensity: np.ndarray
    normalisation_volume: float
    normalisation: str
    n_zero_contrast: int
    method: str
    histogram_bin: float | None
    n_sizes: int
    label: str
    deuterated: tuple[str, ...]
    notes: tuple[str, ...]
    elapsed: float
    _params: dict = field(repr=False)
    _structure_summary: dict = field(repr=False)

    def __post_init__(self) -> None:
        for name in ('q', 'intensity'):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def params(self) -> dict[str, Any]:
        """The calculator's parameters (``set_params``), as set."""
        return dict(self._params)

    @property
    def _kernel_text(self) -> str:
        if self.histogram_bin is None:
            return self.method
        return f'{self.method} (bin {self.histogram_bin:.3g} Ang)'

    @property
    def structure_summary(self) -> dict[str, Any]:
        """Source, format, kind, size and reader notes of the structure."""
        return dict(self._structure_summary)

    def _process_text(self) -> str:
        """Provenance in ASCII, for file content."""
        s, p = self._structure_summary, self._params
        text = (
            f'Generic Scattering Calculator: {s["basename"]} ({s["format"]}, {s["kind"]}, '
            f'{s["n_points"]} points, sha256 {str(s["source_sha256"])[:12]}); '
            f'scale={p["scale"]:g}, background={p["background"]:g} 1/cm, '
            f'sld_solvent={p["sld_solvent"]:g} 1e-6/Ang^2, '
            f'V_norm={self.normalisation_volume:.6g} Ang^3 ({self.normalisation}), '
            f'kernel={self._kernel_text}'
        )
        if self.n_sizes > 1:
            text += (
                f', size_pd={p["size_pd"]:g} over {self.n_sizes} sizes '
                f'(nsigma {p["size_pd_nsigma"]:g})'
            )
        if self.notes:
            text += '; notes: ' + ' | '.join(self.notes)
        return text.encode('ascii', 'replace').decode('ascii')

    def to_data1d(self, *, dI=None):
        """Return the curve as a fit-ready sasdata ``Data1D``.

        The curve is theory, so it carries no dQ and, unless *dI* is given, no
        dI. It carries ``qmin``, ``qmax`` and an empty ``mask``, which fit
        engines such as IQuana's expect. Fitting it then needs an explicit
        weighting and no resolution smearing.

        sasdata is an optional dependency (``pip install "sinccity[sasdata]"``).

        Args:
            dI: Optional uncertainties: finite, non-negative, one per q.
        """
        try:
            from sasdata.dataloader.data_info import Data1D, Process
        except ImportError as exc:
            raise ImportError('to_data1d needs sasdata: pip install "sinccity[sasdata]"') from exc
        dy = None
        if dI is not None:
            dy = np.array(dI, dtype=float, copy=True)
            if dy.shape != self.q.shape or not np.all(np.isfinite(dy)) or np.any(dy < 0):
                raise ValueError('dI must be finite, non-negative, and have one value per q.')
        data = Data1D(x=self.q.copy(), y=self.intensity.copy(), dy=dy)
        data.title = f'GSC {self.label}'
        process = Process()
        process.name = 'sinccity'
        process.description = self._process_text()
        data.process.append(process)
        data.qmin, data.qmax = float(self.q[0]), float(self.q[-1])
        data.mask = np.zeros(self.q.size, dtype=bool)
        return data

    def save_csv(self, path) -> None:
        """Write q and I(q) as CSV, with the settings in an ASCII header."""
        header = (
            f'{self._process_text()}\n'
            f'source: {self._structure_summary["source"]}\n'
            'q (1/Ang), I (1/cm)'
        )
        np.savetxt(path, np.column_stack([self.q, self.intensity]), delimiter=',', header=header)

    def _lines(self) -> list[tuple[str, str]]:
        s, p = self._structure_summary, self._params
        rows = [
            ('Label', self.label),
            ('Structure', f'{s["basename"]}: {s["n_points"]} {s["kind"]}'),
            ('q', f'{self.q.size} points, {self.q[0]:.4g} to {self.q[-1]:.4g} {INVERSE_ANGSTROM}'),
            ('I(q)', f'{np.nanmin(self.intensity):.4g} to {np.nanmax(self.intensity):.4g} 1/cm'),
            ('Solvent SLD', f'{p["sld_solvent"]:g} {SLD_UNIT}'),
            ('Deuterated', ', '.join(self.deuterated) or 'none'),
            ('Scale, background', f'{p["scale"]:g}, {p["background"]:g} 1/cm'),
            (
                'V_norm',
                f'{self.normalisation_volume:.6g} {CUBIC_ANGSTROM} ({self.normalisation}); '
                f'{self.n_zero_contrast} zero-contrast points',
            ),
            ('Kernel', f'{self._kernel_text}, {self.elapsed:.3g} s'),
        ]
        if self.n_sizes > 1:
            rows.append(
                (
                    'Sizes',
                    f'size_pd {p["size_pd"]:g}, {self.n_sizes} sizes, nsigma {p["size_pd_nsigma"]:g}',
                )
            )
        rows.extend(('Note', note) for note in self.notes)
        return rows

    def describe(self) -> None:
        """Print how the profile was computed."""
        for label, value in self._lines():
            print(f'  {label:<18} {value}')

    def _repr_html_(self) -> str:
        cells = ''.join(
            f'<tr><th style="text-align:left">{html.escape(k)}</th><td>{html.escape(v)}</td></tr>'
            for k, v in self._lines()
        )
        return f'<table>{cells}</table>'

    def plot(self, *, data=None, log_scale: bool = True, show: bool | None = None):
        """Plot I(q); see ``sinccity.plotting.plot_profile``."""
        from .plotting import plot_profile

        return plot_profile(self, data=data, log_scale=log_scale, show=show)


class GenericScatteringCalculator:
    """Orientationally averaged I(q) of a structure, by the Debye formula.

    The nuclear, 1D part of SasView's Generic Scattering Calculator::

        w_j  = (sld_j - sld_solvent * 1e-6) * volume_j
        I(q) = scale * 1e8 * D(q) / V_norm + background
        D(q) = sum_jk w_j w_k sin(q r_jk) / (q r_jk)

    ``V_norm`` is the volume of the points that scatter (``w != 0``) unless
    ``normalisation_volume`` is set. For space-filling pixels and cells,
    ``scale`` is then a volume fraction, as in sasmodels.

    The ``kernel`` parameter picks how D(q) is summed; see ``set_params``. The
    lattice, histogram and average kernels first build a distance spectrum
    that does not depend on the solvent; the last one is kept, so a change of
    ``sld_solvent``, ``scale``, ``background`` or ``normalisation_volume``
    costs only the sums over q.

    Example:
        >>> gsc = GenericScatteringCalculator('particle.sld')
        >>> gsc.set_params(sld_solvent=6.34)
        >>> profile = gsc.calculate()
        >>> gsc.plot_profile()
    """

    def __init__(self, structure: Structure | str | os.PathLike | None = None) -> None:
        self._structure: Structure | None = None
        self._profile: ScatteringProfile | None = None
        self._q_args: dict[str, Any] | None = dict(DEFAULT_Q_GRID)
        self._q = _q_grid(**DEFAULT_Q_GRID)
        self._params: dict[str, Any] = dict(DEFAULT_PARAMS)
        # (structure, deuterated, kernel, active mask, bin, spectrum) of the last pair pass
        self._cache: tuple | None = None
        if isinstance(structure, Structure):
            self.set_structure(structure)
        elif structure is not None:
            self.load_structure(structure)

    # ------------------------------------------------------------- properties

    @property
    def structure(self) -> Structure | None:
        return self._structure

    @property
    def profile(self) -> ScatteringProfile | None:
        """The result of the last ``calculate()``, cleared by every setter."""
        return self._profile

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    # ---------------------------------------------------------------- setters

    def load_structure(self, path, **reader_options: Any) -> Structure:
        """Read a PDB, GRO, SLD or VTK file and make it the current structure.

        Args:
            path: The structure file.
            **reader_options: As ``sinccity.load`` takes them.
        """
        structure = load(path, **reader_options)
        self._install(structure)
        return structure

    def set_structure(self, structure: Structure) -> None:
        """Use an existing ``Structure``, for example one from ``sinccity.load``."""
        if not isinstance(structure, Structure):
            raise TypeError(f'Expected a Structure, got {type(structure).__name__}.')
        self._install(structure)

    def _install(self, structure: Structure) -> None:
        self._structure = structure
        self._params['normalisation_volume'] = 'auto'
        self._profile = None
        self._cache = None
        for note in structure.notes:
            warnings.warn(f'{os.path.basename(structure.source)}: {note}', stacklevel=3)
        rg = structure.radius_of_gyration('sld', sld_solvent=self._params['sld_solvent'])
        detail = (
            f'Rg {rg.value:.3g} {ANGSTROM}' if np.isfinite(rg.value) else f'Rg n/a ({rg.reason})'
        )
        if structure.lattice is not None:
            detail += (
                f', lattice step {[round(s, 6) for s in structure.lattice.step]} {ANGSTROM}, '
                f'pixel volume {structure.volume[0]:.6g} {CUBIC_ANGSTROM}'
            )
        logger.info(
            f'{OK} Loaded {os.path.basename(structure.source)}: {structure.n_points} '
            f'{structure.kind}, {detail}; normalisation volume reset to auto'
        )

    def set_q_grid(
        self, qmin: float = 3e-4, qmax: float = 0.3, npoints: int = 30, *, spacing: str = 'log'
    ) -> None:
        """Build the q grid: *npoints* from *qmin* to *qmax* (1/Å), log or linear.

        Unlike SasView, qmin does not follow qmax; ``set_q_grid(qmax=0.5)``
        keeps qmin = 3e-4.
        """
        args = {'qmin': qmin, 'qmax': qmax, 'npoints': npoints, 'spacing': spacing}
        self._q = _q_grid(**args)
        self._q_args = args
        self._profile = None
        logger.info(
            f'{OK} q grid: {npoints} {spacing} points, {qmin:g} to {qmax:g} {INVERSE_ANGSTROM}'
        )

    def set_q(self, q) -> None:
        """Use explicit q values (1/Å): positive, finite and increasing."""
        self._q = validate_q_grid(q).copy()
        self._q_args = None
        self._profile = None
        logger.info(f'{OK} q grid: {self._q.size} explicit values')

    def set_params(
        self,
        *,
        scale=_UNSET,
        background=_UNSET,
        sld_solvent=_UNSET,
        normalisation_volume=_UNSET,
        deuterated=_UNSET,
        kernel=_UNSET,
        histogram_bin=_UNSET,
        size_pd=_UNSET,
        size_pd_n=_UNSET,
        size_pd_nsigma=_UNSET,
    ) -> None:
        """Set any of the calculation parameters; the others are kept.

        Args:
            scale: Multiplies I(q); >= 0.
            background: Flat background, 1/cm; >= 0, because a computed curve
                has no subtraction residue to correct.
            sld_solvent: Solvent SLD in 10⁻⁶ Å⁻², as in every sasmodels model
                (D2O is 6.34). SasView's calculator takes Å⁻² instead.
            normalisation_volume: V_norm in Å³, or ``'auto'`` for the volume of
                the points that scatter. SasView calls this field total_volume.
            deuterated: Residue names whose hydrogens count as D, applied with
                ``Structure.deuterated`` at ``calculate()``; None or ``()`` for none.
            kernel: How D(q) is summed. ``'auto'`` (the default): ``'lattice'``
                for pixels on a lattice whose padded grid fits, else
                ``'histogram'``. ``'lattice'``: FFT autocorrelation, exact.
                ``'histogram'``: pairs counted by distance, exact to about
                1e-4. ``'average'``: the squared orientation-averaged amplitude
                about the bounding-box centre, exact only for spherically
                symmetric structures, with time linear in the points.
                ``'pairs'``: every pair at every q, exact to the last digit.
            histogram_bin: Bin width of the histogram kernel, Å, or ``'auto'``
                for ``min(0.01, 1e-3 / (q_max * f_max))``, f_max the largest
                size factor.
            size_pd: Relative standard deviation of the size of the whole
                structure (a Gaussian, as sasmodels' ``radius_pd``); 0 is one size.
            size_pd_n: Number of sizes, >= 1.
            size_pd_nsigma: Half-range of the sizes in standard deviations, > 0.
        """
        changes = {k: v for k, v in locals().items() if k != 'self' and v is not _UNSET}
        self._params = _checked_params(self._params, changes)
        self._profile = None
        logger.info(f'{OK} Parameters: {self._describe_params()}')

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def _describe_params(self) -> str:
        p = self._params
        volume = p['normalisation_volume']
        volume = volume if volume == 'auto' else f'{volume:.6g} {CUBIC_ANGSTROM}'
        text = (
            f'scale {p["scale"]:g}, background {p["background"]:g} 1/cm, '
            f'sld_solvent {p["sld_solvent"]:g} {SLD_UNIT}, normalisation volume {volume}'
        )
        if p['deuterated']:
            text += f', deuterated {", ".join(p["deuterated"])}'
        text += f', kernel {p["kernel"]}'
        if p['histogram_bin'] != 'auto':
            text += f' (bin {p["histogram_bin"]:g} {ANGSTROM})'
        if p['size_pd'] > 0:
            text += (
                f', size_pd {p["size_pd"]:g} ({p["size_pd_n"]} sizes, '
                f'nsigma {p["size_pd_nsigma"]:g})'
            )
        return text

    # ------------------------------------------------------------ calculation

    def _require_structure(self) -> Structure:
        if self._structure is None:
            raise ValueError('No structure loaded. Use load_structure() first.')
        return self._structure

    def _scattering_structure(self) -> Structure:
        """The current structure as it scatters: the ``deuterated`` hydrogens as D."""
        structure = self._require_structure()
        if not self._params['deuterated']:
            return structure
        try:
            return structure.deuterated(self._params['deuterated'])
        except StructureError as exc:
            raise StructureError(f'{exc} Pass set_params(deuterated=()) to clear it.') from exc

    def calculate(
        self, *, progress: Callable[[float], bool | None] | None = None, label: str | None = None
    ) -> ScatteringProfile:
        """Compute I(q) on the current grid and return a ``ScatteringProfile``.

        Args:
            progress: Optional callback for long pair passes. It receives the
                completed fraction; returning True stops the calculation with
                ``CalculationInterrupted``, leaving the previous state intact.
            label: The curve's name; the structure file's name by default.
        """
        profile = self._compute(progress, label)
        self._profile = profile
        return profile

    def calculate_series(
        self,
        series: list[dict[str, Any]],
        *,
        progress: Callable[[float], bool | None] | None = None,
    ) -> list[ScatteringProfile]:
        """Compute one profile per entry of *series*, on the current structure and q grid.

        Each entry is ``{'label': ..., 'params': {...}}``: ``set_params`` keys
        that override the current parameters for that entry only. Every entry
        is checked before anything runs. Entries run grouped by deuteration
        and kernel, so the pair pass runs once per group, and come back in the
        order given. A series is plain JSON, so ``get_settings()`` plus a series
        is a task another process runs with
        ``from_settings(settings).calculate_series(series)``.

        Afterwards the parameters are those before the call, and ``profile``
        is that of the last entry.

        Args:
            series: The entries.
            progress: Receives the completed fraction of the whole series;
                returning True stops it with ``CalculationInterrupted`` and
                leaves the parameters and ``profile`` as they were.

        Raises:
            ValueError: Naming the entry that is not a dict, or whose key or
                value is not allowed, including deuterated residue names
                the structure lacks.
        """
        structure = self._require_structure()
        entries = []
        for number, entry in enumerate(series):
            label = entry.get('label') if isinstance(entry, dict) else None
            try:
                if not isinstance(entry, dict):
                    raise ValueError('an entry is a dict of label and params.')
                unknown = sorted(set(entry) - {'label', 'params'})
                if unknown:
                    raise ValueError(f'unknown keys {", ".join(unknown)}; use label and params.')
                changes = entry.get('params', {})
                if not isinstance(changes, dict):
                    raise ValueError('params is a dict of set_params keys.')
                params = _checked_params(self._params, changes)
                if params['deuterated']:
                    structure.deuterated(params['deuterated'])  # names not in the structure
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Series entry {number} ({label!r}): {exc}') from exc
            entries.append((label, params))

        def group(index: int) -> tuple:
            p = entries[index][1]
            return p['deuterated'], p['kernel'], self._bin_width(p)  # finer bins first

        def overall(done: int):
            """The progress callback of one entry, reporting the fraction of the series."""
            if progress is None:
                return None
            return lambda fraction: progress((done + fraction) / len(entries))

        profiles: list[ScatteringProfile | None] = [None] * len(entries)
        saved = self._params
        try:
            for done, index in enumerate(sorted(range(len(entries)), key=group)):
                label, params = entries[index]
                self._params = params  # the entry's parameters, for its profile only
                report = overall(done)
                profiles[index] = self._compute(report, label)
                if report is not None and report(1.0):
                    raise CalculationInterrupted('Series stopped by the progress callback.')
        finally:
            self._params = saved
        if profiles:
            self._profile = profiles[-1]
        return profiles

    def _compute(self, progress, label: str | None) -> ScatteringProfile:
        """The profile for the current parameters; the caller decides whether to keep it."""
        structure = self._require_structure()
        p = self._params
        q = self._q
        notes = list(structure.notes)
        sld = self._scattering_structure().sld
        rho = p['sld_solvent'] * 1e-6
        contrast = sld - rho
        active = contrast * structure.volume != 0
        n_zero = int(np.sum(~active))
        explicit = p['normalisation_volume'] != 'auto'
        start = time.perf_counter()
        bin_width = None
        factors, shares = self._sizes(p)

        if not active.any():
            self._warn(
                notes,
                f'Every point matches the solvent SLD ({p["sld_solvent"]:g}); I(q) is the background.',
            )
            intensity = np.full(q.size, p['background'])
            v_norm = p['normalisation_volume'] if explicit else float('nan')
            method = 'none'
        else:
            # the shares' mean volume, as sasmodels normalises a size distribution
            mean_volume = float(np.sum(shares * factors**3) * structure.volume[active].sum())
            v_norm = p['normalisation_volume'] if explicit else mean_volume
            method = self._kernel(structure, active, p['kernel'])
            self._check(structure, contrast, active, factors.max(), notes)
            debye, bin_width = self._debye(method, structure, sld, active, progress, notes)
            intensity = p['scale'] * 1e8 * debye / v_norm + p['background']

        summary = {
            'source': structure.source,
            'basename': os.path.basename(structure.source),
            'source_sha256': structure.source_sha256,
            'format': structure.format,
            'kind': structure.kind,
            'n_points': structure.n_points,
            'reader_options': copy.deepcopy(structure.reader_options),
            'element_sources': dict(structure.element_sources),
            'notes': list(structure.notes),
        }
        profile = ScatteringProfile(
            q=q,
            intensity=intensity,
            normalisation_volume=float(v_norm),
            normalisation='explicit' if explicit else 'active-volume',
            n_zero_contrast=n_zero,
            method=method,
            histogram_bin=bin_width,
            n_sizes=int(factors.size),
            label=os.path.basename(structure.source) if label is None else str(label),
            deuterated=p['deuterated'],
            notes=tuple(notes),
            elapsed=time.perf_counter() - start,
            _params=dict(p),
            _structure_summary=summary,
        )
        logger.info(
            f'{OK} I(q) at {q.size} q values: {int(active.sum())} scattering points, '
            f'{n_zero} at zero contrast, V_norm {v_norm:.6g} {CUBIC_ANGSTROM} '
            f'({profile.normalisation}), {profile._kernel_text} kernel, {profile.elapsed:.3g} s'
        )
        return profile

    @staticmethod
    def _warn(notes: list[str], text: str) -> None:
        warnings.warn(text, stacklevel=5)
        notes.append(text)

    def _check(self, structure: Structure, contrast, active, f_max, notes: list[str]) -> None:
        """Warnings that change how the curve should be read."""
        largest = np.abs(contrast).max()
        near = active & (np.abs(contrast) <= NEAR_MATCH * largest)
        if near.any():
            self._warn(
                notes,
                f'{int(near.sum())} points almost match the solvent (|contrast| <= '
                f'{NEAR_MATCH:g} of the largest) but still count towards V_norm. Set '
                'normalisation_volume explicitly for contrast-variation series.',
            )

        size = None
        if structure.kind == 'pixels':
            if structure.lattice is not None:
                size = max(structure.lattice.step)
            else:
                size = float(np.cbrt(structure.volume.max()))
        elif structure.kind == 'cells':
            size = float(structure.cell_diameter.max() / np.sqrt(3))
        q_max = self._q[-1] * f_max  # the largest size samples the spectrum furthest out
        if size and q_max * size > CELL_SIZE_LIMIT:
            self._warn(
                notes,
                f'q_max * cell size = {q_max * size:.3g} > {CELL_SIZE_LIMIT:g} (cell size '
                f'{size:.4g} Ang, largest size factor {f_max:.3g}): treating each '
                f'{structure.kind[:-1]} as a point degrades above q = '
                f'{CELL_SIZE_LIMIT / size:.4g} 1/Ang.',
            )

        if structure.kind == 'atoms' and self._params['sld_solvent'] != 0:
            elemental = float(structure.volume[active].sum())
            protein = float(
                structure.masses[active].sum() * PROTEIN_SPECIFIC_VOLUME / AVOGADRO * 1e24
            )
            self._warn(
                notes,
                f'Atomic volumes come from elemental densities: {elemental:.4g} Ang^3 in total, '
                f'{elemental / protein:.2f} times a protein-like estimate ({protein:.4g} Ang^3 at '
                f'{PROTEIN_SPECIFIC_VOLUME} cm^3/g). They are not molecular excluded volumes, so '
                'in a solvent I(q) is only qualitative; there is no hydration shell or H/D '
                'exchange either.',
            )

    @staticmethod
    def _lattice_index(structure: Structure, active) -> tuple[np.ndarray, tuple[int, ...]]:
        """Lattice indices of the active points, from 0, and the grid shape they need."""
        index = structure.lattice.indices[active]
        index = index - index.min(axis=0)
        return index, tuple(int(n) for n in index.max(axis=0) + 1)

    def _kernel(self, structure: Structure, active, requested: str) -> str:
        """The kernel *requested* resolves to for this structure."""
        fits = (
            structure.lattice is not None
            and lattice_cells(self._lattice_index(structure, active)[1]) <= LATTICE_MAX_CELLS
        )
        if requested == 'auto':
            return 'lattice' if fits else 'histogram'
        if requested == 'lattice' and not fits:
            raise ValueError(
                "kernel='lattice' needs points on a lattice whose zero-padded grid has at most "
                f'{LATTICE_MAX_CELLS} cells.'
            )
        return requested

    @staticmethod
    def _sizes(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """The size factors and their shares, from the ``size_pd`` parameters in *p*."""
        return gaussian_sizes(p['size_pd'], p['size_pd_n'], p['size_pd_nsigma'])

    def _bin_width(self, p: dict[str, Any]) -> float:
        """The histogram bin for *p*, Å: as set, or the largest that keeps the phase error small."""
        if p['histogram_bin'] != 'auto':
            return p['histogram_bin']
        return min(HISTOGRAM_BIN_MAX, HISTOGRAM_PHASE / (self._q[-1] * self._sizes(p)[0].max()))

    def _debye(self, kernel, structure, sld, active, progress, notes):
        """D(q) of the *active* points with SLDs *sld*, summed over the sizes.

        A size f contributes ``share * f^6 * D(f q)``. Also returns the
        histogram bin used, None for the other kernels.
        """
        q = self._q
        rho = self._params['sld_solvent'] * 1e-6
        n = int(active.sum())
        factors, shares = self._sizes(self._params)

        def over_sizes(one) -> np.ndarray:
            return sum(s * f**6 * one(i, f) for i, (f, s) in enumerate(zip(factors, shares)))

        last = [time.monotonic()]

        def report(fraction: float) -> bool | None:
            if time.monotonic() - last[0] > 5:
                logger.info(f'  {kernel} pass {fraction:.0%} done')
                last[0] = time.monotonic()
            return progress(fraction) if progress is not None else None

        if kernel == 'pairs':
            if n > LARGE_PAIR_LOOP:
                self._warn(
                    notes,
                    f'{n} scattering points: the exact pair sum scales as N^2 and may take many '
                    "minutes; kernel='histogram' gives the same to about 1e-4 in a fraction of it.",
                )
            if factors.size > 1:
                self._warn(
                    notes,
                    f"kernel='pairs' repeats the whole pair sum for each of the {factors.size} "
                    "sizes; kernel='histogram' does the same in a fraction of the time.",
                )
            xyz = structure.xyz[active]
            weight = (sld[active] - rho) * structure.volume[active]
            return over_sizes(
                lambda i, f: debye_pairs(
                    f * q, xyz, weight, progress=lambda x: report((i + x) / factors.size)
                )
            ), None
        if kernel == 'average':
            notes.append(
                '<A>^2 kernel: exact only for spherically symmetric structures; the result '
                'depends on the centre.'
            )
        spectrum, bin_width = self._spectrum(kernel, structure, sld, active, report, notes)
        return over_sizes(lambda i, f: spectrum.debye(f * q, rho)), bin_width

    def _spectrum(self, kernel, structure, sld, active, progress, notes):
        """The distance spectrum of the *active* points, and the histogram bin it has.

        The cached spectrum is reused when it still applies, a cached histogram
        with a finer bin than the one needed included.
        """
        deuterated = self._params['deuterated']
        bin_width = self._bin_width(self._params) if kernel == 'histogram' else None
        if self._cache is not None:
            c_structure, c_deuterated, c_kernel, c_active, c_bin, spectrum = self._cache
            if (
                c_structure is structure
                and (c_deuterated, c_kernel) == (deuterated, kernel)
                and np.array_equal(c_active, active)
                and (bin_width is None or c_bin <= bin_width)
            ):
                return spectrum, c_bin
        xyz, volume = structure.xyz[active], structure.volume[active]
        b = sld[active] * volume
        if kernel == 'lattice':
            index, shape = self._lattice_index(structure, active)
            grid_b, grid_v = np.zeros(shape), np.zeros(shape)
            np.add.at(grid_b, tuple(index.T), b)
            np.add.at(grid_v, tuple(index.T), volume)
            spectrum = spectrum_lattice(grid_b, grid_v, structure.lattice.step)
        elif kernel == 'histogram':
            if xyz.shape[0] > LARGE_HISTOGRAM:
                self._warn(
                    notes,
                    f'{xyz.shape[0]} scattering points: the histogram pass scales as N^2 and may '
                    "take minutes; kernel='average' suits round aggregates.",
                )
            spectrum = spectrum_histogram(xyz, b, volume, bin_width, progress=progress)
        else:
            spectrum = spectrum_radial(xyz, b, volume, structure.centre)
        self._cache = (structure, deuterated, kernel, active, bin_width, spectrum)
        return spectrum, bin_width

    # ----------------------------------------------------------------- output

    def describe(self) -> None:
        """Print the structure, the q grid and the parameters."""
        self._require_structure().describe()
        contrast = self._scattering_structure().sld - self._params['sld_solvent'] * 1e-6
        print(f'  {"Zero contrast":<18} {int(np.sum(contrast == 0))} points at the current solvent')
        print(
            f'  {"q grid":<18} {self._q.size} points, {self._q[0]:.4g} to {self._q[-1]:.4g} {INVERSE_ANGSTROM}'
        )
        print(f'  {"Parameters":<18} {self._describe_params()}')

    def plot_points(self, **kwargs: Any):
        """Draw the structure's points as they scatter.

        Zero-contrast points use the current solvent, and the ``deuterated``
        hydrogens have the SLD of D.
        """
        kwargs.setdefault('sld_solvent', self._params['sld_solvent'])
        return self._scattering_structure().plot_points(**kwargs)

    def plot_profile(self, *, data=None, log_scale: bool = True, show: bool | None = None):
        """Plot the current profile, optionally over measured *data* (a ``Data1D``)."""
        if self._profile is None:
            raise ValueError('No current profile. Call calculate() first.')
        return self._profile.plot(data=data, log_scale=log_scale, show=show)

    # ------------------------------------------------------------ settings

    def get_settings(self) -> dict[str, Any]:
        """A JSON-serialisable record of the source, grid, parameters and versions.

        A structure no reader made (``from_arrays``, or a copy derived from a
        file) is recorded as ``'in-memory'``, which ``from_settings`` refuses;
        a derived copy also records ``derived_from``, for information only.
        """
        structure = self._require_structure()
        replayable = structure.reader_options is not None
        settings = {
            'source': structure.source if replayable else 'in-memory',
            'basename': os.path.basename(structure.source),
            'source_sha256': structure.source_sha256,
            'reader_options': copy.deepcopy(structure.reader_options),
            'q': self._q.tolist(),
            'q_grid': dict(self._q_args) if self._q_args else None,
            'params': dict(self._params),
            'versions': {
                'sinccity': __version__,
                **{name: _package_version(name) for name in ('numpy', 'periodictable')},
            },
        }
        if not replayable and structure.source != 'in-memory':
            settings['derived_from'] = structure.source
        return settings

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> 'GenericScatteringCalculator':
        """Rebuild a calculator from ``get_settings()``, warning about what changed."""
        if settings['source'] == 'in-memory':
            raise ValueError('These settings describe an in-memory structure; use set_structure().')
        calculator = cls()
        structure = calculator.load_structure(
            settings['source'], **(settings.get('reader_options') or {})
        )
        if structure.source_sha256 != settings['source_sha256']:
            warnings.warn(
                f'{settings["source"]} has changed since the settings were saved.', stacklevel=2
            )
        saved = settings['versions'].get('periodictable')
        if saved and saved != _package_version('periodictable'):
            warnings.warn(
                f'periodictable {saved} was used for the saved settings; scattering lengths '
                'and densities may differ in this version.',
                stacklevel=2,
            )
        if settings.get('q_grid'):
            calculator.set_q_grid(**settings['q_grid'])
        else:
            calculator.set_q(settings['q'])
        # A record without a kernel (0.1.0) replays with the kernel that made it.
        legacy = {'kernel': 'auto' if structure.lattice is not None else 'pairs'}
        calculator.set_params(**{**legacy, **settings['params']})
        return calculator
