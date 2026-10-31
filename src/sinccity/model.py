"""The real-space structure a scattering calculation runs on."""

import copy
import dataclasses
import html
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._console import ANGSTROM, CUBIC_ANGSTROM, SLD_UNIT

#: Partial specific volume of a typical protein, cm³/g.
PROTEIN_SPECIFIC_VOLUME = 0.73

#: Relative tolerance for positions on a lattice, as a fraction of the step.
LATTICE_TOLERANCE = 1e-4


class StructureError(ValueError):
    """A structure file or array set cannot be used for a calculation."""


def residue_names(names, option: str) -> list[str]:
    """One residue name or an iterable of them, as a list.

    Raises:
        ValueError: If *names* is neither, naming the *option* it was given for.
    """
    if isinstance(names, str):
        return [names]
    try:
        values = list(names)
    except TypeError:
        values = None
    if values is None or not all(isinstance(name, str) for name in values):
        raise ValueError(f'{option} must be residue names, got {names!r}.')
    return [str(name) for name in values]


def _owned(values: Any, dtype: Any = float) -> np.ndarray:
    """A read-only copy, so later edits to the caller's array cannot reach in."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Lattice:
    """Points on ``origin + step * indices``. A step of 0 marks a single plane."""

    origin: tuple[float, float, float]
    step: tuple[float, float, float]
    indices: np.ndarray  # (N, 3) int, each axis starting at 0

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(n) + 1 for n in self.indices.max(axis=0))


def detect_lattice(xyz: np.ndarray) -> Lattice | None:
    """Return the lattice every point sits on, or None.

    The step on each axis is the smallest gap between distinct coordinates.
    Every coordinate must then be an integer number of steps from the lowest,
    within ``LATTICE_TOLERANCE`` of a step, which tolerates files written with
    limited precision and sparse subsets of a lattice.
    """
    origin, steps, indices = [], [], []
    for values in np.asarray(xyz, dtype=float).T:
        unique = np.unique(values)
        extent = unique[-1] - unique[0]
        gaps = np.diff(unique)
        gaps = gaps[gaps > 1e-9 * max(extent, 1.0)]
        if gaps.size == 0:
            origin.append(float(unique[0]))
            steps.append(0.0)
            indices.append(np.zeros(values.size, dtype=int))
            continue
        step = float(gaps.min())
        position = (values - unique[0]) / step
        index = np.rint(position)
        if np.max(np.abs(position - index)) > LATTICE_TOLERANCE:
            return None
        origin.append(float(unique[0]))
        steps.append(step)
        indices.append(index.astype(int))
    return Lattice(tuple(origin), tuple(steps), _owned(np.column_stack(indices), int))


@dataclass(frozen=True, eq=False)
class RgResult:
    """A radius of gyration, or the reason there is none."""

    value: float  # Å, nan when undefined
    rg_squared: float  # Å², signed; can be negative for signed contrast
    reason: str = ''


@dataclass(frozen=True, eq=False)
class Structure:
    """Point scatterers: atoms, lattice pixels or mesh-cell centroids.

    Arrays are read-only copies owned by the structure. That protects against
    accidental mutation; it is not a guarantee (``setflags`` can undo it).

    Attributes:
        kind: ``'atoms'``, ``'pixels'`` or ``'cells'``.
        xyz: Positions, shape (N, 3), Å.
        sld: Nuclear scattering length density per point, Å⁻², as the files
            store it (the calculator's ``sld_solvent`` is in 10⁻⁶ Å⁻²).
        volume: Volume per point, Å³, all positive.
        symbols: Element symbols, atoms only.
        bonds: Index pairs from PDB ``CONECT`` records, atoms only.
        cell_diameter: Largest vertex-to-vertex distance per cell, cells only.
        lattice: The lattice the points sit on, pixels only.
        resids, resnames, names: Residue number, residue name and atom name
            per atom, from PDB and GRO files; None otherwise.
        element_sources: Where the elements of each residue name came from:
            ``'masses'``, ``'elements'``, ``'pdb'``, ``'names'`` (guessed) or
            ``'given'`` (``from_atoms``). Empty without residue names.
        box: The rectangular periodic box the file declared, Å; None when it
            declared none, a placeholder or a triclinic one.
        reader_options: The options the reader was called with, every key
            filled, so the file plus this dict rebuilds the structure. None
            for structures no reader made.
        source: Absolute path of the file, or ``'in-memory'``.
        source_sha256: Hash of the file as read.
        format: ``'pdb'``, ``'gro'``, ``'sld'``, ``'vtk'`` or ``'arrays'``.
        notes: What the reader dropped, approximated or assumed.
    """

    kind: str
    xyz: np.ndarray
    sld: np.ndarray
    volume: np.ndarray
    symbols: np.ndarray | None = None
    bonds: np.ndarray | None = None
    cell_diameter: np.ndarray | None = None
    lattice: Lattice | None = None
    resids: np.ndarray | None = None
    resnames: np.ndarray | None = None
    names: np.ndarray | None = None
    element_sources: dict[str, str] = field(default_factory=dict)
    box: tuple[float, float, float] | None = None
    reader_options: dict[str, Any] | None = None
    source: str = 'in-memory'
    source_sha256: str | None = None
    format: str = 'arrays'
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in ('atoms', 'pixels', 'cells'):
            raise StructureError(f"kind must be 'atoms', 'pixels' or 'cells', got {self.kind!r}.")
        xyz = np.asarray(self.xyz, dtype=float)
        n = xyz.shape[0] if xyz.ndim == 2 else 0
        if xyz.ndim != 2 or xyz.shape[1] != 3 or n == 0:
            raise StructureError(f'Positions must have shape (N, 3) with N > 0, got {xyz.shape}.')
        sld = np.asarray(self.sld, dtype=float)
        volume = np.asarray(self.volume, dtype=float)
        for name, values in (('sld', sld), ('volume', volume)):
            if values.shape != (n,):
                raise StructureError(
                    f'{name} must have one value per point ({n}), got {values.shape}.'
                )
        if not (
            np.all(np.isfinite(xyz)) and np.all(np.isfinite(sld)) and np.all(np.isfinite(volume))
        ):
            raise StructureError('Positions, SLD and volumes must all be finite.')
        if np.any(volume <= 0):
            raise StructureError(
                f'Every volume must be positive; {int(np.sum(volume <= 0))} are not.'
            )
        set_ = object.__setattr__  # frozen dataclass: fields are written once, here
        set_(self, 'xyz', _owned(xyz))
        set_(self, 'sld', _owned(sld))
        set_(self, 'volume', _owned(volume))
        set_(self, 'notes', tuple(self.notes))
        has_resids, has_resnames, has_names = (
            getattr(self, name) is not None for name in ('resids', 'resnames', 'names')
        )
        if (has_resids or has_resnames or has_names) and not (
            has_resids and has_resnames and self.kind == 'atoms'
        ):
            raise StructureError(
                'resids and resnames come together, names only with them, and only for atoms.'
            )
        for name, dtype in (('symbols', str), ('resids', int), ('resnames', str), ('names', str)):
            values = getattr(self, name)
            if values is not None:
                if len(values) != n:
                    raise StructureError(f'{name} must have one entry per point.')
                set_(self, name, _owned(values, dtype))
        set_(self, 'element_sources', dict(self.element_sources))
        if self.box is not None:
            set_(self, 'box', tuple(float(length) for length in self.box))
        if self.reader_options is not None:
            set_(self, 'reader_options', copy.deepcopy(dict(self.reader_options)))
        if self.bonds is not None:
            bonds = np.asarray(self.bonds, dtype=int).reshape(-1, 2)
            if bonds.size and (bonds.min() < 0 or bonds.max() >= n):
                raise StructureError('Bond indices must refer to existing points.')
            set_(self, 'bonds', _owned(np.unique(np.sort(bonds, axis=1), axis=0), int))
        if self.cell_diameter is not None:
            set_(self, 'cell_diameter', _owned(self.cell_diameter))

    @classmethod
    def from_arrays(cls, xyz, sld, volume, *, symbols=None, bonds=None) -> 'Structure':
        """Build a structure from arrays: positions (Å), SLD (Å⁻²) and volumes (Å³).

        Points with element *symbols* are atoms; otherwise they are pixels,
        and the lattice they sit on, if any, is detected.
        """
        if symbols is not None:
            return cls('atoms', xyz, sld, volume, symbols=symbols, bonds=bonds)
        xyz = np.asarray(xyz, dtype=float)
        lattice = detect_lattice(xyz) if xyz.ndim == 2 and xyz.shape[-1] == 3 else None
        return cls('pixels', xyz, sld, volume, lattice=lattice)

    @classmethod
    def from_atoms(
        cls, xyz, symbols, *, resids=None, resnames=None, names=None, bonds=None
    ) -> 'Structure':
        """Build atoms whose SLD and volume come from their elements, as the readers do.

        This is how atoms read by another program come in without a file.
        Positions in Å.
        """
        from .elements import sld_volume

        symbols = [str(symbol).capitalize() for symbol in symbols]
        sld, volume = sld_volume(symbols)
        sources = {} if resnames is None else dict.fromkeys(sorted(set(resnames)), 'given')
        return cls(
            'atoms',
            xyz,
            sld,
            volume,
            symbols=symbols,
            bonds=bonds,
            resids=resids,
            resnames=resnames,
            names=names,
            element_sources=sources,
        )

    def deuterated(self, residues) -> 'Structure':
        """A copy in which the hydrogens of these residue names have the SLD of D.

        Volumes are unchanged: D and H have the same natural atomic volume. The
        copy keeps the source, but no reader made it, so its ``reader_options``
        are None.

        Raises:
            StructureError: If the structure has no residue names, or a name is
                not in it or has no hydrogens.
        """
        from .elements import constants

        residues = residue_names(residues, 'residues')
        if self.resnames is None or self.symbols is None:
            raise StructureError('Only atoms with residue names can be deuterated.')
        unknown = sorted(set(residues) - set(self.resnames.tolist()))
        if unknown:
            raise StructureError(f'Deuterated residues not in the structure: {", ".join(unknown)}.')
        hydrogens = (self.symbols == 'H') & np.isin(self.resnames, residues)
        without = sorted(set(residues) - set(self.resnames[hydrogens].tolist()))
        if without:
            raise StructureError(f'No hydrogens to deuterate in {", ".join(without)}.')
        sld = self.sld.copy()
        sld[hydrogens] = constants('D')[0]
        return dataclasses.replace(self, sld=sld, reader_options=None)

    def with_symbols(self, symbols) -> 'Structure':
        """A copy with corrected elements, and SLD and volumes recomputed from them.

        The residue names whose elements change are recorded as
        ``'elements'`` in ``element_sources``. The copy keeps the source, but no
        reader made it, so its ``reader_options`` are None.
        """
        from .elements import sld_volume

        if self.symbols is None:
            raise StructureError('Only atoms have elements.')
        symbols = np.array([str(symbol).capitalize() for symbol in symbols])
        if symbols.shape != self.symbols.shape:
            raise StructureError(f'{symbols.size} elements given for {self.n_points} atoms.')
        sld, volume = sld_volume(symbols)
        sources = dict(self.element_sources)
        if self.resnames is not None:
            changed = np.unique(self.resnames[symbols != self.symbols]).tolist()
            sources.update(dict.fromkeys(changed, 'elements'))
        return dataclasses.replace(
            self,
            symbols=symbols,
            sld=sld,
            volume=volume,
            element_sources=sources,
            reader_options=None,
        )

    # ------------------------------------------------------------------ facts

    @property
    def n_points(self) -> int:
        return int(self.sld.size)

    @property
    def total_volume(self) -> float:
        """Sum of all point volumes, Å³ (V_all)."""
        return float(self.volume.sum())

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    @property
    def centre(self) -> np.ndarray:
        """The middle of the bounding box, Å, as SasView's ``transform_center`` takes it."""
        low, high = self.bounding_box
        return (low + high) / 2

    @property
    def masses(self) -> np.ndarray:
        """Atomic masses in g/mol (atoms only)."""
        if self.symbols is None:
            raise StructureError('Masses are only defined for atomic structures.')
        from periodictable import elements

        return np.array([elements.symbol(s).mass for s in self.symbols])

    def radius_of_gyration(self, weights: str = 'sld', *, sld_solvent: float = 0.0) -> RgResult:
        """Radius of gyration, as a diagnostic that never raises.

        Args:
            weights: ``'sld'`` weights each point by its contrast times volume
                (for atoms in vacuum, the scattering length); ``'mass'`` uses
                atomic masses and the centre of mass (atoms only).
            sld_solvent: Solvent SLD in 10⁻⁶ Å⁻², for ``'sld'`` weights.
        """
        if weights == 'mass':
            if self.symbols is None:
                return RgResult(float('nan'), float('nan'), 'mass needs element symbols')
            w = self.masses
        elif weights == 'sld':
            w = (self.sld - sld_solvent * 1e-6) * self.volume
        else:
            raise ValueError(f"weights must be 'sld' or 'mass', got {weights!r}.")
        total = w.sum()
        if total == 0:
            return RgResult(float('nan'), float('nan'), 'zero net contrast')
        centre = (w[:, None] * self.xyz).sum(axis=0) / total
        rg2 = float(np.sum(w * np.sum((self.xyz - centre) ** 2, axis=1)) / total)
        if rg2 < 0:
            return RgResult(float('nan'), rg2, 'negative Rg² from signed contrast')
        return RgResult(float(np.sqrt(rg2)), rg2)

    def summary(self) -> dict[str, Any]:
        """The facts ``describe()`` prints, as a plain dict."""
        low, high = self.bounding_box
        info: dict[str, Any] = {
            'source': self.source,
            'format': self.format,
            'kind': self.kind,
            'n_points': self.n_points,
            'bounding_box': [low.tolist(), high.tolist()],
            'total_volume': self.total_volume,
            'sld_mean': float(self.sld.mean() * 1e6),
            'sld_range': [float(self.sld.min() * 1e6), float(self.sld.max() * 1e6)],
            'n_zero_sld': int(np.sum(self.sld == 0)),
            'rg_sld': self.radius_of_gyration('sld').value,
            'notes': list(self.notes),
        }
        if self.kind == 'atoms':
            info['rg_mass'] = self.radius_of_gyration('mass').value
        if self.lattice is not None:
            info['lattice_step'] = list(self.lattice.step)
        if self.cell_diameter is not None:
            info['max_cell_diameter'] = float(self.cell_diameter.max())
        if self.resids is not None:
            from .molecules import residue_runs

            info['n_residues'] = int(residue_runs(self.resids, self.resnames)[0].size)
            info['residue_atoms'] = dict(sorted(Counter(self.resnames.tolist()).items()))
        if self.element_sources:
            info['element_sources'] = dict(self.element_sources)
        if self.box is not None:
            info['box'] = list(self.box)
        return info

    def _lines(self) -> list[tuple[str, str]]:
        info = self.summary()
        low, high = info['bounding_box']
        rows = [
            ('Source', f'{self.source} ({self.format})'),
            ('Points', f'{self.n_points} {self.kind}'),
            (
                'Bounding box',
                f'{np.round(low, 3).tolist()} to {np.round(high, 3).tolist()} {ANGSTROM}',
            ),
            ('Total volume', f'{self.total_volume:.6g} {CUBIC_ANGSTROM}'),
            (
                'SLD',
                f'mean {info["sld_mean"]:.4g}, range {info["sld_range"][0]:.4g} to '
                f'{info["sld_range"][1]:.4g} {SLD_UNIT}; {info["n_zero_sld"]} points at 0',
            ),
            ('Rg (SLD, vacuum)', f'{info["rg_sld"]:.4g} {ANGSTROM}'),
        ]
        if 'rg_mass' in info:
            rows.append(('Rg (mass)', f'{info["rg_mass"]:.4g} {ANGSTROM}'))
        if self.lattice is not None:
            step = max(self.lattice.step)
            rows.append(('Lattice step', f'{list(self.lattice.step)} {ANGSTROM}'))
            if step > 0:
                rows.append(('Sampling limit', f'q < pi/step = {np.pi / step:.4g} (SasView)'))
        if self.cell_diameter is not None:
            rows.append(('Max cell diameter', f'{info["max_cell_diameter"]:.4g} {ANGSTROM}'))
        if 'n_residues' in info:
            atoms = ', '.join(f'{name} x{count}' for name, count in info['residue_atoms'].items())
            rows.append(('Residues', f'{info["n_residues"]}; atoms per name: {atoms}'))
        if self.element_sources:
            sources = ', '.join(f'{name} {source}' for name, source in self.element_sources.items())
            rows.append(('Elements from', sources))
        if self.box is not None:
            rows.append(('Box', f'{np.round(self.box, 3).tolist()} {ANGSTROM}'))
        rows.extend(('Note', note) for note in self.notes)
        return rows

    def describe(self) -> None:
        """Print a summary of the structure."""
        for label, value in self._lines():
            print(f'  {label:<18} {value}')

    def _repr_html_(self) -> str:
        cells = ''.join(
            f'<tr><th style="text-align:left">{html.escape(k)}</th><td>{html.escape(v)}</td></tr>'
            for k, v in self._lines()
        )
        return f'<table>{cells}</table>'

    def __repr__(self) -> str:
        return f'<Structure {self.kind}: {self.n_points} points from {self.source}>'

    def plot_points(self, **kwargs: Any):
        """Draw the points in 3D; see ``sinccity.plotting.plot_points``."""
        from .plotting import plot_points

        return plot_points(self, **kwargs)
