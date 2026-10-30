"""Readers for PDB, GRO, SLD and legacy VTK structure files.

All readers return a :class:`Structure` with positions in Å and SLD in Å⁻²,
raise :class:`StructureError` on unusable input, and record the file's
SHA-256. Behaviour follows SasView's Generic Scattering Calculator readers
(``sas.sascalc.calculator.sas_gen``, BSD-3-Clause, Copyright (c) 2009-2026,
SasView Developers), with the deviations documented in the package README: the
VTK cell geometry is recomputed, because SasView's volumes for hexahedra,
wedges and pyramids are wrong. The GRO reader and the residue options follow
coords2sans.
"""

import dataclasses
import hashlib
import os
from collections import Counter

import numpy as np

from .elements import guess_pdb, guess_stem, sld_volume
from .model import Structure, StructureError, detect_lattice
from .molecules import residue_runs

#: Residue names of water, which the PDB reader drops unless ``residues`` is given.
WATER_RESIDUES = ('HOH', 'DOD', 'WAT', 'SOL', 'TIP3', 'TIP3P', 'TIP4P', 'SPC', 'SPCE', 'OPC')


def _hash_file(path: str) -> str | None:
    """SHA-256 of a file, or None when it cannot be read."""
    try:
        digest = hashlib.sha256()
        with open(path, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def load(path, *, include_hetatm: bool = False, residues=None) -> Structure:
    """Read a structure file, choosing the reader by extension.

    Options a format does not have are ignored with a note.

    Args:
        path: A ``.pdb``, ``.gro``, ``.sld`` (or ``.txt``) or ``.vtk`` file.
        include_hetatm: PDB only; also read ``HETATM`` records (never waters).
        residues: PDB and GRO; the residue names to read, and only those.
    """
    extension = os.path.splitext(os.fspath(path))[1].lower()
    readers = {'.pdb': read_pdb, '.gro': read_gro, '.sld': read_sld, '.txt': read_sld}
    reader = readers.get(extension, read_vtk if extension == '.vtk' else None)
    if reader is None:
        raise StructureError(
            f'Unknown structure file type {extension!r} for {path}. Use .pdb, .gro, .sld or .vtk.'
        )
    options = {'include_hetatm': include_hetatm, 'residues': residues}
    accepted = reader.__kwdefaults__ or {}
    structure = reader(path, **{k: v for k, v in options.items() if k in accepted})
    ignored = [k for k, v in options.items() if k not in accepted and v not in (None, False)]
    if ignored:
        note = f'Ignored {", ".join(ignored)}: not an option for {extension} files.'
        structure = dataclasses.replace(structure, notes=(*structure.notes, note))
    return structure


def _source(path) -> tuple[str, str | None]:
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(path):
        raise StructureError(f'Structure file not found: {path}')
    return path, _hash_file(path)


def _names(residues) -> list[str] | None:
    """The ``residues`` option as a JSON-ready list."""
    if residues is None:
        return None
    return [residues] if isinstance(residues, str) else [str(name) for name in residues]


def _box(lengths, triclinic: bool, notes: list[str]) -> tuple[float, float, float] | None:
    """The rectangular box, or None with a note for a triclinic one."""
    if triclinic:
        notes.append('The periodic box is triclinic; it is not stored.')
        return None
    return tuple(float(length) for length in lengths)


def _atoms(path, sha, format_, notes, options, *, symbols, **fields) -> Structure:
    """The atom structure the PDB and GRO readers return."""
    try:
        sld, volume = sld_volume(symbols)
    except StructureError as exc:
        raise StructureError(f'{path}: {exc}') from exc
    return Structure(
        'atoms',
        sld=sld,
        volume=volume,
        symbols=symbols,
        reader_options=options,
        source=path,
        source_sha256=sha,
        format=format_,
        notes=tuple(notes),
        **fields,
    )


# ============================================================================
# PDB
# ============================================================================


def read_pdb(path, *, include_hetatm: bool = False, residues=None) -> Structure:
    """Read the atoms of a PDB file.

    Policies: the first ``MODEL`` only; one alternate-location ID for the whole
    structure (blank plus the first non-blank ID seen); occupancy not applied;
    waters never read; elements from columns 77-78, else from the atom name.
    Residue names are read from columns 18-21, where MD programs write
    four-letter names. The ``CRYST1`` box is stored unless it is the 1 Å
    placeholder. Each policy that drops or ignores something adds a note.

    Args:
        path: The PDB file.
        include_hetatm: Also read ``HETATM`` records.
        residues: The residue names to read. When given, it is the whole
            selection: the water and ``HETATM`` rules do not apply.
    """
    path, sha = _source(path)
    residues = _names(residues)
    with open(path, encoding='utf-8', errors='replace') as handle:
        lines = handle.read().splitlines()

    xyz, symbols, serials, conect = [], [], [], []
    resids, resnames, names = [], [], []
    dropped_het: Counter = Counter()
    dropped_water = dropped_altloc = models = 0
    partial_occupancy = False
    altloc_kept = cryst = None
    for number, line in enumerate(lines, start=1):
        record = line[:6]
        if record == 'MODEL ':
            models += 1
        if models > 1:
            continue
        if record == 'CRYST1':
            cryst = line
            continue
        if record == 'CONECT':
            fields = [line[i : i + 5].strip() for i in range(6, 31, 5)]
            conect.append([int(f) for f in fields if f.isdigit()])
            continue
        if record not in ('ATOM  ', 'HETATM'):
            continue
        residue = line[17:21].strip()
        if residues is not None:
            if residue not in residues:
                continue
        elif residue in WATER_RESIDUES:
            dropped_water += 1
            continue
        elif record == 'HETATM' and not include_hetatm:
            dropped_het[residue] += 1
            continue
        altloc = line[16:17].strip()
        if altloc:
            altloc_kept = altloc_kept or altloc
            if altloc != altloc_kept:
                dropped_altloc += 1
                continue
        try:
            position = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            symbol = line[76:78].strip() or guess_pdb(line)
            occupancy = line[54:60].strip()
            partial_occupancy |= bool(occupancy) and float(occupancy) < 1.0
            resid = int(line[22:26].strip() or 0)
        except (ValueError, IndexError) as exc:
            raise StructureError(f'{path}, line {number}: cannot read the atom record.') from exc
        xyz.append(position)
        symbols.append(symbol[0].upper() + symbol[1:].lower())
        resids.append(resid)
        resnames.append(residue)
        names.append(line[12:16].strip())
        serial = line[6:11].strip()
        serials.append(int(serial) if serial.isdigit() else None)

    if not xyz:
        if residues is not None:
            raise StructureError(f'{path} holds no atom of the residues {", ".join(residues)}.')
        raise StructureError(f'{path} contains no atom records.')

    index_of = {s: i for i, s in enumerate(serials) if s is not None}
    bonds = [
        (index_of[group[0]], index_of[other])
        for group in conect
        if group and group[0] in index_of
        for other in group[1:]
        if other in index_of
    ]

    notes = []
    if models > 1:
        notes.append(f'Read the first of {models} models.')
    if dropped_het:
        listed = ', '.join(f'{name} x{count}' for name, count in sorted(dropped_het.items()))
        notes.append(
            f'Skipped {sum(dropped_het.values())} HETATM records ({listed}); '
            'pass include_hetatm=True to read them.'
        )
    if dropped_water:
        notes.append(f'Skipped {dropped_water} water atoms.')
    if dropped_altloc:
        notes.append(f'Kept alternate location {altloc_kept!r}; skipped {dropped_altloc} atoms.')
    if partial_occupancy:
        notes.append('Some atoms have occupancy below 1; occupancy is not applied.')

    box = None
    if cryst is not None:
        try:
            lengths = [float(cryst[i : i + 9]) for i in (6, 15, 24)]
            angles = [float(cryst[i : i + 7]) for i in (33, 40, 47)]
        except ValueError as exc:
            raise StructureError(f'{path}: cannot read the CRYST1 record.') from exc
        if lengths != [1.0, 1.0, 1.0]:  # many programs write a 1 Ang placeholder
            box = _box(lengths, angles != [90.0, 90.0, 90.0], notes)

    return _atoms(
        path,
        sha,
        'pdb',
        notes,
        {'include_hetatm': include_hetatm, 'residues': residues},
        xyz=np.array(xyz),
        symbols=symbols,
        resids=resids,
        resnames=resnames,
        names=names,
        bonds=np.array(bonds, dtype=int).reshape(-1, 2),
        box=box,
    )


# ============================================================================
# GRO
# ============================================================================


def read_gro(path, *, residues=None) -> Structure:
    """Read the atoms of a GROMACS ``.gro`` file.

    Fields are cut at their fixed columns, since names may touch numbers. As
    GROMACS defines it, the width of a position field is the distance between
    the decimal points of the first atom's x and y, which covers files written
    with ``-ndec``. Positions are converted from nm to Å; velocities are
    ignored. Elements are guessed from the atom names (see
    :func:`sinccity.elements.guess_stem`).

    Args:
        path: The ``.gro`` file.
        residues: The residue names to read. A solvated system needs it: there
            is no water rule, and the reader notes when it reads more than one
            residue name without a selection.
    """
    path, sha = _source(path)
    residues = _names(residues)
    with open(path, encoding='utf-8', errors='replace') as handle:
        lines = handle.read().splitlines()
    try:
        atoms = lines[2 : 2 + int(lines[1])]
        box = [float(value) * 10 for value in lines[2 + len(atoms)].split()]
        dot = atoms[0].index('.', 20)
        width = atoms[0].index('.', dot + 1) - dot
        columns = [slice(20 + k * width, 20 + (k + 1) * width) for k in range(3)]
        resids = np.array([int(line[0:5]) for line in atoms])
        resnames = np.array([line[5:10].strip() for line in atoms])
        names = np.array([line[10:15].strip() for line in atoms])
        xyz = np.array([[line[c] for c in columns] for line in atoms], dtype=float) * 10
    except (IndexError, ValueError) as exc:
        raise StructureError(f'{path} is not a readable .gro file ({exc}).') from exc
    if len(box) not in (3, 9):
        raise StructureError(f'{path}: the last line must hold 3 or 9 box values.')

    notes = []
    if residues is None:
        found = sorted(set(resnames.tolist()))
        if len(found) > 1:
            notes.append(
                f'Read {len(found)} residue names ({", ".join(found)}); pass residues= to '
                'select the aggregate from its solvent.'
            )
    else:
        kept = np.isin(resnames, residues)
        if not kept.any():
            raise StructureError(f'{path} holds no atom of the residues {", ".join(residues)}.')
        resids, resnames, names, xyz = resids[kept], resnames[kept], names[kept], xyz[kept]

    box = _box(box[:3], any(box[3:]), notes)
    _, sizes = residue_runs(resids)
    single = np.repeat(sizes == 1, sizes)
    symbols = [guess_stem(name, one) for name, one in zip(names.tolist(), single)]
    return _atoms(
        path,
        sha,
        'gro',
        notes,
        {'residues': residues},
        xyz=xyz,
        symbols=symbols,
        resids=resids,
        resnames=resnames,
        names=names,
        box=box,
    )


# ============================================================================
# SLD
# ============================================================================


def read_sld(path) -> Structure:
    """Read a SasView SLD text file.

    The first line is a header and is always skipped. Columns: ``x y z sld``
    (4), ``x y z sld mx my mz`` (7) or ``x y z sld mx my mz volume`` (8);
    six columns are magnetic-only and refused. Without a volume column the
    points must form a lattice, and each pixel's volume is the product of the
    steps. A sparse subset that skips planes looks like a coarser lattice, so
    such files need the volume column.
    """
    path, sha = _source(path)
    try:
        data = np.loadtxt(path, skiprows=1, ndmin=2)
    except ValueError as exc:
        raise StructureError(f'{path} is not a numeric SLD file: {exc}') from exc
    columns = data.shape[1]
    if columns == 6:
        raise StructureError(
            f'{path} holds magnetic SLD only (6 columns); a nuclear SLD is needed.'
        )
    if columns not in (4, 7, 8) or data.shape[0] == 0:
        raise StructureError(f'{path} must have 4, 7 or 8 columns after one header line.')
    if not np.all(np.isfinite(data)):
        raise StructureError(f'{path} contains non-finite values.')
    xyz = data[:, :3]
    if np.unique(xyz, axis=0).shape[0] != xyz.shape[0]:
        raise StructureError(f'{path} lists the same position more than once.')

    notes = []
    if columns >= 7 and np.any(data[:, 4:7] != 0):
        notes.append('Magnetic SLD columns are ignored.')
    lattice = detect_lattice(xyz)
    if columns == 8:
        volume = data[:, 7]
    elif lattice is not None and all(step > 0 for step in lattice.step):
        volume = np.full(xyz.shape[0], float(np.prod(lattice.step)))
    else:
        raise StructureError(
            f'{path}: without a volume column the points must form a 3D lattice with at '
            'least two positions on each axis. Add the pixel volume as column 8.'
        )
    if np.any(volume <= 0):
        raise StructureError(f'{path}: every pixel volume must be positive.')
    return Structure(
        'pixels',
        xyz,
        data[:, 3],
        volume,
        lattice=lattice,
        source=path,
        source_sha256=sha,
        format='sld',
        reader_options={},
        notes=tuple(notes),
    )


# ============================================================================
# VTK (legacy ASCII, UNSTRUCTURED_GRID)
# ============================================================================

# Outward-oriented faces per VTK cell type, in VTK's vertex order. Voxels
# (type 11) are reordered to the hexahedron order before use.
_FACES = {
    10: ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    12: ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    13: ((0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)),
    14: ((0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)),
}
_VERTICES = {10: 4, 11: 8, 12: 8, 13: 6, 14: 5}
_VOXEL_TO_HEXAHEDRON = (0, 1, 3, 2, 4, 5, 7, 6)
_SECTIONS = {
    'POINTS',
    'CELLS',
    'CELL_TYPES',
    'CELL_DATA',
    'POINT_DATA',
    'SCALARS',
    'VECTORS',
    'METADATA',
}


def cell_geometry(vertices: np.ndarray, cell_type: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed volumes and volume centroids for cells of one VTK type.

    Each face is split into triangles (quadrilaterals are fanned from their
    face centre) and each triangle is joined to the cell's vertex mean. The
    signed tetrahedron volumes sum to the cell volume by the divergence
    theorem, convex or not; their volume-weighted centroids give the cell's
    centroid.

    Args:
        vertices: Shape (n_cells, n_vertices, 3), in VTK vertex order.
        cell_type: 10, 11, 12, 13 or 14.
    """
    if cell_type == 11:
        vertices, cell_type = vertices[:, _VOXEL_TO_HEXAHEDRON], 12
    reference = vertices.mean(axis=1)
    volume = np.zeros(vertices.shape[0])
    moment = np.zeros((vertices.shape[0], 3))
    for face in _FACES[cell_type]:
        corners = vertices[:, face]
        if len(face) == 3:
            triangles = [(corners[:, 0], corners[:, 1], corners[:, 2])]
        else:
            centre = corners.mean(axis=1)
            triangles = [(corners[:, i], corners[:, (i + 1) % 4], centre) for i in range(4)]
        for a, b, c in triangles:
            tet = np.einsum('ij,ij->i', a - reference, np.cross(b - reference, c - reference)) / 6
            volume += tet
            moment += tet[:, None] * (a + b + c + reference) / 4
    with np.errstate(invalid='ignore', divide='ignore'):
        centroid = moment / volume[:, None]
    return volume, centroid


class _Tokens:
    """A cursor over the whitespace-separated tokens of a VTK body."""

    def __init__(self, text: str, path: str) -> None:
        self.items = text.split()
        self.pos = 0
        self.path = path

    def more(self) -> bool:
        return self.pos < len(self.items)

    def peek(self) -> str:
        return self.items[self.pos] if self.more() else ''

    def next(self) -> str:
        if not self.more():
            raise StructureError(f'{self.path}: unexpected end of file.')
        self.pos += 1
        return self.items[self.pos - 1]

    def numbers(self, count: int, dtype=float) -> np.ndarray:
        if self.pos + count > len(self.items):
            raise StructureError(f'{self.path}: expected {count} more values.')
        try:
            values = np.array(self.items[self.pos : self.pos + count], dtype=float)
        except ValueError as exc:
            raise StructureError(f'{self.path}: non-numeric value in a data block.') from exc
        self.pos += count
        return values.astype(dtype)


def _read_cells(tokens: _Tokens, n_cells_field: int, size: int) -> list[np.ndarray]:
    """Cell connectivity in either the legacy (<= 4.2) or the 5.1 layout."""
    if tokens.peek().upper() == 'OFFSETS':  # 5.1: CELLS n_offsets n_connectivity
        tokens.next()
        tokens.next()  # data type
        offsets = tokens.numbers(n_cells_field, int)
        if tokens.next().upper() != 'CONNECTIVITY':
            raise StructureError(f'{tokens.path}: CONNECTIVITY expected after OFFSETS.')
        tokens.next()
        connectivity = tokens.numbers(size, int)
        return [connectivity[offsets[i] : offsets[i + 1]] for i in range(n_cells_field - 1)]
    flat = tokens.numbers(size, int)
    cells, i = [], 0
    while i < flat.size:
        count = flat[i]
        cells.append(flat[i + 1 : i + 1 + count])
        i += count + 1
    if len(cells) != n_cells_field:
        raise StructureError(
            f'{tokens.path}: CELLS declares {n_cells_field} cells, found {len(cells)}.'
        )
    return cells


def read_vtk(path) -> Structure:
    """Read a legacy ASCII VTK ``UNSTRUCTURED_GRID`` as one point per cell.

    Supports cell types 10 (tetrahedron), 11 (voxel), 12 (hexahedron), 13
    (wedge) and 14 (pyramid), in the classic and the 5.1 ``OFFSETS`` layout.
    The nuclear SLD is the single one-component ``SCALARS`` field, given per
    cell or per point (point values are averaged to cells with SasView's
    inverse-square weights about the cell's vertex mean). Three-component
    fields are magnetic and ignored. Each cell becomes a point at its volume
    centroid with the cell's volume and SLD. Units are taken as Å and Å⁻².
    """
    path, sha = _source(path)
    with open(path, encoding='utf-8', errors='replace') as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if len(lines) < 4 or not lines[0].lower().startswith('# vtk datafile version'):
        raise StructureError(f'{path} is not a legacy VTK file.')
    if lines[2].upper() != 'ASCII':
        raise StructureError(
            f'{path} is binary. Re-export it as ASCII, for example '
            "meshio.write(path, mesh, binary=False), or ParaView's Save Data with ASCII."
        )
    if lines[3].upper().split() != ['DATASET', 'UNSTRUCTURED_GRID']:
        raise StructureError(
            f'{path}: only DATASET UNSTRUCTURED_GRID is supported, got {lines[3]!r}.'
        )

    tokens = _Tokens('\n'.join(lines[4:]), path)
    points = cells = types = None
    association, fields = None, []
    while tokens.more():
        word = tokens.next().upper()
        if word == 'POINTS':
            n = int(tokens.next())
            tokens.next()
            points = tokens.numbers(3 * n).reshape(n, 3)
        elif word == 'CELLS':
            first, size = int(tokens.next()), int(tokens.next())
            cells = _read_cells(tokens, first, size)
        elif word == 'CELL_TYPES':
            types = tokens.numbers(int(tokens.next()), int)
        elif word in ('CELL_DATA', 'POINT_DATA'):
            association = (word, int(tokens.next()))
        elif word in ('SCALARS', 'VECTORS'):
            if association is None:
                raise StructureError(f'{path}: {word} outside CELL_DATA or POINT_DATA.')
            name = tokens.next()
            tokens.next()  # data type
            components = 3
            if word == 'SCALARS':
                components = int(tokens.next()) if tokens.peek().isdigit() else 1
                if tokens.peek().upper() == 'LOOKUP_TABLE':
                    tokens.next()
                    tokens.next()
            values = tokens.numbers(association[1] * components).reshape(association[1], components)
            fields.append((name, association[0], values))
        elif word == 'METADATA':
            while tokens.more() and tokens.peek().upper() not in _SECTIONS:
                tokens.next()
        else:
            raise StructureError(f'{path}: unsupported VTK section {word!r}.')

    if points is None or cells is None or types is None:
        raise StructureError(f'{path} needs POINTS, CELLS and CELL_TYPES sections.')
    if len(types) != len(cells):
        raise StructureError(f'{path}: {len(cells)} cells but {len(types)} cell types.')

    nuclear = [f for f in fields if f[2].shape[1] == 1]
    magnetic = [f for f in fields if f[2].shape[1] == 3]
    if not nuclear:
        what = 'only magnetic data' if magnetic else 'no SLD data'
        raise StructureError(f'{path} has {what}; one single-component SCALARS field is needed.')
    if len(nuclear) > 1:
        names = ', '.join(f[0] for f in nuclear)
        raise StructureError(f'{path} has several candidate SLD fields ({names}); keep only one.')
    name, where, values = nuclear[0]
    values = values[:, 0]

    n_cells = len(cells)
    centroid = np.empty((n_cells, 3))
    volume = np.empty(n_cells)
    diameter = np.empty(n_cells)
    sld = values if where == 'CELL_DATA' else np.empty(n_cells)
    if where == 'CELL_DATA' and values.size != n_cells:
        raise StructureError(f'{path}: CELL_DATA has {values.size} values for {n_cells} cells.')
    if where == 'POINT_DATA' and values.size != len(points):
        raise StructureError(
            f'{path}: POINT_DATA has {values.size} values for {len(points)} points.'
        )
    for cell_type in np.unique(types):
        if cell_type not in _VERTICES:
            raise StructureError(f'{path}: unsupported cell type {cell_type}; use types 10-14.')
        members = np.flatnonzero(types == cell_type)
        connectivity = [cells[i] for i in members]
        if any(c.size != _VERTICES[cell_type] for c in connectivity):
            raise StructureError(
                f'{path}: a type-{cell_type} cell has the wrong number of vertices.'
            )
        index = np.array(connectivity)
        if index.min() < 0 or index.max() >= len(points):
            raise StructureError(f'{path}: a cell refers to a point that does not exist.')
        vertices = points[index]
        volume[members], centroid[members] = cell_geometry(vertices, int(cell_type))
        spans = vertices[:, :, None, :] - vertices[:, None, :, :]
        diameter[members] = np.sqrt((spans**2).sum(axis=-1)).max(axis=(1, 2))
        if where == 'POINT_DATA':
            distance2 = ((vertices - vertices.mean(axis=1, keepdims=True)) ** 2).sum(axis=-1)
            weight = 1.0 / distance2
            sld[members] = (weight * values[index]).sum(axis=1) / weight.sum(axis=1)

    floor = 1e-12 * np.median(np.abs(volume))
    bad = np.flatnonzero(volume <= floor)
    if bad.size:
        kind = 'inverted' if volume[bad[0]] < 0 else 'degenerate'
        raise StructureError(f'{path}: cell {bad[0]} is {kind} (volume {volume[bad[0]]:.3g}).')

    notes = []
    if magnetic:
        notes.append('Magnetic SLD fields are ignored.')
    if where == 'POINT_DATA':
        notes.append(f'SLD field {name!r} was averaged from points to cells.')
    return Structure(
        'cells',
        centroid,
        sld,
        volume,
        cell_diameter=diameter,
        source=path,
        source_sha256=sha,
        format='vtk',
        reader_options={},
        notes=tuple(notes),
    )
