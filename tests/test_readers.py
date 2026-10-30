"""The PDB, SLD and VTK readers."""

import warnings

import numpy as np
import pytest
from support import C60, CUBE_VTK, gro_text, quiet_calculator

from sinccity import StructureError, load, read_gro, read_pdb, read_sld, read_vtk
from sinccity.readers import cell_geometry

# ============================================================================
# PDB
# ============================================================================


def pdb_line(serial, name, residue, x, element, record='ATOM', altloc=' ', occupancy=1.0, resseq=1):
    return (
        f'{record:<6}{serial:>5} {name:<4}{altloc}{residue:<4}A{resseq:>4}    '
        f'{x:8.3f}{0.0:8.3f}{0.0:8.3f}{occupancy:6.2f}{0.0:6.2f}          {element:>2}'
    )


class TestPdb:
    def test_c60(self):
        s = read_pdb(C60)
        assert (s.kind, s.n_points) == ('atoms', 60)
        assert np.allclose(s.xyz[0], [-0.733, -1.008, 3.326])
        assert set(s.symbols) == {'C'}
        assert s.bonds.tolist() == [[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]
        assert s.radius_of_gyration().value == pytest.approx(3.552, abs=1e-3)
        assert s.radius_of_gyration('mass').value == pytest.approx(3.552, abs=1e-3)
        from periodictable import nsf

        assert s.sld[0] == pytest.approx(nsf.neutron_sld('C')[0] * 1e-6)

    def test_policies(self, tmp_path):
        lines = [
            pdb_line(1, 'N', 'ALA', 0.0, 'N'),
            pdb_line(2, 'CA', 'ALA', 1.0, 'C', altloc='B'),
            pdb_line(3, 'CA', 'ALA', 1.1, 'C', altloc='A'),
            pdb_line(5, 'FE', 'HEM', 2.0, 'FE', record='HETATM', resseq=2),
            pdb_line(6, 'SE', 'MSE', 3.0, 'SE', record='HETATM', resseq=3, occupancy=0.5),
            pdb_line(7, 'O', 'HOH', 4.0, 'O', record='HETATM', resseq=4),
            'CONECT    1    2    5',
            'CONECT    5    6',
            'MODEL        2',
            pdb_line(8, 'N', 'ALA', 9.0, 'N'),
        ]
        path = tmp_path / 'policies.pdb'
        path.write_text('MODEL        1\n' + '\n'.join(lines) + '\n')

        default = read_pdb(path)
        assert default.symbols.tolist() == ['N', 'C']  # altloc B came first and is kept
        assert default.bonds.tolist() == [[0, 1]]  # bonds to skipped atoms dropped
        notes = ' '.join(default.notes)
        assert 'first of 2 models' in notes and 'HEM x1' in notes and 'MSE x1' in notes
        assert '1 water' in notes and "alternate location 'B'" in notes

        with_het = read_pdb(path, include_hetatm=True)
        assert with_het.symbols.tolist() == ['N', 'C', 'Fe', 'Se']  # case-normalised, no water
        assert with_het.bonds.tolist() == [[0, 1], [0, 2], [2, 3]]
        assert 'occupancy below 1' in ' '.join(with_het.notes)

    def test_unknown_element_is_an_error(self, tmp_path):
        path = tmp_path / 'bad.pdb'
        path.write_text(pdb_line(1, 'XX', 'UNK', 0.0, 'XX') + '\n')
        with pytest.raises(StructureError, match='Xx'):
            read_pdb(path)

    def test_missing_file(self):
        with pytest.raises(StructureError, match='not found'):
            read_pdb('missing.pdb')

    def test_residue_data_four_letter_names_and_md_waters(self, tmp_path):
        lines = [
            pdb_line(1, 'N1', 'CTAB', 0.0, 'N', resseq=1),
            pdb_line(2, 'C1', 'CTAB', 1.0, 'C', resseq=1),
            pdb_line(3, 'BR', 'BR', 5.0, 'BR', record='HETATM', resseq=2),
            pdb_line(4, 'OW', 'SOL', 4.0, 'O', resseq=3),
            pdb_line(5, 'OH2', 'TIP3', 6.0, 'O', resseq=4),
        ]
        path = tmp_path / 'md.pdb'
        path.write_text('\n'.join(lines) + '\n')
        s = read_pdb(path)
        assert s.resnames.tolist() == ['CTAB', 'CTAB'] and s.resids.tolist() == [1, 1]
        assert s.names.tolist() == ['N1', 'C1']
        assert 'Skipped 2 water atoms.' in s.notes and 'BR x1' in ' '.join(s.notes)
        assert s.reader_options == {
            'include_hetatm': False,
            'residues': None,
            'masses': None,
            'elements': None,
            'unwrap': 'auto',
        }

        selected = read_pdb(path, residues=['BR', 'SOL'])  # the whole selection
        assert selected.resnames.tolist() == ['BR', 'SOL'] and selected.notes == ()
        assert selected.reader_options['residues'] == ['BR', 'SOL']
        assert read_pdb(path, residues='CTAB').n_points == 2
        with pytest.raises(StructureError, match='no atom of the residues CTA'):
            read_pdb(path, residues=['CTA'])

    def test_cryst1_box(self, tmp_path):
        atom = pdb_line(1, 'C', 'ALA', 0.0, 'C')
        for cryst, box in (
            (
                'CRYST1   40.000   50.000   60.000  90.00  90.00  90.00 P 1           1',
                (40, 50, 60),
            ),
            ('CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1', None),
            ('CRYST1   40.000   40.000   40.000  60.00  60.00  90.00 P 1           1', None),
        ):
            path = tmp_path / 'box.pdb'
            path.write_text(f'{cryst}\n{atom}\n')
            assert read_pdb(path).box == box
        assert 'triclinic' in ' '.join(read_pdb(path).notes)


# ============================================================================
# GRO
# ============================================================================


class TestGro:
    ATOMS = [
        (1, 'CTA', 'N1', 0.1, 0.2, 0.3),
        (1, 'CTA', 'C21', 0.2, 0.2, 0.3),
        (1, 'CTA', 'CA', 0.3, 0.2, 0.3),
        (2, 'CL', 'CL', 0.5, 0.5, 0.5),
        (3, 'SOL', 'OW', 1.0, 1.0, 1.0),
    ]

    def test_columns_units_box_and_guesses(self, tmp_path):
        path = tmp_path / 'a.gro'
        path.write_text(gro_text(self.ATOMS))
        s = read_gro(path)
        assert (s.format, s.n_points, s.box) == ('gro', 5, (50.0, 60.0, 70.0))
        assert np.allclose(s.xyz[0], [1.0, 2.0, 3.0])
        assert s.symbols.tolist() == ['N', 'C', 'C', 'Cl', 'O']  # CA in CTA is carbon
        assert s.resids.tolist() == [1, 1, 1, 2, 3]
        assert 'pass residues=' in ' '.join(s.notes)
        assert s.reader_options == {
            'residues': None,
            'masses': None,
            'elements': None,
            'unwrap': 'auto',
        }
        assert s.element_sources == {'CL': 'names', 'CTA': 'names', 'SOL': 'names'}
        assert 'CL -> Cl in CL' in ' '.join(s.notes)
        selected = read_gro(path, residues=['CTA'], masses={'CTA': [14.007, 12.011, 12.011]})
        assert selected.n_points == 3 and selected.notes == ()
        with pytest.raises(StructureError, match='no atom of the residues NA'):
            read_gro(path, residues=['NA'])

    def test_ndec_widths_and_touching_fields(self, tmp_path):
        atoms = [(12345, 'TIP3P', 'HW123', 1.23456789, -0.5, 2.0), (12345, 'TIP3P', 'OW', 0, 0, 0)]
        path = tmp_path / 'wide.gro'
        path.write_text(gro_text(atoms, ndec=6))
        s = read_gro(path, unwrap=False)
        assert s.resnames.tolist() == ['TIP3P', 'TIP3P'] and s.names.tolist() == ['HW123', 'OW']
        assert s.resids.tolist() == [12345, 12345]
        assert np.allclose(s.xyz[0], [12.34568, -5.0, 20.0])

    def test_triclinic_box_and_bad_files(self, tmp_path):
        path = tmp_path / 'tri.gro'
        triclinic = '   5.0   5.0   5.0   0.0   0.0   2.5   0.0   0.0   0.0'
        path.write_text(gro_text(self.ATOMS[:3], box=triclinic))
        with pytest.raises(StructureError, match='pass unwrap=False'):
            read_gro(path)
        s = read_gro(path, unwrap=False)
        assert s.box is None and 'triclinic' in ' '.join(s.notes)
        path.write_text('title\n  3\n')
        with pytest.raises(StructureError, match='not a readable .gro file'):
            read_gro(path)


class TestUnwrapping:
    # Two CTA molecules of a micelle in a 5 nm box. The second one is split by
    # the x face: its second atom was written back at the other side.
    SPLIT = [
        (1, 'CTA', 'C1', 4.8, 2.0, 2.0),
        (1, 'CTA', 'C2', 4.9, 2.0, 2.0),
        (2, 'CTA', 'C1', 4.95, 2.1, 2.0),
        (2, 'CTA', 'C2', 0.05, 2.1, 2.0),
    ]

    def test_split_aggregate_is_rejoined(self, tmp_path):
        path = tmp_path / 'split.gro'
        path.write_text(gro_text(self.SPLIT, box='   5.0   5.0   5.0'))
        s = read_gro(path)
        assert np.allclose(s.xyz[:, 0], [48.0, 49.0, 49.5, 50.5])
        assert 'moved 1 atoms' in ' '.join(s.notes)
        assert np.allclose(read_gro(path, unwrap=False).xyz[3, 0], 0.5)

    def test_aggregate_across_the_face_and_a_whole_one(self, tmp_path):
        across = [(1, 'CTA', 'C1', 4.9, 2.0, 2.0), (2, 'CTA', 'C1', 0.1, 2.0, 2.0)]
        path = tmp_path / 'across.gro'
        path.write_text(gro_text(across, box='   5.0   5.0   5.0'))
        s = read_gro(path)
        assert abs(s.xyz[1, 0] - s.xyz[0, 0]) == pytest.approx(2.0)  # neighbours again
        path.write_text(gro_text(self.SPLIT[:2], box='   5.0   5.0   5.0'))
        whole = read_gro(path)
        assert np.allclose(whole.xyz[:, 0], [48.0, 49.0])
        assert not any('Unwrapped' in note for note in whole.notes)

    def test_pdb_unwraps_only_on_request(self, tmp_path):
        atoms = [
            pdb_line(i, 'C', 'CTA', x, 'C', resseq=resid)
            for i, (resid, x) in enumerate([(1, 48.0), (1, 49.0), (2, 49.5), (2, 0.5)], 1)
        ]
        cryst = 'CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1'
        path = tmp_path / 'box.pdb'
        path.write_text('\n'.join([cryst, *atoms]) + '\n')
        assert read_pdb(path).xyz[3, 0] == pytest.approx(0.5)
        assert read_pdb(path, unwrap=True).xyz[3, 0] == pytest.approx(50.5)
        path.write_text('\n'.join(atoms) + '\n')
        with pytest.raises(StructureError, match='needs a periodic box'):
            read_pdb(path, unwrap=True)
        with pytest.raises(ValueError, match='unwrap must be'):
            read_pdb(path, unwrap='yes')


def test_load_notes_options_a_format_does_not_have(tmp_path):
    path = tmp_path / 'a.gro'
    path.write_text(gro_text(TestGro.ATOMS))
    s = load(path, include_hetatm=True, residues=['CTA'])
    assert s.n_points == 3 and any('Ignored include_hetatm' in note for note in s.notes)
    vtk = load(CUBE_VTK, residues=['CTA'])
    assert vtk.reader_options == {} and any('Ignored residues' in note for note in vtk.notes)


# ============================================================================
# SLD
# ============================================================================


class TestSld:
    def lattice(self, n=3, step=(2.0, 3.0, 4.0)):
        index = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing='ij'), -1).reshape(-1, 3)
        return index * np.array(step)

    def write(self, path, rows):
        np.savetxt(path, rows, header='x y z sld ...', comments='')
        return path

    def test_four_seven_and_eight_columns(self, tmp_path):
        xyz = self.lattice()
        sld = np.full((len(xyz), 1), 2e-6)
        four = read_sld(self.write(tmp_path / 'a.sld', np.hstack([xyz, sld])))
        assert np.allclose(four.volume, 24.0) and four.lattice.step == (2.0, 3.0, 4.0)
        magnetic = np.hstack([xyz, sld, np.ones((len(xyz), 3)) * 1e-7])
        seven = read_sld(self.write(tmp_path / 'b.sld', magnetic))
        assert 'Magnetic SLD columns are ignored.' in seven.notes
        eight = read_sld(
            self.write(tmp_path / 'c.sld', np.hstack([magnetic, np.full((len(xyz), 1), 5.0)]))
        )
        assert np.allclose(eight.volume, 5.0)

    def test_single_row_and_plane_with_volumes(self, tmp_path):
        one = read_sld(self.write(tmp_path / 'one.sld', [[0, 0, 0, 1e-6, 0, 0, 0, 7.0]]))
        assert one.n_points == 1 and one.volume[0] == 7.0
        plane = np.hstack(
            [self.lattice()[:9], np.full((9, 1), 1e-6), np.zeros((9, 3)), np.ones((9, 1))]
        )
        assert read_sld(self.write(tmp_path / 'plane.sld', plane)).n_points == 9

    def test_shuffled_imprecise_and_sparse_lattices(self, tmp_path):
        xyz = self.lattice() + 1e-7  # written with limited precision
        rng = np.random.default_rng(0)
        rows = np.hstack([xyz, np.full((len(xyz), 1), 1e-6)])[rng.permutation(len(xyz))]
        shuffled = read_sld(self.write(tmp_path / 'shuffled.sld', rows))
        assert np.allclose(shuffled.lattice.step, (2.0, 3.0, 4.0))
        sparse = read_sld(self.write(tmp_path / 'sparse.sld', rows[::2]))
        assert sparse.lattice is not None

    def test_refusals(self, tmp_path):
        irregular = np.random.default_rng(0).random((10, 3)) * 10
        with pytest.raises(StructureError, match='column 8'):
            read_sld(
                self.write(tmp_path / 'irregular.sld', np.hstack([irregular, np.ones((10, 1))]))
            )
        with pytest.raises(StructureError, match='magnetic SLD only'):
            read_sld(self.write(tmp_path / 'mag.sld', np.zeros((4, 6))))
        duplicated = np.array([[0, 0, 0, 1e-6], [0, 0, 0, 1e-6]])
        with pytest.raises(StructureError, match='same position'):
            read_sld(self.write(tmp_path / 'dup.sld', duplicated))


# ============================================================================
# VTK
# ============================================================================

UNIT_CELLS = {
    # type: (vertices in VTK order, volume, centroid)
    10: ([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6, [0.25, 0.25, 0.25]),
    11: (
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        1.0,
        [0.5, 0.5, 0.5],
    ),
    12: (
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        1.0,
        [0.5, 0.5, 0.5],
    ),
    13: (
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1]],
        0.5,
        [1 / 3, 1 / 3, 0.5],
    ),
    14: ([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], 1 / 3, [0.5, 0.5, 0.25]),
}


def vtk_text(points, cells, types, scalars, where='CELL_DATA', layout='legacy', extra=''):
    points = np.asarray(points, dtype=float)
    out = ['# vtk DataFile Version 3.0', 'test', 'ASCII', 'DATASET UNSTRUCTURED_GRID']
    out.append(f'POINTS {len(points)} float')
    out += [' '.join(map(str, p)) for p in points]
    if layout == 'legacy':
        out.append(f'CELLS {len(cells)} {sum(len(c) + 1 for c in cells)}')
        out += [' '.join(map(str, [len(c), *c])) for c in cells]
    else:
        offsets = np.cumsum([0] + [len(c) for c in cells])
        out.append(f'CELLS {len(offsets)} {offsets[-1]}')
        out += ['OFFSETS vtktypeint64', ' '.join(map(str, offsets)), 'CONNECTIVITY vtktypeint64']
        out.append(' '.join(str(i) for c in cells for i in c))
    out.append(f'CELL_TYPES {len(types)}')
    out += [str(t) for t in types]
    out.append(f'{where} {len(scalars)}')
    out += ['SCALARS nuclear float 1', 'LOOKUP_TABLE default', ' '.join(map(str, scalars))]
    return '\n'.join(out) + '\n' + extra


class TestVtk:
    def test_sasview_cube(self):
        s = read_vtk(CUBE_VTK)
        assert (s.kind, s.n_points) == ('cells', 5)
        assert s.volume.sum() == pytest.approx(8.0)
        assert np.allclose(s.sld, 1.0)
        assert np.allclose(s.xyz[0], 0.0)
        assert 'Magnetic SLD fields are ignored.' in s.notes

    @pytest.mark.parametrize('cell_type', sorted(UNIT_CELLS))
    def test_unit_cells_translated_and_rotated(self, cell_type):
        vertices, volume, centroid = UNIT_CELLS[cell_type]
        vertices = np.array(vertices, dtype=float)
        theta = 0.4
        rotation = np.array(
            [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
        )
        for moved, centre in (
            (vertices + 10, np.array(centroid) + 10),
            (vertices @ rotation.T, rotation @ centroid),
        ):
            v, c = cell_geometry(moved[None], cell_type)
            assert v[0] == pytest.approx(volume) and np.allclose(c[0], centre)

    def test_non_convex_and_inverted_cells(self):
        # A pyramid on a dart-shaped (non-convex) base of area 1: V = area * height / 3.
        dart = np.array([[0, 0, 0], [2, 0, 0], [0.5, 0.5, 0], [0, 2, 0], [0.3, 0.3, 1.0]])
        volume, _ = cell_geometry(dart[None], 14)
        assert volume[0] == pytest.approx(1 / 3)
        inverted = np.array(UNIT_CELLS[10][0], dtype=float)[[0, 2, 1, 3]]
        assert cell_geometry(inverted[None], 10)[0][0] < 0

    def test_layouts_mixed_cells_and_point_data(self, tmp_path):
        tet, _, _ = UNIT_CELLS[10]
        hexa, _, _ = UNIT_CELLS[12]
        points = np.vstack([np.array(hexa) + [5, 0, 0], tet])
        cells = [list(range(8)), [8, 9, 10, 11]]
        values = np.arange(12, dtype=float)
        structures = []
        for layout in ('legacy', '5.1'):
            path = tmp_path / f'{layout}.vtk'
            path.write_text(vtk_text(points, cells, [12, 10], values, 'POINT_DATA', layout))
            structures.append(read_vtk(path))
        assert np.allclose(structures[0].xyz, structures[1].xyz)
        assert np.allclose(structures[0].sld, structures[1].sld)
        assert np.allclose(structures[0].volume, [1.0, 1 / 6])
        shifted = tmp_path / 'shifted.vtk'
        shifted.write_text(vtk_text(points + 100, cells, [12, 10], values, 'POINT_DATA'))
        assert np.allclose(read_vtk(shifted).sld, structures[0].sld)  # translation invariant

    def test_refusals(self, tmp_path):
        tet = UNIT_CELLS[10][0]
        two = vtk_text(
            tet,
            [[0, 1, 2, 3]],
            [10],
            [1.0],
            extra='SCALARS other float 1\nLOOKUP_TABLE default\n2\n',
        )
        (tmp_path / 'two.vtk').write_text(two)
        with pytest.raises(StructureError, match='several candidate SLD fields'):
            read_vtk(tmp_path / 'two.vtk')
        flat = vtk_text([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2, 3]], [10], [1.0])
        (tmp_path / 'flat.vtk').write_text(flat)
        with pytest.raises(StructureError, match='degenerate'):
            read_vtk(tmp_path / 'flat.vtk')
        binary = vtk_text(tet, [[0, 1, 2, 3]], [10], [1.0]).replace('ASCII', 'BINARY')
        (tmp_path / 'binary.vtk').write_text(binary)
        with pytest.raises(StructureError, match='binary=False'):
            read_vtk(tmp_path / 'binary.vtk')

    def test_cell_size_check(self):
        gsc = quiet_calculator(CUBE_VTK)  # cell size 2*sqrt(2)/sqrt(3) = 1.63 Ang
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            gsc.calculate()  # q_max 0.3: q * size = 0.49
        gsc.set_q_grid(qmax=1.0)
        with pytest.warns(UserWarning, match='treating each cell as a point'):
            gsc.calculate()

    def test_mesh_refinement_trend(self, tmp_path):
        """A voxelised sphere split into tetrahedra approaches the sphere at low q."""
        errors = []
        q = np.array([0.02, 0.05])
        for step in (4.0, 2.0):
            radius = 10.0
            n = int(radius / step) + 1
            axis = np.arange(-n, n) * step
            corners = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
            corners = corners[((corners + step / 2) ** 2).sum(1) <= radius**2]
            hexa = np.array(UNIT_CELLS[12][0], dtype=float) * step
            six = (
                (0, 1, 2, 6),
                (0, 2, 3, 6),
                (0, 3, 7, 6),
                (0, 7, 4, 6),
                (0, 4, 5, 6),
                (0, 5, 1, 6),
            )
            points, cells = [], []
            for corner in corners:
                base = len(points)
                points.extend(hexa + corner)
                cells.extend([base + i for i in tet] for tet in six)
            path = tmp_path / f'sphere{step}.vtk'
            path.write_text(vtk_text(points, cells, [10] * len(cells), [1e-6] * len(cells)))
            gsc = quiet_calculator(path)
            gsc.set_q(q)
            intensity = gsc.calculate().intensity
            volume = gsc.structure.volume.sum()
            x = q * (3 * volume / (4 * np.pi)) ** (1 / 3)
            sphere = 1e8 * (1e-6) ** 2 * volume * (3 * (np.sin(x) - x * np.cos(x)) / x**3) ** 2
            errors.append(np.max(np.abs(intensity / sphere - 1)))
        assert errors[1] < errors[0]
