# SincCity — Generic Scattering Calculator

The orientationally averaged I(q) of a real-space structure, by the Debye
formula: atoms from a **PDB** or GROMACS **GRO** file, pixels from a
SasView **SLD** file, or the cells of a legacy ASCII **VTK** mesh. It is the
nuclear, 1D part of SasView's *Generic Scattering Calculator*, as a plain
Python package for scripts and notebooks. There are no 2D, magnetic or
polarisation options.

`sinccity` is developed alongside [IQuana](https://github.com/ai4se1dk/IQuana)
but does not depend on it. The two meet through `profile.to_data1d()`, which
returns a sasdata `Data1D` that `IQuana.set_data` (or any other sasdata
consumer) accepts.

## Installation

```bash
pip install -e .                  # numpy and periodictable only
pip install -e ".[plot,sasdata]"  # plus plotly (plots) and sasdata (Data1D output)
```

The required dependencies are `numpy` and `periodictable` (scattering lengths
and densities of atoms). Two extras are optional: `plot` installs `plotly`,
which `plot_points` and `plot_profile` need, and `sasdata` installs `sasdata`,
which `to_data1d` needs. Without them those calls raise an `ImportError` that
names the extra.

## Usage

```python
from sinccity import GenericScatteringCalculator

gsc = GenericScatteringCalculator()
gsc.load_structure('particle.sld')  # .pdb, .sld or .vtk
gsc.plot_points()  # SasView's "Draw Points", in 3D
gsc.set_q_grid(3e-4, 0.3, 30)  # optional: SasView's defaults
gsc.set_params(sld_solvent=6.34)  # optional: 10^-6 Å^-2, D2O here
profile = gsc.calculate()
profile.describe()
gsc.plot_profile()  # or gsc.plot_profile(data=some_data1d)
data = profile.to_data1d()  # a sasdata Data1D, ready for a fit
```

The calculation is the Debye sum over all points:

```text
w_j  = (sld_j - sld_solvent * 1e-6) * volume_j        SLD in Å^-2, volume in Å^3
D(q) = Σ_j Σ_k w_j w_k sin(q r_jk) / (q r_jk)
I(q) = scale * 1e8 * D(q) / V_norm + background       1/cm
```

`sld_solvent` is in 10⁻⁶ Å⁻², like every sasmodels SLD parameter. SasView's
calculator takes Å⁻² instead; a value that looks like Å⁻² triggers a warning.
The structure's own SLDs stay in Å⁻², as the files store them.

**Normalisation.** `V_norm` is the volume of the points that scatter, those
whose SLD differs from the solvent's. For a particle of pixels or cells in a
box of solvent, the curve at `scale=1` then equals sasmodels' form factor at
`scale=1`, so `scale` is a volume fraction. When the solvent exactly matches a
population of points, those points stop counting and the curve jumps by the
volume ratio; for contrast-variation series set
`set_params(normalisation_volume=...)` explicitly (`'auto'` restores the
default). SasView calls the override `total_volume` and, when some points
have zero contrast, also multiplies by the ratio of all points' volume to the
scattering volume. This calculator does not.

**Size polydispersity.** `set_params(size_pd=0.1)` sums blown-up and shrunk
copies of the whole structure, as sasmodels does for `radius_pd`: `size_pd_n`
size factors (35) evenly over ±`size_pd_nsigma` (3) standard deviations,
factors at or below 0 dropped, each weighted by the Gaussian's height there.
A factor f stretches every distance by f and every amplitude by f³, so

```text
I(q) = scale * 1e8 * Σ n_f f⁶ D(f q) / (Σ n_f f³ V_norm) + background
```

with the shares n_f adding up to 1, so `scale` stays a volume fraction. The
lattice, histogram and average kernels evaluate their one distance spectrum
at f·q for every size, which costs only the sums over q; `'pairs'` repeats
the whole pair sum per size and warns. The automatic histogram bin and the
cell-size warning use the largest factor, `1 + size_pd_nsigma * size_pd`.

**Deuteration.** `set_params(deuterated=['CTA'])` gives the hydrogens of
these residue names the SLD of D at `calculate()`; volumes are unchanged,
since D and H have the same natural atomic volume. It needs a structure with
residue names (PDB, GRO or `Structure.from_atoms` with `resnames=`); a name
that is not in the structure, or whose atoms include no hydrogen, is an error.
`structure.deuterated(['CTA'])` returns the same change as a new structure.

## File formats

Positions are in Å and SLDs in Å⁻² in all four (a `.gro` file's nm are converted).

| Format | What is read |
|---|---|
| PDB | `ATOM` records of the first model. `include_hetatm=True` adds `HETATM` records, never waters (`HOH`, `DOD`, `WAT` and the MD names `SOL`, `TIP3`, `TIP4`, `SPC`, `SPCE`, `OPC`). One alternate-location ID for the whole structure, occupancy not applied. Residue names from columns 18–21, where MD programs write four-letter names. Elements as below. Each element gets SasView's SLD and *natural atomic volume* from periodictable; an unknown element is an error. `CONECT` bonds are drawn. A `CRYST1` box is stored unless it is unreadable or a placeholder (a side of 1 Å or less). Residue numbers that are not decimal integers (hybrid-36) are renumbered in order. |
| GRO | All atoms, cut at GROMACS's fixed columns (names may touch numbers; `-ndec` files are read). Velocities are ignored; the box comes from the last line and is used to unwrap the aggregate. Elements as below. |
| SLD | One header line, then `x y z sld`, `x y z sld mx my mz` or `x y z sld mx my mz volume`. Magnetic columns are ignored; six-column magnetic-only files are refused. Without a volume column the points must form a lattice and each pixel's volume is the product of the steps. A sparse subset that skips planes looks like a coarser lattice, so give such files the volume column. |
| VTK | Legacy ASCII `UNSTRUCTURED_GRID` (classic or 5.1 layout) with tetrahedra, voxels, hexahedra, wedges and pyramids. The nuclear SLD is the one single-component `SCALARS` field, per cell or per point (averaged to cells as SasView does). Each cell becomes one point at its volume centroid. Binary files are refused with an export recipe. |

**Selecting residues.** `load_structure(path, residues=['CTA'])` reads only
the atoms of these residue names, from PDB and GRO files. When given, it is
the whole selection: the PDB water and `HETATM` rules do not apply. A `.gro`
file of a solvated system needs it; without it the reader notes how many
residue names it read. Options a format does not have are ignored with a
note. The structure keeps the options it was read with
(`structure.reader_options`), so `get_settings()` records them however the
structure was installed.

**Elements.** For PDB and GRO files each residue name gets its elements from
the first source that has them:

1. `masses={'CTA': [14.007, 12.011, ...]}`: the atom masses in u in topology
   order. Each mass gives the element of nearest mass, H to U and D, within
   0.01 u. Hydrogen mass repartitioning (H at 3.024 u) or united atoms (CH₂
   at 14.027 u) are no element's and an error: pass `elements=` for them.
2. `elements={'CTA': ['N', 'C', ...]}`: the element symbols in order.
3. The PDB element column, when it is filled for every atom of the residue.
4. A guess from the atom names, with a note. PDB files use SasView's column
   rule (` CA ` is carbon, `CA` starting in column 13 calcium). GRO files have
   no column convention: leading digits are dropped, then the first letter is
   the element, except that a one-atom residue whose name is a two-letter
   element (`NA`, `CL`) is that element. So `CA` in a protein is carbon and a
   `CA` ion calcium. The note lists every two-letter guess, since those are
   the likeliest to be wrong.

Masses and element lists are matched by the atom's position in its residue
(a run of atoms with one residue number and name), so the atom names of the structure
and the topology may differ. `structure.element_sources` records the source
per residue name. `structure.with_symbols(symbols)` returns a copy with
corrected elements, and `Structure.from_atoms(xyz, symbols, resids=,
resnames=, names=)` builds atoms that were read elsewhere.

**Periodic boxes.** GROMACS writes every position back into the box, which
can split molecules and the aggregate. With `unwrap='auto'` (the default) a
`.gro` file is unwrapped: each molecule is made whole around its first atom,
then moved by whole box lengths next to the aggregate's centre, the circular
mean of the molecule centres. A note counts the atoms moved. It works for one
aggregate smaller than half the box. A `.gro` file whose box is zero (no
periodic box) is read as it is, with a note. PDB files are unwrapped only with
`unwrap=True`, because many programs write a placeholder `CRYST1`; a missing
or placeholder box is then an error. Triclinic boxes cannot be unwrapped
(`unwrap=False` reads such files without the box).

The readers follow SasView's (`sas.sascalc.calculator.sas_gen`, BSD-3-Clause,
Copyright (c) 2009-2026, SasView Developers) with one deviation: the VTK cell
geometry is recomputed, because SasView's volumes for hexahedra, wedges and
pyramids are wrong. The GRO reader and the residue options come from
coords2sans.

## Things to know

- **PDB structures in a solvent are only qualitative.** The natural atomic
  volumes use bulk elemental densities, liquid H₂, N₂ and O₂ for H, N and O.
  A typical residue gets about 310 Å³ instead of about 133 Å³, so in D₂O a
  protein comes out about 8× too intense at q → 0. There is no hydration
  shell and no H/D exchange either. `calculate()` warns with the actual
  volume ratio. In vacuum (`sld_solvent=0`) the result is right. For atoms,
  `scale` is not a volume fraction.
- **Points stand in for cells.** Pixels and VTK cells are treated as points,
  which degrades as q × cell size approaches 1. `calculate()` warns above
  that, with the cell size taken as the lattice step or the cell diameter
  divided by √3.
- **Kernels.** `set_params(kernel=...)` picks how D(q) is summed; the
  profile's `method` records it.

  | `kernel` | How | Exact? | Time |
  |---|---|---|---|
  | `'lattice'` | FFT autocorrelation of the pixel grid | yes | well under a second for 10⁵ pixels |
  | `'histogram'` | pairs counted by distance, then one sum per q over the occupied distances | to about 1e-4 | about 2 s for 11 000 atoms, N² |
  | `'average'` | ⟨A⟩²: each point's distance from the bounding-box centre | only for spherically symmetric structures | linear in N |
  | `'pairs'` | every pair at every q | to the last digit | 2–3 s for 4 000 points, 25 s for 8 000 |

  `'auto'` (the default) is `'lattice'` for pixels on a lattice and
  `'histogram'` for everything else. The histogram bin is `'auto'`:
  `min(0.01 Å, 1e-3 / (q_max · f_max))`, with f_max the largest size factor
  (1 without polydispersity), which keeps the rounding phase error below
  5e-4 rad; `histogram_bin=` sets it, and `profile.histogram_bin` records it.
  A cached spectrum with a finer bin is reused.
  `'average'` reproduces SasView's `is_avg=True` and suits round aggregates
  above about 10⁵ atoms. `calculate(progress=...)` can report progress or
  stop the pair passes of `'histogram'` and `'pairs'`.
- **Contrast series are cheap.** The lattice, histogram and average kernels
  keep the scattering lengths and the volumes apart in their distance
  spectrum, so the calculator keeps the last one and a new `sld_solvent`,
  `scale`, `background` or `normalisation_volume` costs only the sums over
  q. A new structure, deuteration, kernel, a finer bin or a change of which
  points match the solvent repeats the pair pass.
- **The curve is theory.** `profile.to_data1d()` returns a `Data1D` with no
  dI and no dQ, with `qmin`, `qmax` and an empty `mask` set. A fit of it
  should state its weighting and resolution explicitly; pass
  `to_data1d(dI=...)` to attach uncertainties.
- **Reproducibility.** `gsc.get_settings()` records the file path and
  SHA-256, the q grid, the parameters and the periodictable version.
  `GenericScatteringCalculator.from_settings(settings)` rebuilds the
  calculator and warns if the file or the version changed.
- **Console output.** Progress lines go through the `sinccity` logger;
  `sinccity.set_verbosity('quiet')` silences them.

## Tests

```bash
pytest tests
```

## Licence

BSD 3-Clause, like IQuana. The test structures in `tests/data/` are
SasView test data, BSD 3-Clause, Copyright (c) 2009-2026, SasView Developers.
