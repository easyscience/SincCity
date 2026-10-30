# SincCity — Generic Scattering Calculator

The orientationally averaged I(q) of a real-space structure, by the exact
Debye formula: atoms from a **PDB** file, pixels from a SasView **SLD** file,
or the cells of a legacy ASCII **VTK** mesh. It is the nuclear, 1D part of
SasView's *Generic Scattering Calculator*, as a plain Python package for
scripts and notebooks. There are no 2D, magnetic or polarisation options.

`sinccity` is developed alongside [IQuana](https://github.com/ai4se1dk/IQuana)
but does not depend on it. The two meet through `profile.to_data1d()`, which
returns a sasdata `Data1D` that `IQuana.set_data` (or any other sasdata
consumer) accepts.

## Installation

```bash
pip install -e sinccity
```

The pixi environment of the repository installs it automatically. The
dependencies are `numpy`, `plotly`, `periodictable` (scattering lengths and
densities for PDB atoms) and `sasdata` (the `Data1D` output).

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

The calculation is the exact Debye sum over all points:

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

## File formats

Positions are in Å and SLDs in Å⁻² in all three.

| Format | What is read |
|---|---|
| PDB | `ATOM` records of the first model. `include_hetatm=True` adds `HETATM` records, never waters. One alternate-location ID for the whole structure, occupancy not applied. Elements from columns 77–78 (any case), else from the atom name. Each element gets SasView's SLD and *natural atomic volume* from periodictable; an unknown element is an error. `CONECT` bonds are drawn. |
| SLD | One header line, then `x y z sld`, `x y z sld mx my mz` or `x y z sld mx my mz volume`. Magnetic columns are ignored; six-column magnetic-only files are refused. Without a volume column the points must form a lattice and each pixel's volume is the product of the steps. A sparse subset that skips planes looks like a coarser lattice, so give such files the volume column. |
| VTK | Legacy ASCII `UNSTRUCTURED_GRID` (classic or 5.1 layout) with tetrahedra, voxels, hexahedra, wedges and pyramids. The nuclear SLD is the one single-component `SCALARS` field, per cell or per point (averaged to cells as SasView does). Each cell becomes one point at its volume centroid. Binary files are refused with an export recipe. |

The readers follow SasView's (`sas.sascalc.calculator.sas_gen`, BSD-3-Clause,
Copyright (c) 2009-2026, SasView Developers) with one deviation: the VTK cell
geometry is recomputed, because SasView's volumes for hexahedra, wedges and
pyramids are wrong.

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
- **Speed.** Pixels on a lattice use an FFT autocorrelation, which gives the
  same sum exactly and takes well under a second for 10⁵ pixels. Atoms and
  cells use the pair sum, which scales as N²: about 0.1 s for 1 000 points,
  2–3 s for 4 000 and 25 s for 8 000. `calculate(progress=...)` can report
  progress or stop a long run.
- **The curve is theory.** `profile.to_data1d()` returns a `Data1D` with no
  dI and no dQ, with `qmin`, `qmax` and an empty `mask` set. A fit of it
  should state its weighting and resolution explicitly; pass
  `to_data1d(dI=...)` to attach uncertainties.
- **Reproducibility.** `gsc.get_settings()` records the file path and
  SHA-256, the q grid, the parameters and the periodictable version.
  `GenericScatteringCalculator.from_settings(settings)` rebuilds the
  calculator and warns if the file or the version changed.
- **Console output.** Progress lines go through the `gsc` logger;
  `gsc.set_verbosity('quiet')` silences them.

The SANS-fitter user guide has a worked example of fitting a computed curve,
and `examples/structure_calculator_example.py` and
`notebooks/structure_calculator.ipynb` in that repository are complete
walkthroughs.

## Tests

```bash
pytest tests
```

## Licence

BSD 3-Clause, like IQuana. The test structures in `tests/data/` are
SasView test data, BSD 3-Clause, Copyright (c) 2009-2026, SasView Developers.
