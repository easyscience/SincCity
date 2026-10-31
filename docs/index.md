# SincCity

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
pip install sinccity                  # numpy and periodictable only
pip install "sinccity[plot,sasdata]"  # plus plotly (plots) and sasdata (Data1D output)
```

The required dependencies are `numpy` and `periodictable` (scattering lengths
and densities of atoms). Two extras are optional: `plot` installs `plotly`,
which `plot_points` and `plot_profile` need, and `sasdata` installs `sasdata`,
which `to_data1d` needs. Without them those calls raise an `ImportError` that
names the extra.

## Quick start

```python
from sinccity import GenericScatteringCalculator

gsc = GenericScatteringCalculator()
gsc.load_structure('particle.sld')  # .pdb, .gro, .sld or .vtk
gsc.plot_points()  # SasView's "Draw Points", in 3D
gsc.set_q_grid(3e-4, 0.3, 30)  # optional: SasView's defaults
gsc.set_params(sld_solvent=6.34)  # optional: 10^-6 Å^-2, D2O here
profile = gsc.calculate()
profile.describe()
gsc.plot_profile()  # or gsc.plot_profile(data=some_data1d)
data = profile.to_data1d()  # a sasdata Data1D, ready for a fit
```

An aggregate from a molecular-dynamics run, in several contrasts:

```python
gsc.load_structure(
    'CTA178-eq5.gro',
    residues=['CTA'],  # the micelle, not its solvent
    masses={'CTA': cta_masses},  # from the topology, in atom order
)
gsc.set_params(scale=0.01, size_pd=0.1)
profiles = gsc.calculate_series([
    {'label': 'h-CTA in D2O', 'params': {'sld_solvent': 6.36}},
    {'label': 'h-CTA in H2O', 'params': {'sld_solvent': -0.56}},
    {'label': 'd-CTA in H2O', 'params': {'sld_solvent': -0.56, 'deuterated': ['CTA']}},
])
```

## Where to go next

- [Structures](guide/structures.md): the file formats, residue selection,
  elements and periodic boxes.
- [Calculation](guide/calculation.md): the formula, units, normalisation,
  size polydispersity, deuteration and the kernels.
- [Series and reproducibility](guide/series.md): contrast series, settings
  records and worker processes.
- [Coming from coords2sans](guide/coords2sans.md): the names that changed.
- [API reference](api/calculator.md): every public class and function.

## Licence

BSD 3-Clause, like IQuana. The readers follow SasView's
(`sas.sascalc.calculator.sas_gen`, BSD-3-Clause, Copyright (c) 2009-2026,
SasView Developers), and the test structures are SasView test data under the
same licence.
