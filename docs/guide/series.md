# Series and reproducibility

## Contrast series are cheap

The lattice, histogram and average kernels keep the scattering lengths and the
volumes apart in their distance spectrum, so the calculator keeps the last one
and a new `sld_solvent`, `scale`, `background` or `normalisation_volume` costs
only the sums over q. A new structure, deuteration, kernel, a finer bin or a
change of which points match the solvent repeats the pair pass.

## `calculate_series`

`gsc.calculate_series(series)` computes one profile per entry, each a `label`
and `set_params` keys that apply to that entry only:

```python
profiles = gsc.calculate_series([
    {'label': 'h-CTA in D2O', 'params': {'sld_solvent': 6.36}},
    {'label': 'h-CTA in H2O', 'params': {'sld_solvent': -0.56}},
    {'label': 'd-CTA in H2O', 'params': {'sld_solvent': -0.56, 'deuterated': ['CTA']}},
])
```

- All entries are checked before anything runs; a bad entry is a
  `ValueError` that names it.
- Entries that share a deuteration state and kernel share one pair pass.
- The profiles come back in the order given, each with its `label`.
- The parameters are left as they were, and `gsc.profile` is that of the last
  entry.
- `progress=` receives the fraction of the whole series; returning True stops
  it with `CalculationInterrupted` and changes nothing.

## Settings records

`gsc.get_settings()` records the file path and SHA-256, the reader options,
the q grid, every parameter and the package versions.
`GenericScatteringCalculator.from_settings(settings)` rebuilds the calculator
and warns if the file or the periodictable version changed.

Records of version 0.1 replay with the new parameters at their defaults,
except the kernel: `'auto'` for pixels on a lattice and otherwise `'pairs'`,
the pair sum 0.1 used, so a record gives the curve it was made with.

A structure no file and options can rebuild (`from_arrays`, `from_atoms`, or
a `deuterated` / `with_symbols` copy) is recorded as `'in-memory'`, which
`from_settings` refuses; a derived copy also records `derived_from`.

## Worker processes

A settings record plus a series is plain JSON, so another process can run it:

```python
import sinccity
from sinccity import GenericScatteringCalculator

sinccity.set_verbosity('quiet')  # the logger is per process
profiles = GenericScatteringCalculator.from_settings(settings).calculate_series(series)
```
