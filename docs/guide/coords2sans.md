# Coming from coords2sans

`sinccity` takes over coords2sans's features; its names and units differ in
places.

| coords2sans | sinccity |
|---|---|
| `Structure.read(path, residues, masses, elements)` | `sinccity.load(path, residues=, masses=, elements=)` or `gsc.load_structure(...)` |
| `Structure(resids, resnames, names, positions, elements)` | `Structure.from_atoms(xyz, symbols, resids=, resnames=, names=)` |
| `structure.positions`, `structure.elements` (editable) | `structure.xyz`, `structure.symbols`; `structure.with_symbols(new)` |
| `structure.sld(deuterated=[...])` | `structure.deuterated([...]).sld`, or `set_params(deuterated=[...])` |
| `SansCalculator(solvent_sld=6.36e-6)` | `set_params(sld_solvent=6.36)`, in 10⁻⁶ Å⁻² |
| `q_min`, `q_max`, `q_points`, `q_log` | `set_q_grid(qmin, qmax, npoints, spacing='log' or 'linear')`; the default grid is SasView's, 3e-4 to 0.3 Å⁻¹ in 30 points |
| `polydispersity`, `size_points`, `SIZE_RANGE` | `size_pd`, `size_pd_n`, `size_pd_nsigma` |
| `average_amplitude=True` | `kernel='average'` |
| `BIN_WIDTH` (0.01 Å) | `histogram_bin`, by default at most 0.01 Å and smaller at high q or with a size spread |
| `calculate(structure, label)` → `Curve` | `set_structure(structure)`, then `calculate(label=)` → `ScatteringProfile` |
| `Curve.save`, `Curve.plot(ax)` | `profile.save_csv(path)` (the same two columns); `ax.plot(profile.q, profile.intensity)` or `profile.plot()` (plotly) |
| `SansTask` | `get_settings()` plus a series, run by `from_settings(settings).calculate_series(series)` |

Two rules differ on purpose. PDB files without an element column use
SasView's name rule (` CA ` is carbon), and in `.gro` files a two-letter
element name counts only in a one-atom residue (so `CA` in a protein is
carbon). V_norm is the volume of the atoms that scatter, which equals
coords2sans's volume of all atoms unless an atom matches the solvent.
