# Structures

`gsc.load_structure(path, ...)` and `sinccity.load(path, ...)` read a file by
its extension and return a [`Structure`](../api/structures.md). Positions are
in Å and SLDs in Å⁻² in all four formats (a `.gro` file's nm are converted).

## File formats

| Format | What is read |
|---|---|
| PDB | `ATOM` records of the first model. `include_hetatm=True` adds `HETATM` records, never waters (`HOH`, `DOD`, `WAT` and the MD names `SOL`, `TIP3`, `TIP4`, `SPC`, `SPCE`, `OPC`). One alternate-location ID for the whole structure, occupancy not applied. Residue names from columns 18–21, where MD programs write four-letter names. Elements as below. Each element gets SasView's SLD and *natural atomic volume* from periodictable; an unknown element is an error. `CONECT` bonds are drawn. A `CRYST1` box is stored unless it is unreadable or a placeholder (a side of 1 Å or less). Residue numbers that are not decimal integers (hybrid-36) are renumbered in order. |
| GRO | All atoms, cut at GROMACS's fixed columns (names may touch numbers; `-ndec` files are read). Velocities are ignored; the box comes from the last line and is used to unwrap the aggregate. Elements as below. |
| SLD | One header line, then `x y z sld`, `x y z sld mx my mz` or `x y z sld mx my mz volume`. Magnetic columns are ignored; six-column magnetic-only files are refused. Without a volume column the points must form a lattice and each pixel's volume is the product of the steps. A sparse subset that skips planes looks like a coarser lattice, so give such files the volume column. |
| VTK | Legacy ASCII `UNSTRUCTURED_GRID` (classic or 5.1 layout) with tetrahedra, voxels, hexahedra, wedges and pyramids. The nuclear SLD is the one single-component `SCALARS` field, per cell or per point (averaged to cells as SasView does). Each cell becomes one point at its volume centroid. Binary files are refused with an export recipe. |

The readers follow SasView's with one deviation: the VTK cell geometry is
recomputed, because SasView's volumes for hexahedra, wedges and pyramids are
wrong. The GRO reader and the residue options come from coords2sans.

## Selecting residues

`load_structure(path, residues=['CTA'])` reads only the atoms of these residue
names, from PDB and GRO files. When given, it is the whole selection: the PDB
water and `HETATM` rules do not apply. A `.gro` file of a solvated system
needs it; without it the reader notes how many residue names it read.
Options a format does not have are ignored with a note. The structure keeps
the options it was read with (`structure.reader_options`), so
`get_settings()` records them however the structure was installed.

## Elements

For PDB and GRO files each residue name gets its elements from the first
source that has them:

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
(a run of atoms with one residue number and name), so the atom names of the
structure and the topology may differ. `structure.element_sources` records
the source per residue name. `structure.with_symbols(symbols)` returns a copy
with corrected elements, and `Structure.from_atoms(xyz, symbols, resids=,
resnames=, names=)` builds atoms that were read elsewhere.

## Periodic boxes

GROMACS writes every position back into the box, which can split molecules
and the aggregate. With `unwrap='auto'` (the default) a `.gro` file is
unwrapped: each molecule is made whole around its first atom, then moved by
whole box lengths next to the aggregate's centre, the circular mean of the
molecule centres. A note counts the atoms moved. It works for one aggregate
smaller than half the box. A `.gro` file whose box is zero (no periodic box)
is read as it is, with a note.

PDB files are unwrapped only with `unwrap=True`, because many programs write
a placeholder `CRYST1`; a missing or placeholder box is then an error.
Triclinic boxes cannot be unwrapped (`unwrap=False` reads such files without
the box).

## Deuteration

`structure.deuterated(['CTA'])` returns a copy in which the hydrogens of these
residue names have the SLD of D; volumes are unchanged, since D and H have the
same natural atomic volume. It needs a structure with residue names (PDB, GRO
or `Structure.from_atoms` with `resnames=`); a name that is not in the
structure, or whose atoms include no hydrogen, is an error. The calculator
applies the same change with `set_params(deuterated=['CTA'])`.
