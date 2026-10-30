"""Residue-level steps shared by the PDB and GRO readers and by :class:`Structure`.

Unwrapping and element assignment follow coords2sans. A residue (a molecule)
is a run of atoms with one residue number.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .elements import MASS_TABLE, element_from_mass, guess_stem
from .model import StructureError


def residue_runs(resids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First index and atom count of each residue: a run of atoms with one residue number."""
    resids = np.asarray(resids)
    starts = np.flatnonzero(np.r_[True, resids[1:] != resids[:-1]])
    return starts, np.diff(np.r_[starts, resids.size])


def unwrap_box(resids: np.ndarray, xyz: np.ndarray, box) -> tuple[np.ndarray, int]:
    """Put an aggregate back together across the faces of its periodic box.

    GROMACS writes every position back inside the box, which can split
    molecules and the aggregate. Each molecule is made whole first: its atoms
    move to their images nearest its first atom. The aggregate's centre is the
    circular mean of the molecule centres along each axis, which a split does
    not move. Then each molecule moves by whole box lengths to its image
    nearest that centre. This works for one aggregate smaller than half the box.

    Returns:
        The positions, and the number of atoms that moved.
    """
    box = np.asarray(box, dtype=float)
    starts, sizes = residue_runs(resids)
    first = np.repeat(xyz[starts], sizes, axis=0)
    images = np.round((xyz - first) / box)
    whole = xyz - box * images
    centres = np.add.reduceat(whole, starts) / sizes[:, None]
    angle = 2 * np.pi * centres / box
    mean = np.arctan2(np.sin(angle).mean(0), np.cos(angle).mean(0))
    centre = (box / (2 * np.pi) * mean) % box  # folded into the box: a whole aggregate stays put
    images += np.repeat(np.round((centres - centre) / box), sizes, axis=0)
    return xyz - box * images, int(np.any(images != 0, axis=1).sum())


def assign_elements(
    resids: np.ndarray,
    resnames: np.ndarray,
    names: np.ndarray,
    masses: Mapping[str, Sequence[float]],
    elements: Mapping[str, Sequence[str]],
    file_elements: np.ndarray | None = None,
    guesses: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, str], str | None]:
    """Find each atom's element, per residue name, from the first source that has it.

    The sources, in order: ``masses``, ``elements``, the PDB element column
    (*file_elements*, used when filled for every atom of the residue), and a
    guess from the atom names (*guesses*, or :func:`guess_stem` when None).
    Masses and element lists are matched by the atom's position inside its
    residue, so structure and topology atom names may differ.

    Returns:
        The element of each atom, the source per residue name (``'masses'``,
        ``'elements'``, ``'pdb'`` or ``'names'``), and a note listing the
        residues whose elements were guessed, or None.

    Raises:
        StructureError: If a residue has another number of atoms than its
            list, a mass is no element's, or a guess is no element.
    """
    starts, sizes = residue_runs(resids)
    position = np.arange(len(resids)) - np.repeat(starts, sizes)
    size = np.repeat(sizes, sizes)
    hint = 'pass masses= or elements= for it'
    symbols = np.empty(len(resids), dtype=object)  # no fixed width to truncate a given symbol
    sources: dict[str, str] = {}
    for resname in np.unique(resnames).tolist():
        atoms = resnames == resname
        given = masses.get(resname, elements.get(resname))
        if given is not None and np.any(size[atoms] != len(given)):
            raise StructureError(
                f'A {resname} residue does not have the {len(given)} atoms given for it.'
            )
        if resname in masses:
            try:
                symbols[atoms] = element_from_mass(np.asarray(given, dtype=float)[position[atoms]])
            except StructureError as exc:
                raise StructureError(f'{resname}: {exc} Pass elements= for it.') from exc
            sources[resname] = 'masses'
        elif resname in elements:
            symbols[atoms] = [str(e).capitalize() for e in np.asarray(given)[position[atoms]]]
            sources[resname] = 'elements'
        elif file_elements is not None and np.all(file_elements[atoms] != ''):
            symbols[atoms], sources[resname] = file_elements[atoms], 'pdb'
        else:
            if guesses is None:
                guessed = [guess_stem(n, s == 1) for n, s in zip(names[atoms], size[atoms])]
            else:
                guessed = guesses[atoms]
            unknown = sorted({n for n, g in zip(names[atoms], guessed) if g not in MASS_TABLE})
            if unknown:
                raise StructureError(
                    f'No element for atom names of {resname} ({", ".join(unknown)}); {hint}.'
                )
            symbols[atoms], sources[resname] = guessed, 'names'

    symbols = symbols.astype(str)
    guessed = [name for name, source in sources.items() if source == 'names']
    if not guessed:
        return symbols, sources, None
    two = np.isin(resnames, guessed) & (np.char.str_len(symbols) == 2)
    pairs = sorted(
        {f'{n} -> {s} in {r}' for n, s, r in zip(names[two], symbols[two], resnames[two])}
    )
    note = f'Elements of {", ".join(guessed)} guessed from atom names'
    if pairs:
        note += f' (two-letter: {"; ".join(pairs)})'
    return symbols, sources, f'{note}; pass masses= or elements= to set them.'
