"""Element data, as SasView computes it, and the guesses of an element from an atom name."""

import string
from functools import cache

import numpy as np
import periodictable

from .model import StructureError

#: Avogadro constant, 1/mol (the value SasView uses).
AVOGADRO = 6.02214129e23

#: Mass in u of H to U (Z = 1 to 92) and of D. Any two are 0.0196 u apart or more.
MASS_TABLE = {
    element.symbol: element.mass
    for element in [*(e for e in periodictable.elements if 0 < e.number <= 92), periodictable.D]
}


@cache
def constants(symbol: str) -> tuple[float, float, float]:
    """SLD (Å⁻²), natural atomic volume (Å³) and mass (u) of an element.

    SasView's formulas: the coherent neutron SLD at the element's bulk density,
    and the volume of one atom at that density.

    Raises:
        StructureError: If periodictable has no finite SLD, density or mass for it.
    """
    from periodictable import formula, nsf

    try:
        atom = formula(symbol)
        # The volume first: without a density it raises a TypeError, while
        # neutron_sld fails an assert.
        volume = 1e24 * atom.mass / atom.density / AVOGADRO
        sld = nsf.neutron_sld(symbol)[0] * 1e-6
    except (ValueError, TypeError, AttributeError) as exc:
        raise StructureError(f'No neutron SLD or density for element {symbol!r}.') from exc
    if not (np.isfinite(sld) and np.isfinite(volume) and volume > 0):
        raise StructureError(f'No neutron SLD or density for element {symbol!r}.')
    return sld, volume, float(atom.mass)


def sld_volume(symbols) -> tuple[np.ndarray, np.ndarray]:
    """SLD (Å⁻²) and volume (Å³) per atom, from the element symbols.

    Raises:
        StructureError: Naming every symbol periodictable has no data for.
    """
    unique, index = np.unique(np.asarray(symbols, dtype=str), return_inverse=True)
    table, unknown = [], []
    for symbol in unique.tolist():
        try:
            table.append(constants(symbol)[:2])
        except StructureError:
            unknown.append(symbol)
    if unknown:
        raise StructureError(f'no neutron SLD or density for element(s) {", ".join(unknown)}.')
    table = np.array(table).reshape(-1, 2)
    return table[index, 0], table[index, 1]


def guess_pdb(line: str) -> str:
    """SasView's guess of the element from the atom-name columns 13-16 of a PDB record."""
    name = line[12:16].strip()
    if line[12].isdigit():
        return name[1]
    if len(name) == 4 or line[12] == ' ':
        return name[0]
    return name[:2]


def guess_stem(name: str, single_atom: bool) -> str:
    """The element of an atom name without a column convention (``.gro``).

    Leading digits are dropped. A two-letter stem that is an element (``CL``,
    ``NA``) is that element only in a residue of one atom, an ion; otherwise
    the element is the first letter, so ``CA`` in a protein is carbon. The
    result may be no element; the caller checks.

    Raises:
        StructureError: If the name has no letter after the leading digits.
    """
    stem = name.lstrip(string.digits)
    if not stem[:1].isalpha():
        raise StructureError(f'cannot guess an element from the atom name {name!r}.')
    if single_atom and len(stem) == 2 and stem.capitalize() in MASS_TABLE:
        return stem.capitalize()
    return stem[:1].upper()
