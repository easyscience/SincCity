"""Element data, as SasView computes it, and the guesses of an element from an atom name."""

import string
from functools import cache

import numpy as np
import periodictable
from periodictable import formula, nsf

from .model import StructureError

#: Avogadro constant, 1/mol (the value SasView uses).
AVOGADRO = 6.02214129e23

#: Mass in u of H to U (Z = 1 to 92) and of D. Any two are 0.0196 u apart or more;
#: heavier elements have whole-number masses that tie.
MASS_TABLE = {
    element.symbol: element.mass
    for element in [*(e for e in periodictable.elements if 0 < e.number <= 92), periodictable.D]
}

#: Largest gap in u between a topology mass and its element. Force fields round
#: masses by less than 0.005 u; a united-atom CH2 (14.027 u) lies 0.02 u from N
#: and must not pass.
MASS_TOLERANCE = 0.01


@cache
def constants(symbol: str) -> tuple[float, float, float]:
    """SLD (Å⁻²), natural atomic volume (Å³) and mass (u) of an element.

    SasView's formulas: the coherent neutron SLD at the element's bulk density,
    and the volume of one atom at that density.

    Raises:
        StructureError: If periodictable has no finite SLD, density or mass for it.
    """
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


def element_from_mass(masses) -> np.ndarray:
    """The element of each mass (u): the one of nearest mass in ``MASS_TABLE``.

    Raises:
        StructureError: If a mass lies more than ``MASS_TOLERANCE`` from every
            element, as with hydrogen mass repartitioning or united atoms.
    """
    masses = np.asarray(masses, dtype=float)
    symbols = np.array(list(MASS_TABLE))
    table = np.array(list(MASS_TABLE.values()))
    nearest = np.abs(masses[:, None] - table).argmin(axis=1)
    far = ~(np.abs(table[nearest] - masses) <= MASS_TOLERANCE)  # NaN is far too
    if far.any():
        raise StructureError(f"The mass {masses[far][0]:g} u is no element's.")
    return symbols[nearest]


def guess_pdb(line: str) -> str:
    """SasView's guess of the element from the atom-name columns 13-16 of a PDB record.

    Empty when the name is too short to guess from; the caller checks.
    """
    name = line[12:16].strip()
    if line[12].isdigit():
        return name[1:2]
    if len(name) == 4 or line[12] == ' ':
        return name[:1]
    return name[:2]


def guess_stem(name: str, single_atom: bool) -> str:
    """The element of an atom name without a column convention (``.gro``).

    Leading digits are dropped. A two-letter stem that is an element (``CL``,
    ``NA``) is that element only in a residue of one atom, an ion; otherwise
    the element is the first letter, so ``CA`` in a protein is carbon. The
    result may be no element, or empty; the caller checks.
    """
    stem = name.lstrip(string.digits)
    if single_atom and len(stem) == 2 and stem.capitalize() in MASS_TABLE:
        return stem.capitalize()
    return stem[:1].upper()
