"""Element data, as SasView computes it, and the guesses of an element from an atom name."""

from functools import cache

import numpy as np

from .model import StructureError

#: Avogadro constant, 1/mol (the value SasView uses).
AVOGADRO = 6.02214129e23


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


def guess_pdb(line: str) -> str:
    """SasView's guess of the element from the atom-name columns 13-16 of a PDB record."""
    name = line[12:16].strip()
    if line[12].isdigit():
        return name[1]
    if len(name) == 4 or line[12] == ' ':
        return name[0]
    return name[:2]
